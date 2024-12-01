"""Audio recording functionality using PyAudio and SoundCard."""
import wave
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Callable
import pyaudio
import numpy as np
from scipy import signal
import audioop
import soundcard as sc
import json
import threading
import time

from config.config import AUDIO_CONFIG, AUDIO_DEVICES_CONFIG

logger = logging.getLogger(__name__)

class MultiSourceRecorder:
    """Handles recording from multiple audio sources simultaneously."""
    
    def __init__(self, chunk_callback: Optional[Callable[[bytes, str], None]] = None):
        """Initialize the multi-source recorder.
        
        Args:
            chunk_callback: Optional callback function to process audio chunks in real-time.
                          Takes (chunk: bytes, device_id: str) as arguments.
        """
        self.audio = pyaudio.PyAudio()
        self.streams: Dict[str, pyaudio.Stream] = {}
        self.output_recorders: Dict[str, sc.Microphone] = {}
        self.frames: Dict[str, List[bytes]] = {}
        self.is_recording: bool = False
        self.threads: List[threading.Thread] = []
        self.chunk_callback = chunk_callback
        
        # Audio settings from config
        self.format = pyaudio.paFloat32
        self.channels = AUDIO_CONFIG["channels"]
        self.rate = AUDIO_CONFIG["rate"]
        self.chunk = AUDIO_CONFIG["chunk"]
        self.sample_width = AUDIO_CONFIG["sample_width"]
        
        # Load device configuration
        self.load_device_config()
        
        # Initialize audio processing components
        self._init_audio_processing()

    def _init_audio_processing(self):
        """Initialize audio processing components."""
        # Initialize filter
        self._init_filter()
        self.filter_states = {}
        
        # Initialize overlap handling
        self.overlap_size = self.chunk // 8
        self.overlap_buffers = {}
        
        # Create Hann window for smooth crossfade
        self.fade_in = np.hanning(self.overlap_size * 2)[:self.overlap_size]
        self.fade_out = np.hanning(self.overlap_size * 2)[self.overlap_size:]
        
        # Enhanced VAD settings with dynamic thresholds
        self.vad_base_threshold = 400  # Base energy threshold
        self.vad_frame_histories = {}
        self.vad_history_size = 20  # Increased for better context
        self.silence_durations = {}
        self.max_silence_duration = 0.7  # Slightly reduced
        self.min_speech_duration = 0.25  # Minimum duration for valid speech
        self.energy_ratio_threshold = 1.8  # Ratio for dynamic threshold
        
        # Speech probability tracking
        self.speech_probs = {}
        self.speech_prob_threshold = 0.65
        self.speech_prob_alpha = 0.12  # Smoothing factor
        self.speech_history_size = 10
        self.speech_histories = {}
        
        # Enhanced noise reduction
        self.noise_floors = {}
        self.noise_reduction_strength = 0.85
        self.noise_frame_counts = {}
        self.min_noise_frames = 20  # Increased for better estimation
        self.noise_update_rate = 0.04
        self.noise_floor_min_update_interval = 0.5  # Seconds
        self.last_noise_update = {}
        
        # Spectral gating parameters
        self.freq_mask_smooth_hz = 100
        self.n_std_thresh_stationary = 1.5
        self.temp_window_size = 0.2  # seconds
        
        # RMS history for dynamic threshold adjustment
        self.rms_histories = {}
        self.rms_history_size = 50
        self.rms_threshold_ratio = 0.15

    def _init_filter(self):
        """Initialize enhanced audio filters."""
        nyquist = self.rate * 0.5
        
        # Main bandpass filter focused on speech frequencies
        speech_low = 85.0   # Hz - capture some bass for voice
        speech_high = 8000.0  # Hz - preserve consonants
        
        # Normalize frequencies
        self.speech_low = speech_low / nyquist
        self.speech_high = speech_high / nyquist
        
        # Steeper filter slopes for better isolation
        self.filter_b, self.filter_a = signal.butter(8, [self.speech_low, self.speech_high], btype='band')
        
        # Additional filters
        self.noise_b, self.noise_a = signal.butter(6, 120/nyquist, btype='high')  # Remove low rumble
        self.emphasis_b, self.emphasis_a = signal.butter(4, 3000/nyquist, btype='low')  # De-emphasis

    def _calculate_dynamic_vad_threshold(self, device_id: str, current_energy: float) -> float:
        """Calculate dynamic VAD threshold based on signal history."""
        if device_id not in self.rms_histories:
            self.rms_histories[device_id] = []
        
        # Update RMS history
        self.rms_histories[device_id].append(current_energy)
        if len(self.rms_histories[device_id]) > self.rms_history_size:
            self.rms_histories[device_id].pop(0)
        
        # Calculate dynamic threshold
        if len(self.rms_histories[device_id]) > 0:
            mean_energy = np.mean(self.rms_histories[device_id])
            std_energy = np.std(self.rms_histories[device_id])
            dynamic_threshold = mean_energy + (std_energy * self.n_std_thresh_stationary)
            return max(self.vad_base_threshold, dynamic_threshold)
        
        return self.vad_base_threshold

    def _update_speech_probability(self, device_id: str, is_speech: bool) -> float:
        """Update and return speech probability using temporal smoothing."""
        if device_id not in self.speech_histories:
            self.speech_histories[device_id] = []
        
        # Update speech history
        self.speech_histories[device_id].append(1.0 if is_speech else 0.0)
        if len(self.speech_histories[device_id]) > self.speech_history_size:
            self.speech_histories[device_id].pop(0)
        
        # Calculate smoothed probability
        if len(self.speech_histories[device_id]) > 0:
            current_prob = np.mean(self.speech_histories[device_id])
            
            if device_id not in self.speech_probs:
                self.speech_probs[device_id] = current_prob
            else:
                # Exponential smoothing
                self.speech_probs[device_id] = (
                    (1 - self.speech_prob_alpha) * self.speech_probs[device_id] +
                    self.speech_prob_alpha * current_prob
                )
            
            return self.speech_probs[device_id]
        
        return 0.0

    def _update_noise_floor(self, device_id: str, audio_data: np.ndarray, current_time: float):
        """Update noise floor estimate with temporal constraints."""
        if device_id not in self.last_noise_update:
            self.last_noise_update[device_id] = 0
        
        # Check if enough time has passed since last update
        if current_time - self.last_noise_update[device_id] >= self.noise_floor_min_update_interval:
            if device_id not in self.noise_floors:
                self.noise_floors[device_id] = np.abs(audio_data)
                self.noise_frame_counts[device_id] = 1
            else:
                # Update noise floor with current frame
                if self.noise_frame_counts[device_id] < self.min_noise_frames:
                    self.noise_floors[device_id] = (
                        (1 - self.noise_update_rate) * self.noise_floors[device_id] +
                        self.noise_update_rate * np.abs(audio_data)
                    )
                    self.noise_frame_counts[device_id] += 1
                else:
                    # Selective update based on energy level
                    current_energy = np.mean(np.abs(audio_data))
                    noise_energy = np.mean(self.noise_floors[device_id])
                    
                    if current_energy < noise_energy * 1.5:  # Only update if likely noise
                        self.noise_floors[device_id] = (
                            (1 - self.noise_update_rate) * self.noise_floors[device_id] +
                            self.noise_update_rate * np.abs(audio_data)
                        )
            
            self.last_noise_update[device_id] = current_time

    def _process_audio(self, audio_data: np.ndarray, device_id: str) -> np.ndarray:
        """Process audio data with enhanced noise reduction and VAD."""
        try:
            # Initialize states if needed
            if device_id not in self.filter_states:
                self.filter_states[device_id] = signal.lfilter_zi(self.filter_b, self.filter_a)
                self.overlap_buffers[device_id] = None
                self.vad_frame_histories[device_id] = []
                self.silence_durations[device_id] = 0
            
            # Update noise floor
            current_time = time.time()
            self._update_noise_floor(device_id, audio_data, current_time)
            
            # Apply main bandpass filter
            filtered_data, self.filter_states[device_id] = signal.lfilter(
                self.filter_b, self.filter_a,
                audio_data,
                zi=self.filter_states[device_id] * audio_data[0] if audio_data.size > 0 else self.filter_states[device_id]
            )
            
            # Calculate current frame energy
            frame_energy = np.sqrt(np.mean(np.square(filtered_data)))
            
            # Get dynamic VAD threshold
            vad_threshold = self._calculate_dynamic_vad_threshold(device_id, frame_energy)
            
            # Determine if frame contains speech
            is_speech = frame_energy > vad_threshold
            
            # Update speech probability
            speech_prob = self._update_speech_probability(device_id, is_speech)
            
            # Update silence duration
            if speech_prob < self.speech_prob_threshold:
                self.silence_durations[device_id] += len(audio_data) / self.rate
            else:
                self.silence_durations[device_id] = 0
            
            # Apply noise reduction if in silence
            if self.silence_durations[device_id] > self.max_silence_duration:
                if self.noise_floors[device_id] is not None:
                    # Enhanced noise reduction during silence
                    filtered_data -= (self.noise_floors[device_id] * self.noise_reduction_strength)
                    filtered_data = np.clip(filtered_data, -1.0, 1.0)
            
            # Handle overlap with smooth crossfade
            if self.overlap_buffers[device_id] is not None:
                overlap_region = np.zeros(self.overlap_size)
                overlap_region += self.overlap_buffers[device_id] * self.fade_out
                overlap_region += filtered_data[:self.overlap_size] * self.fade_in
                filtered_data[:self.overlap_size] = overlap_region
            
            # Save overlap buffer for next chunk
            self.overlap_buffers[device_id] = filtered_data[-self.overlap_size:].copy()
            
            # Convert to int16 for saving
            filtered_data = np.clip(filtered_data * 32767, -32768, 32767).astype(np.int16)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error processing audio for device {device_id}: {e}")
            return audio_data

    def _input_callback(self, in_data, frame_count, time_info, status, device_id: str):
        """Process audio data from input device."""
        if status:
            logger.warning(f"Stream status for device {device_id}: {status}")
        
        if self.is_recording:
            try:
                # Convert to numpy array (float32)
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # Process audio (converts to int16 internally)
                processed_data = self._process_audio(audio_data, device_id)
                
                # Store processed data
                processed_bytes = processed_data.tobytes()
                self.frames[device_id].append(processed_bytes)
                
                # Call chunk callback if provided
                if self.chunk_callback:
                    self.chunk_callback(in_data, device_id)
                
            except Exception as e:
                logger.error(f"Error in input callback for device {device_id}: {e}")
        
        return (None, pyaudio.paContinue)

    def _record_output_device(self, device_config: dict):
        """Record from an output device using soundcard."""
        device_id = device_config['name']
        try:
            mic = sc.get_microphone(id=str(device_config['id']), include_loopback=True)
            self.output_recorders[device_id] = mic
            
            while self.is_recording:
                try:
                    # Record audio
                    data = mic.record(numframes=self.chunk, samplerate=self.rate)
                    
                    # Convert to mono if needed
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    
                    # Process audio (converts to int16 internally)
                    processed_data = self._process_audio(data, device_id)
                    
                    # Store processed data
                    processed_bytes = processed_data.tobytes()
                    self.frames[device_id].append(processed_bytes)
                    
                    # Call chunk callback if provided
                    if self.chunk_callback:
                        # Convert data to bytes in the same format as input devices
                        data_bytes = data.astype(np.float32).tobytes()
                        self.chunk_callback(data_bytes, device_id)
                    
                except Exception as e:
                    logger.error(f"Error recording from output device {device_id}: {e}")
                    
                time.sleep(0.001)  # Small delay to prevent busy waiting
                
        except Exception as e:
            logger.error(f"Failed to record from output device {device_id}: {e}")

    def get_supported_sample_rates(self, device_index):
        """Test which standard sample rates are supported by the specified device."""
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []
        device_info = self.audio.get_device_info_by_index(device_index)
        max_channels = device_info.get('maxInputChannels')
        
        for rate in standard_rates:
            try:
                if self.audio.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=max_channels,
                    input_format=self.format
                ):
                    supported_rates.append(rate)
            except:
                continue
        return supported_rates

    def _get_best_sample_rate(self, device_index, desired_rate):
        """Determines the best available sample rate for the device."""
        try:
            device_info = self.audio.get_device_info_by_index(device_index)
            supported_rates = self.get_supported_sample_rates(device_index)
            
            if desired_rate in supported_rates:
                return desired_rate
            
            # If desired rate not supported, get closest available rate
            if supported_rates:
                return min(supported_rates, key=lambda x: abs(x - desired_rate))
            
            return int(device_info.get('defaultSampleRate', 44100))
        except Exception as e:
            logger.warning(f"Error determining sample rate: {e}")
            return 44100  # Safe fallback

    def load_device_config(self):
        """Load audio device configuration."""
        try:
            if AUDIO_DEVICES_CONFIG.exists():
                with open(AUDIO_DEVICES_CONFIG, 'r') as f:
                    self.device_config = json.load(f)
                    # Ensure record_output exists
                    if 'record_output' not in self.device_config:
                        self.device_config['record_output'] = True
            else:
                self.device_config = {
                    "input_devices": [],
                    "output_devices": [],
                    "record_output": True
                }
                logger.warning(f"Device config not found at {AUDIO_DEVICES_CONFIG}")
        except Exception as e:
            logger.error(f"Failed to load device config: {e}")
            self.device_config = {
                "input_devices": [],
                "output_devices": [],
                "record_output": True
            }

    def start_recording(self, meeting_dir: Path) -> None:
        """Start recording from all configured devices."""
        if self.is_recording:
            logger.warning("Recording is already in progress")
            return

        try:
            self.is_recording = True
            self.frames.clear()
            self.threads.clear()
            
            # Start input device recording
            for device in self.device_config["input_devices"]:
                device_id = device['name']
                self.frames[device_id] = []
                
                try:
                    # Get best sample rate for device
                    device_rate = self._get_best_sample_rate(device['index'], self.rate)
                    
                    stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=device_rate,
                        input=True,
                        input_device_index=device['index'],
                        frames_per_buffer=self.chunk,
                        stream_callback=lambda *args, d=device_id: self._input_callback(*args, d)
                    )
                    self.streams[device_id] = stream
                    logger.info(f"Started recording from input device: {device_id}")
                except Exception as e:
                    logger.error(f"Failed to start recording from input device {device_id}: {e}")
            
            # Start output device recording only if enabled
            if self.device_config.get('record_output', True):
                for device in self.device_config["output_devices"]:
                    device_id = device['name']
                    self.frames[device_id] = []
                    
                    thread = threading.Thread(
                        target=self._record_output_device,
                        args=(device,),
                        daemon=True
                    )
                    thread.start()
                    self.threads.append(thread)
                    logger.info(f"Started recording from output device: {device_id}")
            else:
                logger.info("Output recording is disabled")
            
            logger.info("Started recording from all devices")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.stop_recording()
            raise

    def stop_recording(self) -> None:
        """Stop recording from all devices."""
        if not self.is_recording:
            return

        logger.info("Stopping recording from all devices")
        self.is_recording = False
        
        # Stop input streams
        for device_id, stream in self.streams.items():
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream for device {device_id}: {e}")
        
        self.streams.clear()
        
        # Wait for output recording threads
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        self.threads.clear()
        self.output_recorders.clear()

    def save_recordings(self, meeting_dir: Path) -> Dict[str, Path]:
        """Save all recorded audio to separate files."""
        saved_files = {}
        
        for device_id, frames in self.frames.items():
            if not frames:
                logger.warning(f"No audio data to save for device {device_id}")
                continue
            
            try:
                # Create device-specific filename
                safe_name = "".join(c if c.isalnum() else "_" for c in device_id)
                filepath = meeting_dir / f"audio_{safe_name}.wav"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Save audio file
                with wave.open(str(filepath), 'wb') as wave_file:
                    wave_file.setnchannels(self.channels)
                    wave_file.setsampwidth(self.sample_width)
                    wave_file.setframerate(self.rate)
                    wave_file.writeframes(b''.join(frames))
                
                saved_files[device_id] = filepath
                logger.info(f"Saved audio for device {device_id} to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save audio for device {device_id}: {e}")
        
        return saved_files

    def validate_recordings(self) -> Dict[str, bool]:
        """Validate all recorded audio data."""
        results = {}
        
        for device_id, frames in self.frames.items():
            if not frames:
                results[device_id] = False
                continue
            
            try:
                # Convert frames to numpy array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                
                # Check for silence
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                if rms < 100:
                    logger.warning(f"Audio from {device_id} appears to be silent")
                    results[device_id] = False
                    continue
                
                # Check for clipping
                if np.abs(audio_data).max() > 32700:
                    logger.warning(f"Audio from {device_id} contains clipping")
                    results[device_id] = False
                    continue
                
                # Calculate signal-to-noise ratio
                noise_floor = np.percentile(np.abs(audio_data), 10)
                signal_peak = np.percentile(np.abs(audio_data), 90)
                snr = 20 * np.log10(signal_peak / (noise_floor + 1e-10))
                
                if snr < 10:
                    logger.warning(f"Poor signal-to-noise ratio for {device_id}: {snr:.1f} dB")
                    results[device_id] = False
                    continue
                
                results[device_id] = True
                logger.info(f"Audio validation passed for {device_id} (SNR: {snr:.1f} dB)")
                
            except Exception as e:
                logger.error(f"Audio validation failed for {device_id}: {e}")
                results[device_id] = False
        
        return results

    def __del__(self):
        """Clean up resources."""
        self.stop_recording()
        if hasattr(self, 'audio'):
            self.audio.terminate()
