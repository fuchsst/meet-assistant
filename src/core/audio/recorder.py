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
import yaml
import threading
import time
from collections import deque

from config.config import AUDIO_CONFIG, AUDIO_DEVICES_CONFIG

logger = logging.getLogger(__name__)

class MultiSourceRecorder:
    def __init__(self, chunk_callback: Optional[Callable[[bytes, str], None]] = None):
        """Initialize the multi-source recorder."""
        self.audio = pyaudio.PyAudio()
        self.streams: Dict[str, pyaudio.Stream] = {}
        self.output_recorders: Dict[str, sc.Microphone] = {}
        self.frames: Dict[str, List[bytes]] = {}
        self.is_recording: bool = False
        self.threads: List[threading.Thread] = []
        self.chunk_callback = chunk_callback
        
        # Use settings from config with optimized values
        self.format = pyaudio.paFloat32  # Higher precision format
        self.channels = AUDIO_CONFIG["channels"]
        self.rate = AUDIO_CONFIG["rate"]
        self.chunk = AUDIO_CONFIG["chunk"]  # Use base chunk size to minimize echo
        self.sample_width = AUDIO_CONFIG["sample_width"]
        
        # Balanced latency for better performance
        self.input_latency = 0.015  # 15ms input latency
        self.output_latency = 0.015  # 15ms output latency
        
        # Session tracking for file naming
        self.current_session = None
        self.session_counts = {}
        
        # Load device configuration
        self.load_device_config()
        
        # Initialize dictionaries for state tracking
        self.filter_states = {}
        self.dc_offset_states = {}
        self.overlap_buffers = {}
        self.current_gains = {}
        self.noise_floors = {}
        self.noise_frame_counts = {}
        self.last_noise_update = {}
        self.peak_levels = {}
        self.rms_levels = {}
        self.echo_buffers = {}  # For echo cancellation
        
        # Initialize audio processing components with optimized parameters
        self._init_audio_processing()

    def _init_audio_processing(self):
        """Initialize audio processing components with optimized parameters."""
        # Initialize filter with balanced transition bands
        self._init_filter()
        
        # Balanced overlap for smooth transitions without artifacts
        self.overlap_size = self.chunk // 6  # Moderate overlap size
        
        # Balanced window function for smooth transitions
        self.fade_in = signal.windows.tukey(self.overlap_size * 2, alpha=0.25)[:self.overlap_size]
        self.fade_out = signal.windows.tukey(self.overlap_size * 2, alpha=0.25)[self.overlap_size:]
        
        # Balanced gain staging parameters
        self.target_rms = 0.175  # Moderate target RMS
        self.max_gain = 2.5  # Moderate max gain
        self.gain_smoothing = 0.92  # Balanced smoothing
        
        # Moderate noise reduction
        self.noise_reduction_strength = 0.25  # Balanced strength
        self.min_noise_frames = 40  # Moderate frame count
        self.noise_update_rate = 0.015  # Balanced update rate
        self.noise_floor_min_update_interval = 1.5  # Moderate interval
        
        # Moderate echo cancellation parameters
        self.echo_delay = int(0.035 * self.rate)  # 35ms echo delay
        self.echo_decay = 0.4  # Moderate echo decay
        
        # Balanced DC offset removal
        self.dc_offset_alpha = 0.992  # Moderate response

    def _init_filter(self):
        """Initialize balanced audio filters optimized for voice."""
        nyquist = self.rate * 0.5
        
        # Balanced frequency bands
        speech_low = 175.0    # Hz - Moderate low cutoff
        speech_high = min(5800.0, nyquist * 0.92)  # Moderate high cutoff
        
        # Normalize frequencies
        self.speech_low = speech_low / nyquist
        self.speech_high = speech_high / nyquist
        
        # Moderate filter slopes
        self.filter_b, self.filter_a = signal.butter(2, [self.speech_low, self.speech_high], btype='bandpass')
        
        # Balanced additional filters
        self.dc_filter_b, self.dc_filter_a = signal.butter(2, 50/nyquist, btype='highpass')
        self.deess_b, self.deess_a = signal.butter(2, 5800/nyquist, btype='lowpass')

    def _update_noise_floor(self, device_id: str, audio_data: np.ndarray, current_time: float):
        """Update noise floor estimate with balanced temporal and spectral constraints."""
        if device_id not in self.last_noise_update:
            self.last_noise_update[device_id] = 0
            self.noise_floors[device_id] = None
            self.noise_frame_counts[device_id] = 0
            
        if current_time - self.last_noise_update[device_id] >= self.noise_floor_min_update_interval:
            frame_energy = np.sqrt(np.mean(np.square(audio_data)))
            
            if self.noise_floors[device_id] is None:
                self.noise_floors[device_id] = np.abs(audio_data)
                self.noise_frame_counts[device_id] = 1
            else:
                if frame_energy < (np.mean(np.abs(self.noise_floors[device_id])) * 1.15):  # Balanced threshold
                    if self.noise_frame_counts[device_id] < self.min_noise_frames:
                        self.noise_floors[device_id] = (
                            (1 - self.noise_update_rate) * self.noise_floors[device_id] +
                            self.noise_update_rate * np.abs(audio_data)
                        )
                        self.noise_frame_counts[device_id] += 1
                    else:
                        self.noise_floors[device_id] = (
                            (1 - self.noise_update_rate * 0.15) * self.noise_floors[device_id] +
                            self.noise_update_rate * 0.15 * np.abs(audio_data)
                        )
            
            self.last_noise_update[device_id] = current_time

    def _apply_gain_staging(self, audio_data: np.ndarray, device_id: str) -> np.ndarray:
        """Apply balanced gain staging with smooth transitions."""
        if device_id not in self.current_gains:
            self.current_gains[device_id] = 1.0
            
        current_rms = np.sqrt(np.mean(np.square(audio_data)))
        
        if current_rms > 0:
            desired_gain = self.target_rms / current_rms
            desired_gain = np.clip(desired_gain, 1/self.max_gain, self.max_gain)
            
            # Smooth gain changes
            self.current_gains[device_id] = (
                self.gain_smoothing * self.current_gains[device_id] +
                (1 - self.gain_smoothing) * desired_gain
            )
        
        # Apply gain with soft clipping
        audio_data = audio_data * self.current_gains[device_id]
        audio_data = np.tanh(audio_data)  # Soft clipping
            
        return audio_data

    def _remove_dc_offset(self, audio_data: np.ndarray, device_id: str) -> np.ndarray:
        """Remove DC offset using balanced high-pass filter."""
        if device_id not in self.dc_offset_states:
            self.dc_offset_states[device_id] = signal.lfilter_zi(self.dc_filter_b, self.dc_filter_a)
            
        audio_data, self.dc_offset_states[device_id] = signal.lfilter(
            self.dc_filter_b,
            self.dc_filter_a,
            audio_data,
            zi=self.dc_offset_states[device_id]
        )
        
        return audio_data

    def _apply_echo_cancellation(self, audio_data: np.ndarray, device_id: str) -> np.ndarray:
        """Apply moderate echo cancellation using delay buffer."""
        if device_id not in self.echo_buffers:
            self.echo_buffers[device_id] = deque(maxlen=self.echo_delay)
            # Initialize buffer with zeros
            for _ in range(self.echo_delay):
                self.echo_buffers[device_id].append(np.zeros_like(audio_data))
        
        # Get delayed signal
        delayed_signal = np.array(self.echo_buffers[device_id][0])
        
        # Update buffer
        self.echo_buffers[device_id].append(audio_data.copy())
        
        # Subtract echo estimate with moderate decay
        audio_data = audio_data - self.echo_decay * delayed_signal
        
        return audio_data

    def _process_audio(self, audio_data: np.ndarray, device_id: str) -> np.ndarray:
        """Process audio data with balanced noise reduction and echo cancellation."""
        try:
            # Initialize or reset filter states if needed
            if device_id not in self.filter_states:
                self._init_device_states(device_id)
            elif len(audio_data) != self.chunk:
                # Reinitialize states if chunk size changes
                self._init_device_states(device_id)
            
            # Remove DC offset
            audio_data = self._remove_dc_offset(audio_data, device_id)
            
            # Apply moderate echo cancellation
            audio_data = self._apply_echo_cancellation(audio_data, device_id)
            
            # Apply balanced gain staging
            audio_data = self._apply_gain_staging(audio_data, device_id)
            
            # Update noise floor
            current_time = time.time()
            self._update_noise_floor(device_id, audio_data, current_time)
            
            # Apply main bandpass filter with state reset prevention
            filtered_data, new_state = signal.lfilter(
                self.filter_b,
                self.filter_a,
                audio_data,
                zi=self.filter_states[device_id]
            )
            
            # Only update state if filter was successful
            if not np.any(np.isnan(new_state)):
                self.filter_states[device_id] = new_state
            else:
                # Reset state if it became invalid
                self.filter_states[device_id] = signal.lfilter_zi(self.filter_b, self.filter_a)
                filtered_data = signal.lfilter(self.filter_b, self.filter_a, audio_data)[0]
            
            # Enhanced noise reduction with spectral subtraction
            if device_id in self.noise_floors and self.noise_floors[device_id] is not None:
                signal_energy = np.abs(filtered_data)
                noise_energy = self.noise_floors[device_id]
                
                # Compute noise mask with balanced transition
                noise_mask = np.maximum(0, signal_energy - noise_energy * self.noise_reduction_strength)
                noise_mask = noise_mask / (signal_energy + 1e-10)
                
                # Apply mask with smooth transition
                filtered_data = filtered_data * noise_mask
            
            # Handle overlap with smooth crossfade
            if self.overlap_buffers[device_id] is not None:
                overlap_region = np.zeros(self.overlap_size)
                overlap_region += self.overlap_buffers[device_id] * self.fade_out
                overlap_region += filtered_data[:self.overlap_size] * self.fade_in
                filtered_data[:self.overlap_size] = overlap_region
            
            # Save overlap buffer for next chunk
            self.overlap_buffers[device_id] = filtered_data[-self.overlap_size:].copy()
            
            # Update level monitoring
            self.peak_levels[device_id] = np.max(np.abs(filtered_data))
            self.rms_levels[device_id] = np.sqrt(np.mean(np.square(filtered_data)))
            
            # Convert to int16 for saving
            filtered_data = np.clip(filtered_data * 32767, -32768, 32767).astype(np.int16)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error processing audio for device {device_id}: {e}")
            return audio_data

    def _init_device_states(self, device_id: str):
        """Initialize or reset filter states for a device."""
        self.filter_states[device_id] = signal.lfilter_zi(self.filter_b, self.filter_a)
        self.dc_offset_states[device_id] = signal.lfilter_zi(self.dc_filter_b, self.dc_filter_a)
        self.overlap_buffers[device_id] = None
        self.current_gains[device_id] = 1.0
        self.noise_floors[device_id] = None
        self.noise_frame_counts[device_id] = 0
        self.last_noise_update[device_id] = 0

    def _input_callback(self, in_data, frame_count, time_info, status, device_id: str):
        """Process audio data from input device with non-blocking stream."""
        if status:
            logger.warning(f"Stream status for device {device_id}: {status}")
        
        if self.is_recording:
            try:
                # Convert to numpy array (float32)
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # Process audio
                processed_data = self._process_audio(audio_data, device_id)
                
                # Store processed data
                processed_bytes = processed_data.tobytes()
                self.frames[device_id].append(processed_bytes)
                
                # Call chunk callback if provided
                if self.chunk_callback:
                    self.chunk_callback(processed_bytes, device_id)
                
            except Exception as e:
                logger.error(f"Error in input callback for device {device_id}: {e}")
        
        return (None, pyaudio.paContinue)

    def _record_output_device(self, device_config: dict):
        """Record from an output device using soundcard with optimized buffer handling."""
        device_id = device_config['name']
        try:
            mic = sc.get_microphone(id=str(device_config['id']), include_loopback=True)
            self.output_recorders[device_id] = mic
            
            # Calculate optimal sleep time based on chunk size and sample rate
            chunk_duration = self.chunk / self.rate
            sleep_time = max(0.001, chunk_duration * 0.75)  # 75% of chunk duration, min 1ms
            
            while self.is_recording:
                try:
                    start_time = time.time()
                    
                    # Use same chunk size as input devices to maintain consistency
                    data = mic.record(numframes=self.chunk, samplerate=self.rate)
                    
                    # Convert to mono if needed
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    
                    # Process audio
                    processed_data = self._process_audio(data, device_id)
                    
                    # Store processed data
                    processed_bytes = processed_data.tobytes()
                    self.frames[device_id].append(processed_bytes)
                    
                    # Call chunk callback if provided
                    if self.chunk_callback:
                        data_bytes = data.astype(np.float32).tobytes()
                        self.chunk_callback(data_bytes, device_id)
                    
                    # Calculate remaining sleep time to maintain consistent timing
                    elapsed = time.time() - start_time
                    remaining_sleep = max(0.0, sleep_time - elapsed)
                    if remaining_sleep > 0:
                        time.sleep(remaining_sleep)
                    
                except Exception as e:
                    logger.error(f"Error recording from output device {device_id}: {e}")
                    time.sleep(0.001)  # Minimal sleep on error
                
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
        """Load audio device configuration from config file."""
        try:
            if AUDIO_DEVICES_CONFIG.exists():
                with open(AUDIO_DEVICES_CONFIG, 'r') as f:
                    self.device_config = yaml.safe_load(f)
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
        """Start recording from all configured devices with optimized settings."""
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
                    
                    # Open stream with non-blocking callback and optimized buffer size
                    stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=device_rate,
                        input=True,
                        input_device_index=device['index'],
                        frames_per_buffer=self.chunk,
                        stream_callback=lambda *args, d=device_id: self._input_callback(*args, d),
                        start=False  # Don't start immediately
                    )
                    
                    # Configure stream parameters for better performance
                    stream.start_stream()
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
        """Stop recording from all devices and clean up resources."""
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

    def _get_output_filepath(self, meeting_dir: Path, device_id: str, session_title: str) -> Path:
        """Get output filepath with proper indexing for multiple recordings."""
        if session_title != self.current_session:
            self.current_session = session_title
            self.session_counts = {}
        
        safe_name = "".join(c if c.isalnum() else "_" for c in device_id)
        base_filename = f"audio_{safe_name}"
        
        # Get current count for this device
        count = self.session_counts.get(device_id, 0)
        
        # Create filename with index if needed
        if count > 0:
            filename = f"{base_filename}_{count:02d}.wav"
        else:
            filename = f"{base_filename}.wav"
        
        filepath = meeting_dir / filename
        
        # Increment count if file exists and we're not doing continuous save
        while filepath.exists():
            count += 1
            self.session_counts[device_id] = count
            filename = f"{base_filename}_{count:02d}.wav"
            filepath = meeting_dir / filename
        
        return filepath

    def save_recordings(self, meeting_dir: Path, session_title: str, continuous: bool = False) -> Dict[str, Path]:
        """Save all recorded audio to separate files with proper format settings."""
        saved_files = {}
        
        for device_id, frames in self.frames.items():
            if not frames:
                logger.warning(f"No audio data to save for device {device_id}")
                continue
            
            try:
                if continuous:
                    # For continuous saving, use base filename without index
                    safe_name = "".join(c if c.isalnum() else "_" for c in device_id)
                    filepath = meeting_dir / f"audio_{safe_name}.wav"
                    
                    if filepath.exists():
                        # Read existing file and append new frames
                        with wave.open(str(filepath), 'rb') as existing_wav:
                            # Verify format matches
                            if (existing_wav.getnchannels() != self.channels or
                                existing_wav.getsampwidth() != self.sample_width or
                                existing_wav.getframerate() != self.rate):
                                logger.error(f"Format mismatch for {device_id}, creating new file")
                                continuous = False
                            else:
                                # Read existing frames
                                existing_frames = existing_wav.readframes(existing_wav.getnframes())
                                # Combine with new frames
                                frames = [existing_frames] + frames
                
                if not continuous:
                    # Get new filepath with index if needed
                    filepath = self._get_output_filepath(meeting_dir, device_id, session_title)
                
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Save audio file with proper format settings
                with wave.open(str(filepath), 'wb') as wave_file:
                    wave_file.setnchannels(self.channels)
                    wave_file.setsampwidth(self.sample_width)
                    wave_file.setframerate(self.rate)
                    wave_file.writeframes(b''.join(frames))
                
                saved_files[device_id] = filepath
                logger.info(f"Saved audio for device {device_id} to {filepath}")
                
                # Clear frames after saving if not continuous
                if not continuous:
                    self.frames[device_id] = []
                
            except Exception as e:
                logger.error(f"Failed to save audio for device {device_id}: {e}")
        
        return saved_files

    def validate_recordings(self) -> Dict[str, bool]:
        """Validate all recorded audio data with enhanced checks."""
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
                if rms < AUDIO_CONFIG["silence_threshold"]:
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
