"""Continuous audio monitoring and transcription."""
import logging
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import pyaudio
import soundcard as sc
import wave
from scipy import signal

from config.config import AUDIO_CONFIG, WHISPER_CONFIG
from src.core.transcription.whisper_processor import WhisperProcessor

logger = logging.getLogger(__name__)

class AudioMonitor:
    """Monitors and transcribes audio input and output streams."""
    
    def __init__(self, language: str = WHISPER_CONFIG["language"]):
        """Initialize audio monitoring components."""
        # Add debug counters
        self.processed_chunks = 0
        self.saved_chunks = 0
        self.transcribed_chunks = 0
        
        self.audio = pyaudio.PyAudio()
        logger.info(f"Initializing WhisperProcessor with language: {language}")
        self.whisper = WhisperProcessor()
        self.language = language
        
        # Audio settings from config
        self.channels = AUDIO_CONFIG["channels"]
        self.rate = AUDIO_CONFIG["rate"]
        self.chunk = AUDIO_CONFIG["chunk"]
        self.format = pyaudio.paInt16
        
        # Silence detection settings from config
        self.silence_threshold = AUDIO_CONFIG["silence_threshold"]
        self.silence_duration = AUDIO_CONFIG["silence_duration"]
        self.max_chunk_duration = AUDIO_CONFIG["max_chunk_duration"]
        
        # Gain control settings
        self.min_gain_db = AUDIO_CONFIG["min_gain_db"]
        self.max_gain_db = AUDIO_CONFIG["max_gain_db"]
        self.gain_db = AUDIO_CONFIG["default_gain_db"]
        self._update_gain_multiplier()
        
        # Initialize audio filter
        self._init_filter()
        self.input_filter_state = None
        self.output_filter_state = None
        
        # Overlap handling
        self.overlap_size = self.chunk // 4  # 25% overlap
        self.input_overlap_buffer = None
        self.output_overlap_buffer = None
        
        # Noise gate threshold (RMS)
        self.noise_gate_threshold = 500
        
        # Monitoring state
        self.is_monitoring = False
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_mic: Optional[sc.Microphone] = None
        
        # Processing queues with size based on max chunk duration
        max_queue_size = int((self.rate * self.max_chunk_duration) / self.chunk)
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        
        # Processing threads
        self.input_thread: Optional[threading.Thread] = None
        self.output_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None

        # Debug directory for saving audio chunks
        self.debug_dir = Path("data/debug_audio")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Meetings directory for transcripts
        self.meetings_dir = Path("data/meetings")
        self.meetings_dir.mkdir(parents=True, exist_ok=True)

        # Initialize chunk counter and accumulator
        self.chunk_counter = 0
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.accumulated_duration = 0.0

        logger.info("AudioMonitor initialized")

    def _init_filter(self):
        """Initialize the audio filter optimized for speech."""
        try:
            # Design a bandpass filter (100Hz - 6kHz for speech)
            nyquist = self.rate / 2.0
            low_freq = 100.0  # Focus on speech frequencies
            high_freq = 6000.0  # Upper limit for speech, well below Nyquist
            
            # Convert to normalized frequencies (must be between 0 and 1)
            self.low = max(0.001, min(0.99, low_freq / nyquist))
            self.high = max(0.001, min(0.99, high_freq / nyquist))
            
            # Ensure high > low
            if self.high <= self.low:
                self.low = max(0.001, self.high - 0.1)
            
            # Create filter coefficients (higher order for better quality)
            self.filter_b, self.filter_a = signal.butter(6, [self.low, self.high], btype='band')
            logger.debug(
                f"Filter initialized with frequencies: {low_freq:.1f}Hz - {high_freq:.1f}Hz "
                f"(normalized: {self.low:.3f} - {self.high:.3f})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize filter: {str(e)}")
            raise


    def _update_gain_multiplier(self):
        """Update the gain multiplier based on the current gain_db setting."""
        self.gain_multiplier = 10 ** (self.gain_db / 20)
        logger.debug(f"Updated gain multiplier to {self.gain_multiplier} ({self.gain_db}dB)")

    def set_gain(self, gain_db: float):
        """Set the input gain in decibels."""
        self.gain_db = max(self.min_gain_db, min(self.max_gain_db, gain_db))
        self._update_gain_multiplier()
        logger.info(f"Set gain to {self.gain_db}dB")

    def _apply_gain_control(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply automatic gain control with limiting."""
        if np.abs(audio_data).max() > 0:
            # Calculate gain needed to reach target level
            gain = 0.5 * 32767 / np.abs(audio_data).max()
            # Limit maximum gain
            gain = min(gain, 2.0)
            audio_data = audio_data * gain
        return np.clip(audio_data, -32768, 32767).astype(np.int16)

    def _apply_noise_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise gate to remove low-level noise."""
        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        if rms < self.noise_gate_threshold:
            return np.zeros_like(audio_data)
        return audio_data

    def _apply_crossfade(self, current_data: np.ndarray, overlap_buffer: Optional[np.ndarray]) -> tuple:
        """Apply crossfade between audio chunks."""
        if overlap_buffer is not None:
            # Create crossfade
            fade_in = np.linspace(0, 1, self.overlap_size)
            fade_out = np.linspace(1, 0, self.overlap_size)
            
            # Apply crossfade
            current_data[:self.overlap_size] = (
                fade_out * overlap_buffer +
                fade_in * current_data[:self.overlap_size]
            )
        
        # Save overlap buffer for next chunk
        new_overlap_buffer = current_data[-self.overlap_size:].copy()
        return current_data, new_overlap_buffer

    def _process_audio_data(self, audio_data: np.ndarray, filter_state, overlap_buffer) -> tuple:
        """Process audio data with all quality improvements."""
        try:
            # Apply noise gate
            audio_data = self._apply_noise_gate(audio_data)
            
            # Apply filtering with state maintenance
            filtered_data, new_filter_state = signal.lfilter(
                self.filter_b, self.filter_a,
                audio_data,
                zi=filter_state * audio_data[0] if audio_data.size > 0 else filter_state
            )
            
            # Apply automatic gain control
            filtered_data = self._apply_gain_control(filtered_data)
            
            # Apply crossfade
            processed_data, new_overlap_buffer = self._apply_crossfade(filtered_data, overlap_buffer)
            
            return processed_data, new_filter_state, new_overlap_buffer
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return audio_data, filter_state, overlap_buffer

    def _input_callback(self, in_data, frame_count, time_info, status) -> tuple:
        """Process incoming audio data."""
        if status and status != pyaudio.paInputOverflow:
            logger.warning(f"Input stream status: {status}")
        
        if self.is_monitoring:
            try:
                # Convert to numpy array
                audio_array = np.frombuffer(in_data, dtype=np.int16)
                
                # Convert to mono if needed
                if self.channels == 1 and len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1).astype(np.int16)
                
                # Process audio with quality improvements
                processed_data, self.input_filter_state, self.input_overlap_buffer = self._process_audio_data(
                    audio_array,
                    self.input_filter_state,
                    self.input_overlap_buffer
                )
                
                try:
                    if self.is_monitoring:  # Check again before putting data
                        self.input_queue.put_nowait(processed_data.tobytes())
                        logger.debug("Successfully queued input audio data")
                except queue.Full:
                    logger.warning("Input queue full, dropping audio data")
                    
            except Exception as e:
                logger.error(f"Input processing error: {str(e)}")
                
        return (None, pyaudio.paComplete if not self.is_monitoring else pyaudio.paContinue)

    def _monitor_output(self) -> None:
        """Monitor system audio output."""
        while self.is_monitoring:
            try:
                # Record audio with overlap
                data = self.output_mic.record(
                    numframes=self.chunk + self.overlap_size,
                    samplerate=self.rate
                )
                
                # Convert float32 to int16
                audio_data = (data * 32767).astype(np.int16)
                
                # Convert to mono if needed
                if self.channels == 1 and len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1).astype(np.int16)
                
                # Process audio with quality improvements
                processed_data, self.output_filter_state, self.output_overlap_buffer = self._process_audio_data(
                    audio_data,
                    self.output_filter_state,
                    self.output_overlap_buffer
                )
                
                try:
                    if self.is_monitoring:  # Check again before putting data
                        self.output_queue.put_nowait(processed_data[:-self.overlap_size].tobytes())
                except queue.Full:
                    pass  # Skip if queue is full
                    
            except Exception as e:
                if self.is_monitoring:  # Only log if still monitoring
                    logger.error(f"Output monitoring error: {str(e)}")
                time.sleep(0.1)

    def start_monitoring(self, transcription_callback: Optional[Callable] = None) -> None:
        """Start monitoring audio streams."""
        if self.is_monitoring:
            logger.warning("Audio monitoring is already active")
            return

        try:
            # Store callback for use in stop_monitoring
            self._current_callback = transcription_callback
            
            # Reset states
            self.input_filter_state = signal.lfilter_zi(self.filter_b, self.filter_a)
            self.output_filter_state = signal.lfilter_zi(self.filter_b, self.filter_a)
            self.input_overlap_buffer = None
            self.output_overlap_buffer = None
            
            # Reset chunk counter and stats
            self.chunk_counter = 0
            self.processed_chunks = 0
            self.saved_chunks = 0
            self.transcribed_chunks = 0
            self.accumulated_audio = np.array([], dtype=np.float32)
            self.accumulated_duration = 0.0
            
            # Start input stream
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._input_callback
            )

            # Get default speaker for loopback recording
            default_speaker = sc.default_speaker()
            self.output_mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)

            # Start threads
            self.output_thread = threading.Thread(target=self._monitor_output, daemon=True)
            self.processing_thread = threading.Thread(
                target=self._process_audio,
                args=(transcription_callback,),
                daemon=True
            )

            self.is_monitoring = True
            self.input_stream.start_stream()
            self.output_thread.start()
            self.processing_thread.start()

            logger.info("Audio monitoring started")

        except Exception as e:
            logger.error(f"Failed to start audio monitoring: {str(e)}", exc_info=True)
            self.stop_monitoring()
            raise

    def _process_audio(self, transcription_callback: Optional[Callable]) -> None:
        """Process audio chunks and handle transcription."""
        current_chunk: list = []
        last_audio_time = time.time()
        chunk_start_time = time.time()
        
        while self.is_monitoring:
            try:
                # Process input audio
                while not self.input_queue.empty():
                    audio_data = self.input_queue.get()
                    current_chunk.append(audio_data)
                    last_audio_time = time.time()

                # Process output audio
                while not self.output_queue.empty():
                    audio_data = self.output_queue.get()
                    current_chunk.append(audio_data)
                    last_audio_time = time.time()

                # Check for chunk completion
                current_time = time.time()
                chunk_duration = current_time - chunk_start_time
                silence_time = current_time - last_audio_time

                if current_chunk and (
                    silence_time >= self.silence_duration or
                    chunk_duration >= self.max_chunk_duration or
                    not self.is_monitoring  # Process chunk if stopping
                ):
                    self._handle_chunk(current_chunk, transcription_callback)
                    current_chunk = []
                    chunk_start_time = current_time

                time.sleep(0.01)  # Small sleep to prevent busy waiting

            except Exception as e:
                if self.is_monitoring:  # Only log if still monitoring
                    logger.error(f"Audio processing error: {str(e)}")

        # Process any remaining chunk when stopping
        if current_chunk:
            try:
                self._handle_chunk(current_chunk, transcription_callback)
            except Exception as e:
                logger.error(f"Error processing final chunk: {str(e)}")

    def _handle_chunk(self, chunk: list, callback: Optional[Callable]) -> None:
        """Process and transcribe an audio chunk."""
        try:
            # Combine chunk data
            audio_data = b''.join(chunk)
            self.processed_chunks += 1
            
            # Save audio chunk to debug directory
            chunk_path = self.debug_dir / f"chunk_{self.chunk_counter:04d}.wav"
            self.save_chunk(audio_data, chunk_path)
            self.saved_chunks += 1
            self.chunk_counter += 1
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Check for silence
            if self._is_silence(audio_array):
                logger.debug("Skipping silent chunk")
                return

            # Normalize audio data
            normalized_array = audio_array.astype(np.float32) / 32768.0

            # Accumulate audio for better transcription
            self.accumulated_audio = np.concatenate([self.accumulated_audio, normalized_array])
            self.accumulated_duration += len(normalized_array) / self.rate

            # Only transcribe when we have enough audio
            min_duration = 1.0  # Minimum 1 second for transcription
            if self.accumulated_duration >= min_duration:
                # Get transcription
                text = self.whisper.transcribe_chunk(self.accumulated_audio, language=self.language)
                
                if text:
                    self.transcribed_chunks += 1
                    if callback:
                        callback(text)

                # Reset accumulator after transcription attempt
                self.accumulated_audio = np.array([], dtype=np.float32)
                self.accumulated_duration = 0.0

        except Exception as e:
            logger.error(f"Chunk processing error: {str(e)}", exc_info=True)

    def _is_silence(self, audio_array: np.ndarray) -> bool:
        """Detect if an audio chunk is silence."""
        try:
            rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
            return rms < self.silence_threshold
        except Exception as e:
            logger.error(f"Silence detection error: {str(e)}")
            return False

    def stop_monitoring(self) -> None:
        """Stop monitoring audio streams."""
        if not hasattr(self, 'is_monitoring') or not self.is_monitoring:
            return

        logger.info("Initiating audio monitoring shutdown...")
        
        try:
            # First set the flag to stop threads
            self.is_monitoring = False
            
            # Stop input stream first
            if hasattr(self, 'input_stream') and self.input_stream:
                if self.input_stream.is_active():
                    self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None

            # Process any remaining audio in the queues
            final_chunk = []
            
            # Get remaining input audio
            while not self.input_queue.empty():
                try:
                    final_chunk.append(self.input_queue.get_nowait())
                except queue.Empty:
                    break

            # Get remaining output audio
            while not self.output_queue.empty():
                try:
                    final_chunk.append(self.output_queue.get_nowait())
                except queue.Empty:
                    break

            # Process final chunk if we have any data
            if final_chunk:
                self._handle_chunk(final_chunk, getattr(self, '_current_callback', None))

            # Wait for threads to finish with timeout
            if hasattr(self, 'output_thread') and self.output_thread:
                self.output_thread.join(timeout=1.0)
            if hasattr(self, 'processing_thread') and self.processing_thread:
                self.processing_thread.join(timeout=1.0)

            logger.info(f"Final stats - Processed: {self.processed_chunks}, "
                       f"Saved: {self.saved_chunks}, "
                       f"Transcribed: {self.transcribed_chunks}")

        except Exception as e:
            logger.error(f"Error during monitoring shutdown: {str(e)}", exc_info=True)
        finally:
            self.is_monitoring = False
            logger.info("Audio monitoring stopped successfully")

    def save_chunk(self, chunk: bytes, filepath: Path) -> None:
        """Save an audio chunk to a WAV file."""
        try:
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.rate)
                wav_file.writeframes(chunk)
        except Exception as e:
            logger.error(f"Failed to save audio chunk: {str(e)}")
            raise

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'is_monitoring'):
            self.stop_monitoring()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
