"""Audio recording functionality using PyAudio."""
import wave
from pathlib import Path
import logging
from typing import Optional
import pyaudio
import numpy as np
from scipy import signal

from config.config import AUDIO_CONFIG

logger = logging.getLogger(__name__)

class AudioRecorder:
    """Handles audio recording functionality using PyAudio."""

    def __init__(self):
        """Initialize the audio recorder with configured settings."""
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.frames: list = []
        self.is_recording: bool = False
        
        # Audio settings from config
        self.format = pyaudio.paInt16  # 16-bit resolution
        self.channels = AUDIO_CONFIG["channels"]
        self.rate = AUDIO_CONFIG["rate"]
        self.chunk = AUDIO_CONFIG["chunk"]
        self.sample_width = AUDIO_CONFIG["sample_width"]
        
        # Maximum duration in seconds (15 seconds)
        self.max_duration = 15
        self.max_frames = int((self.rate * self.max_duration) / self.chunk)
        
        # Initialize filter and buffer
        self._init_filter()
        self.filter_state = None
        self.overlap_buffer = None
        self.overlap_size = self.chunk // 4  # 25% overlap

    def _init_filter(self):
        """Initialize the audio filter."""
        # Design a bandpass filter (100Hz - 8kHz for speech)
        nyquist = self.rate * 0.5
        low_freq = 100.0  # Focus on speech frequencies
        high_freq = 8000.0  # Upper limit for speech
        
        # Convert to normalized frequencies
        self.low = low_freq / nyquist
        self.high = high_freq / nyquist
        
        # Create filter coefficients (higher order for better quality)
        self.filter_b, self.filter_a = signal.butter(6, [self.low, self.high], btype='band')

    def start_recording(self) -> None:
        """Start recording audio."""
        if self.is_recording:
            logger.warning("Recording is already in progress")
            return

        try:
            # Reset states
            self.filter_state = signal.lfilter_zi(self.filter_b, self.filter_a)
            self.overlap_buffer = None
            
            # Configure stream with optimal buffer
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback
            )
            
            self.frames = []
            self.is_recording = True
            self.stream.start_stream()
            
            logger.info("Started audio recording")
        except Exception as e:
            logger.error(f"Failed to start recording: {str(e)}")
            raise

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if not self.is_recording:
            logger.warning("No recording in progress")
            return

        try:
            self.is_recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            logger.info("Stopped audio recording")
        except Exception as e:
            logger.error(f"Failed to stop recording: {str(e)}")
            raise

    def _audio_callback(self, in_data, frame_count, time_info, status) -> tuple:
        """Process audio data from the stream."""
        if status:
            logger.warning(f"Stream status: {status}")
        
        if self.is_recording and len(self.frames) < self.max_frames:
            try:
                # Convert to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                
                # Apply noise gate
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                if rms < 500:  # Adjust threshold as needed
                    audio_data = np.zeros_like(audio_data)
                
                # Apply filtering with state maintenance
                filtered_data, self.filter_state = signal.lfilter(
                    self.filter_b, self.filter_a,
                    audio_data,
                    zi=self.filter_state * audio_data[0] if audio_data.size > 0 else self.filter_state
                )
                
                # Apply automatic gain control
                if np.abs(filtered_data).max() > 0:
                    gain = 0.5 * 32767 / np.abs(filtered_data).max()
                    filtered_data = filtered_data * min(gain, 2.0)  # Limit maximum gain
                
                # Handle overlap
                if self.overlap_buffer is not None:
                    # Create crossfade
                    fade_in = np.linspace(0, 1, self.overlap_size)
                    fade_out = np.linspace(1, 0, self.overlap_size)
                    
                    # Apply crossfade
                    filtered_data[:self.overlap_size] = (
                        fade_out * self.overlap_buffer +
                        fade_in * filtered_data[:self.overlap_size]
                    )
                
                # Save overlap buffer for next chunk
                self.overlap_buffer = filtered_data[-self.overlap_size:].copy()
                
                # Convert back to int16
                filtered_data = np.clip(filtered_data, -32768, 32767).astype(np.int16)
                
                # Store processed data
                self.frames.append(filtered_data.tobytes())
                
            except Exception as e:
                logger.error(f"Error in audio callback: {str(e)}")
                return (in_data, pyaudio.paComplete)
            
            return (in_data, pyaudio.paContinue)
        else:
            return (in_data, pyaudio.paComplete)

    def save_recording(self, filepath: Path) -> None:
        """Save the recorded audio to a WAV file."""
        if not self.frames:
            logger.warning("No audio data to save")
            return

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with wave.open(str(filepath), 'wb') as wave_file:
                wave_file.setnchannels(self.channels)
                wave_file.setsampwidth(self.sample_width)
                wave_file.setframerate(self.rate)
                wave_file.writeframes(b''.join(self.frames))
            
            logger.info(f"Saved audio recording to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save recording: {str(e)}")
            raise

    def validate_audio(self) -> bool:
        """Validate the recorded audio data."""
        if not self.frames:
            logger.warning("No audio data to validate")
            return False

        try:
            # Convert frames to numpy array for analysis
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
            
            # Check for silence (very low amplitude)
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
            if rms < 100:
                logger.warning("Audio recording appears to be silent")
                return False
            
            # Check for clipping
            if np.abs(audio_data).max() > 32700:
                logger.warning("Audio recording contains clipping")
                return False
            
            # Check duration
            duration = len(audio_data) / (self.rate * self.channels)
            if duration < 0.1 or duration > self.max_duration:
                logger.warning(f"Invalid audio duration: {duration:.2f} seconds")
                return False

            logger.info(f"Audio validation passed: {duration:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            return False

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'stream') and self.stream:
            self.stream.close()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
