"""Real-time transcription using Whisper."""
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Generator, Union
import torch
import whisper
import numpy as np

from config.config import WHISPER_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)

class WhisperProcessor:
    """Handles real-time transcription using Whisper."""

    def __init__(self):
        """Initialize Whisper model and settings."""
        self.model_size = WHISPER_CONFIG["model_size"]
        self.language = WHISPER_CONFIG["language"]
        self.task = WHISPER_CONFIG["task"]
        self.model: Optional[whisper.Whisper] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self._load_model()
        
        # Track transcription attempts
        self.transcription_attempts = 0
        self.successful_transcriptions = 0
        
        # Maximum audio duration (15 seconds)
        self.max_duration = 15
        
        logger.info(f"WhisperProcessor initialized with {self.model_size} model")
        logger.info(f"Language: {self.language}, Task: {self.task}, Device: {self.device}")

    def _load_model(self) -> None:
        """Load or download Whisper model."""
        try:
            logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}")
            
            # Create models directory if it doesn't exist
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Load model
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=str(MODELS_DIR)
            )
            
            # Perform a test transcription to ensure model is working
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            _ = self.model.transcribe(test_audio)
            
            logger.info(f"Whisper model loaded successfully on {self.device}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}", exc_info=True)
            raise

    def transcribe_chunk(self, audio_data: Union[np.ndarray, bytes], language: Optional[str] = None) -> str:
        """Transcribe an audio chunk in real-time."""
        if not self.model:
            logger.error("Whisper model not initialized")
            raise RuntimeError("Whisper model not initialized")

        try:
            self.transcription_attempts += 1
            logger.debug(f"Starting transcription attempt {self.transcription_attempts}")
            
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                logger.debug("Converting audio bytes to numpy array")
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data

            # Ensure audio is float32 and normalized to [-1, 1]
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / 32768.0

            # Check audio duration
            audio_duration = len(audio_array) / 16000  # assuming 16kHz sample rate
            if audio_duration > self.max_duration:
                logger.warning(f"Audio chunk too long ({audio_duration:.2f}s), truncating to {self.max_duration}s")
                samples_to_keep = int(16000 * self.max_duration)
                audio_array = audio_array[:samples_to_keep]
                audio_duration = self.max_duration
            
            logger.debug(f"Audio duration: {audio_duration:.2f} seconds")
            
            if audio_duration < 0.1:  # Less than 100ms
                logger.debug("Audio chunk too short, skipping transcription")
                return ""

            # Log audio stats before transcription
            rms = np.sqrt(np.mean(np.square(audio_array)))
            peak = np.abs(audio_array).max()
            logger.debug(
                f"Audio stats - Duration: {audio_duration:.2f}s, "
                f"Shape: {audio_array.shape}, "
                f"RMS: {rms:.3f}, Peak: {peak:.3f}, "
                f"Mean: {audio_array.mean():.3f}"
            )

            # Check if audio has enough signal
            if rms < 0.001:  # Very low signal
                logger.debug(f"Audio signal too weak (RMS: {rms:.6f}), skipping transcription")
                return ""

            # Get transcription
            logger.info(f"Starting Whisper transcription of {audio_duration:.2f}s audio...")
            
            # Set up transcription parameters optimized for real-time
            transcribe_params = {
                "language": language or self.language,
                "task": self.task,
                "fp16": torch.cuda.is_available(),
                "initial_prompt": "This is a real-time transcription.",
                "condition_on_previous_text": True,
                "temperature": 0.0,  # Use greedy decoding for more reliable real-time results
                "compression_ratio_threshold": 2.0,  # Less aggressive filtering
                "no_speech_threshold": 0.3,  # Lower threshold to detect more potential speech
                "logprob_threshold": -1.0,  # Include all text in output
                "best_of": 1,  # Speed up inference
                "beam_size": 1,  # Speed up inference
                "patience": 1,  # Speed up inference
            }
            
            # Perform transcription with memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            result = self.model.transcribe(audio_array, **transcribe_params)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                logger.debug(f"Peak CUDA memory during transcription: {peak_memory:.2f} MB")
            
            logger.debug(f"Raw transcription result: {result}")

            # Extract text with timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            text = result["text"].strip()
            
            if text:
                self.successful_transcriptions += 1
                formatted_text = f"[{timestamp}] {text}"
                logger.info(
                    f"Transcription successful ({self.successful_transcriptions}/{self.transcription_attempts}): "
                    f"{formatted_text}"
                )
                return formatted_text
            else:
                logger.debug("No text detected in transcription result")
                return ""

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            if torch.cuda.is_available():
                logger.error(f"CUDA memory state: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
                torch.cuda.empty_cache()
            return ""

    def transcribe_stream(self, audio_stream: Generator[Union[np.ndarray, bytes], None, None]) -> Generator[str, None, None]:
        """Transcribe an audio stream in real-time."""
        if not self.model:
            logger.error("Whisper model not initialized")
            raise RuntimeError("Whisper model not initialized")

        try:
            logger.info("Starting stream transcription")
            for chunk_num, chunk in enumerate(audio_stream, 1):
                logger.debug(f"Processing stream chunk {chunk_num}")
                transcription = self.transcribe_chunk(chunk)
                if transcription:
                    logger.debug(f"Stream chunk {chunk_num} transcribed: {transcription}")
                    yield transcription
                else:
                    logger.debug(f"No transcription for stream chunk {chunk_num}")

        except Exception as e:
            logger.error(f"Stream transcription error: {str(e)}", exc_info=True)
            yield ""

    def transcribe_file(self, audio_path: Path) -> list[dict]:
        """Transcribe an entire audio file with detailed segments."""
        if not self.model:
            logger.error("Whisper model not initialized")
            raise RuntimeError("Whisper model not initialized")

        try:
            logger.info(f"Starting file transcription: {audio_path}")
            
            # Load and transcribe audio file
            result = self.model.transcribe(
                str(audio_path),
                language=self.language,
                task=self.task,
                fp16=torch.cuda.is_available()
            )
            logger.debug(f"Raw file transcription result: {result}")

            # Format segments with timestamps
            segments = []
            for segment in result["segments"]:
                formatted_segment = {
                    "start": self._format_timestamp(segment["start"]),
                    "end": self._format_timestamp(segment["end"]),
                    "text": segment["text"].strip()
                }
                segments.append(formatted_segment)
                logger.debug(f"Processed segment: {formatted_segment}")

            logger.info(f"File transcription completed: {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"File transcription error: {str(e)}", exc_info=True)
            raise

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def __del__(self):
        """Clean up resources."""
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            try:
                before_mem = torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                after_mem = torch.cuda.memory_allocated()
                logger.info(f"CUDA memory cleared: {(before_mem - after_mem) / 1024**2:.2f} MB freed")
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {str(e)}")
