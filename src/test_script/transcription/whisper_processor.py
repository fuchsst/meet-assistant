"""Real-time transcription using Whisper."""
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Generator
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
            
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def transcribe_chunk(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe an audio chunk in real-time."""
        if not self.model:
            raise RuntimeError("Whisper model not initialized")

        try:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Get transcription
            result = self.model.transcribe(
                audio_array,
                language=self.language,
                task=self.task,
                fp16=torch.cuda.is_available()
            )

            # Extract text with timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            text = result["text"].strip()
            
            if text:
                return f"[{timestamp}] {text}"
            return ""

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""

    def transcribe_stream(self, audio_stream: Generator[bytes, None, None]) -> Generator[str, None, None]:
        """Transcribe an audio stream in real-time."""
        if not self.model:
            raise RuntimeError("Whisper model not initialized")

        try:
            for chunk in audio_stream:
                transcription = self.transcribe_chunk(chunk)
                if transcription:
                    yield transcription

        except Exception as e:
            logger.error(f"Stream transcription error: {str(e)}")
            yield ""

    def transcribe_file(self, audio_path: Path) -> list[dict]:
        """Transcribe an entire audio file with detailed segments."""
        if not self.model:
            raise RuntimeError("Whisper model not initialized")

        try:
            # Load and transcribe audio file
            result = self.model.transcribe(
                str(audio_path),
                language=self.language,
                task=self.task,
                fp16=torch.cuda.is_available()
            )

            # Format segments with timestamps
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": self._format_timestamp(segment["start"]),
                    "end": self._format_timestamp(segment["end"]),
                    "text": segment["text"].strip()
                })

            return segments

        except Exception as e:
            logger.error(f"File transcription error: {str(e)}")
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
            torch.cuda.empty_cache()
