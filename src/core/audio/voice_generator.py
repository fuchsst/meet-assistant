"""Voice generation using multiple TTS backends including F5-TTS and Bark."""
from datetime import datetime
import logging
from pathlib import Path
import tempfile
from typing import Optional, Dict, Any, Union
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
from f5_tts.api import F5TTS
from transformers import AutoProcessor, BarkModel
import nltk

from config.config import VOICE_CONFIG

logger = logging.getLogger(__name__)

class TTSBackend:
    """Base class for TTS backends."""
    
    def generate(self, text: str, voice_config: Dict[str, Any]) -> tuple[np.ndarray, int]:
        """Generate speech from text."""
        raise NotImplementedError

class F5TTSBackend(TTSBackend):
    """F5-TTS backend implementation."""
    
    def __init__(self):
        self.tts = F5TTS(
            model_type=VOICE_CONFIG["model_type"],
            vocoder_name=VOICE_CONFIG["vocoder"]
        )

    def generate(self, text: str, voice_config: Dict[str, Any]) -> tuple[np.ndarray, int]:
        wav, sr, _ = self.tts.infer(
            ref_file=voice_config["ref_audio"],
            ref_text=voice_config["ref_text"],
            gen_text=text,
            speed=voice_config.get("speed", 1.0),
            remove_silence=voice_config.get("remove_silence", True)
        )
        return wav, sr

class BarkBackend(TTSBackend):
    """Bark TTS backend implementation."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark").to(self.device)

    def generate(self, text: str, voice_config: Dict[str, Any]) -> tuple[np.ndarray, int]:
        # Process text with voice preset and emotion
        inputs = self.processor(
            text,
            voice_preset=voice_config.get("voice_preset", "v2/en_speaker_1"),
            return_tensors="pt"
        )
        
        # Add emotion embedding if specified
        if "emotion" in voice_config:
            emotion_embedding = self._get_emotion_embedding(voice_config["emotion"])
            inputs["emotion_embedding"] = emotion_embedding.to(self.device)
        
        # Generate audio
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
        
        # Convert to numpy and adjust sample rate
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        
        return audio_array, sample_rate

    def _get_emotion_embedding(self, emotion: str) -> torch.Tensor:
        """Generate emotion embedding based on specified emotion."""
        # Map emotions to embeddings (simplified version)
        emotion_map = {
            "happy": torch.ones(1024) * 0.5,
            "sad": torch.ones(1024) * -0.5,
            "angry": torch.ones(1024) * 0.8,
            "neutral": torch.zeros(1024)
        }
        return emotion_map.get(emotion, torch.zeros(1024))

class VoiceGenerator:
    """Handles text-to-speech generation and playback with multiple backends."""

    def __init__(self, voice_id: Optional[str] = None, backend: str = "f5-tts"):
        """Initialize TTS engine."""
        try:
            # Initialize instance variables
            self.is_playing = False
            self.current_audio: Optional[np.ndarray] = None
            self.sample_rate = VOICE_CONFIG["sample_rate"]
            
            # Initialize backend
            self.backend = self._create_backend(backend)
            
            # Get voice configuration
            self.voice_id = voice_id or VOICE_CONFIG.get("default_voice", list(VOICE_CONFIG["voices"].keys())[0])
            if self.voice_id not in VOICE_CONFIG["voices"]:
                available_voices = list(VOICE_CONFIG["voices"].keys())
                self.voice_id = available_voices[0] if available_voices else None
                if not self.voice_id:
                    raise ValueError("No voices configured in VOICE_CONFIG")
            
            self.voice_config = VOICE_CONFIG["voices"][self.voice_id]
            
            # Initialize text processor for long-form synthesis
            nltk.download('punkt', quiet=True)
            
            logger.info(f"Voice generator initialized with voice: {self.voice_id} using {backend} backend")
        except Exception as e:
            logger.error(f"Failed to initialize voice generator: {str(e)}")
            raise

    def _create_backend(self, backend_name: str) -> TTSBackend:
        """Create TTS backend based on name."""
        backends = {
            "f5-tts": F5TTSBackend,
            "bark": BarkBackend
        }
        if backend_name not in backends:
            raise ValueError(f"Unsupported backend: {backend_name}")
        return backends[backend_name]()

    def set_voice(self, voice_id: str, voice_config: Optional[Dict[str, Any]] = None) -> None:
        """Change the current voice."""
        try:
            if voice_config:
                # Allow custom voice configuration
                self.voice_config = voice_config
                self.voice_id = voice_id
            elif voice_id in VOICE_CONFIG["voices"]:
                self.voice_id = voice_id
                self.voice_config = VOICE_CONFIG["voices"][voice_id]
            else:
                raise ValueError(f"Invalid voice ID: {voice_id}")
            
            logger.info(f"Changed voice to: {voice_id}")
            
        except Exception as e:
            logger.error(f"Failed to set voice: {str(e)}")
            raise

    def set_emotion(self, emotion: str) -> None:
        """Set emotion for speech generation."""
        try:
            valid_emotions = ["happy", "sad", "angry", "neutral"]
            if emotion not in valid_emotions:
                raise ValueError(f"Invalid emotion. Must be one of: {valid_emotions}")
            
            self.voice_config["emotion"] = emotion
            logger.info(f"Set emotion to: {emotion}")
            
        except Exception as e:
            logger.error(f"Failed to set emotion: {str(e)}")
            raise

    def generate_speech(self, text: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Generate speech from text."""
        try:
            if not self.voice_id:
                raise ValueError("No voice selected")
            
            # Generate speech using selected backend
            wav, sr = self.backend.generate(text, self.voice_config)
            
            if output_path:
                # Save to file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), wav, sr)
                logger.info(f"Generated speech to file: {output_path}")
                return output_path
            else:
                # Store in memory
                self.current_audio = wav
                self.sample_rate = sr
                logger.info("Generated speech to memory")
                return None

        except Exception as e:
            logger.error(f"Speech generation failed: {str(e)}")
            raise

    def generate_long_form(self, text: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Generate speech for long-form text with proper sentence breaks."""
        try:
            # Split text into sentences
            sentences = nltk.sent_tokenize(text)
            
            # Generate audio for each sentence
            audio_segments = []
            silence = np.zeros(int(0.25 * self.sample_rate))  # 0.25s silence between sentences
            
            for sentence in sentences:
                wav, sr = self.backend.generate(sentence, self.voice_config)
                audio_segments.extend([wav, silence.copy()])
            
            # Combine all segments
            combined_audio = np.concatenate(audio_segments)
            
            if output_path:
                # Save to file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), combined_audio, self.sample_rate)
                logger.info(f"Generated long-form speech to file: {output_path}")
                return output_path
            else:
                # Store in memory
                self.current_audio = combined_audio
                logger.info("Generated long-form speech to memory")
                return None
                
        except Exception as e:
            logger.error(f"Long-form speech generation failed: {str(e)}")
            raise

    def play_audio(self, audio_path: Optional[Path] = None) -> None:
        """Play generated audio."""
        try:
            if self.is_playing:
                logger.warning("Audio is already playing")
                return

            # Load audio data
            if audio_path:
                audio_data, sample_rate = sf.read(audio_path)
            elif self.current_audio is not None:
                audio_data = self.current_audio
                sample_rate = self.sample_rate
            else:
                raise ValueError("No audio data available")

            # Play audio
            self.is_playing = True
            sd.play(audio_data, sample_rate, blocking=True)
            self.is_playing = False
            
            logger.info("Audio playback completed")

        except Exception as e:
            logger.error(f"Audio playback failed: {str(e)}")
            self.is_playing = False
            raise

    def stop_playback(self) -> None:
        """Stop current audio playback."""
        try:
            if self.is_playing:
                sd.stop()
                self.is_playing = False
                logger.info("Audio playback stopped")
        except Exception as e:
            logger.error(f"Failed to stop playback: {str(e)}")
            raise

    def clone_voice(self, reference_audio: Path, reference_text: str) -> Dict[str, Any]:
        """Create a new voice profile from reference audio and text."""
        try:
            # Create voice configuration
            voice_config = {
                "ref_audio": str(reference_audio),
                "ref_text": reference_text,
                "type": "cloned",
                "created_at": str(datetime.now())
            }
            
            # Validate voice by generating a test sample
            test_text = "This is a test of the cloned voice."
            self.backend.generate(test_text, voice_config)
            
            logger.info("Voice cloning successful")
            return voice_config
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {str(e)}")
            raise

    def preview_voice(self, text: str = "This is a preview of the selected voice.") -> None:
        """Generate and play a preview of the current voice settings."""
        try:
            # Create temporary file for preview
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Generate preview
                self.generate_speech(text, temp_path)
                
                # Play preview
                self.play_audio(temp_path)
                
                # Cleanup
                temp_path.unlink()
            
            logger.info("Voice preview completed")

        except Exception as e:
            logger.error(f"Voice preview failed: {str(e)}")
            raise

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'is_playing') and self.is_playing:
            self.stop_playback()
