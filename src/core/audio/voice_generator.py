"""Voice generation using F5-TTS."""
import logging
from pathlib import Path
import tempfile
from typing import Optional
import numpy as np
import sounddevice as sd
import soundfile as sf
from f5_tts.api import F5TTS

from config.config import VOICE_CONFIG

logger = logging.getLogger(__name__)

class VoiceGenerator:
    """Handles text-to-speech generation and playback."""

    def __init__(self, voice_id: Optional[str] = None):
        """Initialize TTS engine."""
        try:
            # Initialize instance variables
            self.is_playing = False
            self.current_audio: Optional[np.ndarray] = None
            self.sample_rate = VOICE_CONFIG["sample_rate"]
            self.speed = VOICE_CONFIG["speed"]
            
            # Get voice configuration
            self.voice_id = voice_id or VOICE_CONFIG.get("default_voice", list(VOICE_CONFIG["voices"].keys())[0])
            if self.voice_id not in VOICE_CONFIG["voices"]:
                available_voices = list(VOICE_CONFIG["voices"].keys())
                self.voice_id = available_voices[0] if available_voices else None
                if not self.voice_id:
                    raise ValueError("No voices configured in VOICE_CONFIG")
            
            self.voice_config = VOICE_CONFIG["voices"][self.voice_id]
            
            # Initialize F5-TTS
            self.tts = F5TTS(
                model_type=VOICE_CONFIG["model_type"],
                vocoder_name=VOICE_CONFIG["vocoder"]
            )
            
            logger.info(f"Voice generator initialized with voice: {self.voice_id}")
        except Exception as e:
            logger.error(f"Failed to initialize voice generator: {str(e)}")
            raise

    def set_voice(self, voice_id: str) -> None:
        """Change the current voice."""
        try:
            if voice_id not in VOICE_CONFIG["voices"]:
                raise ValueError(f"Invalid voice ID: {voice_id}")
            
            self.voice_id = voice_id
            self.voice_config = VOICE_CONFIG["voices"][voice_id]
            logger.info(f"Changed voice to: {voice_id}")
            
        except Exception as e:
            logger.error(f"Failed to set voice: {str(e)}")
            raise

    def generate_speech(self, text: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Generate speech from text."""
        try:
            if not self.voice_id:
                raise ValueError("No voice selected")
                
            # Generate speech
            if output_path:
                # Generate directly to file
                wav, sr, _ = self.tts.infer(
                    ref_file=self.voice_config["ref_audio"],
                    ref_text=self.voice_config["ref_text"],
                    gen_text=text,
                    speed=self.speed,
                    file_wave=str(output_path),
                    remove_silence=VOICE_CONFIG["remove_silence"]
                )
                logger.info(f"Generated speech to file: {output_path}")
                return output_path
            else:
                # Generate to memory
                wav, sr, _ = self.tts.infer(
                    ref_file=self.voice_config["ref_audio"],
                    ref_text=self.voice_config["ref_text"],
                    gen_text=text,
                    speed=self.speed,
                    remove_silence=VOICE_CONFIG["remove_silence"]
                )
                self.current_audio = wav
                self.sample_rate = sr
                logger.info("Generated speech to memory")
                return None

        except Exception as e:
            logger.error(f"Speech generation failed: {str(e)}")
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

    def generate_and_play(self, text: str) -> None:
        """Generate speech and play it immediately."""
        try:
            # Generate speech
            self.generate_speech(text)
            
            # Play generated audio
            self.play_audio()
            
            logger.info("Generated and played speech successfully")

        except Exception as e:
            logger.error(f"Generate and play failed: {str(e)}")
            raise

    def create_voice_sample(self, text: str, reference_path: Path) -> None:
        """Create a voice sample with reference text."""
        try:
            # Generate speech
            self.generate_speech(text, reference_path)
            
            logger.info(f"Created voice sample at: {reference_path}")

        except Exception as e:
            logger.error(f"Failed to create voice sample: {str(e)}")
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
