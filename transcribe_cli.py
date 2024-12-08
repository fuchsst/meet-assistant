"""CLI script for transcribing WAV files to VTT format using Whisper."""
import logging
from pathlib import Path
import sys
import fire
import torch
import whisper
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime

from config.config import WHISPER_CONFIG, MODELS_DIR
from src.core.storage.metadata_manager import UnifiedMetadataManager
from src.core.utils.logging_config import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs/transcription_debug.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class WhisperTranscriber:
    """Handles transcription of WAV files using Whisper."""
    
    def __init__(self, language: str = WHISPER_CONFIG["language"]):
        """Initialize transcriber.
        
        Args:
            language: Language for transcription (default: from config)
        """
        self.language = language
        self.metadata_manager = UnifiedMetadataManager()
        
        # Initialize Whisper
        logger.info("Loading Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.debug(f"Loading Whisper model size: {WHISPER_CONFIG['model_size']}")
            self.whisper = whisper.load_model(
                WHISPER_CONFIG["model_size"],
                device=self.device,
                download_root=str(MODELS_DIR)
            )
            logger.info(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

    def _preprocess_text(self, text: str) -> str:
        """Preprocess transcribed text for better formatting."""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove leading ellipses
        if text.startswith("..."):
            text = text[3:].lstrip()
        
        # Ensure first letter is capitalized
        if text:
            text = text[0].upper() + text[1:]
        
        # Ensure proper sentence endings
        if text and text[-1].isalnum():
            text += "."
        
        return text

    def _save_vtt(self, vtt_path: Path, segments: list) -> None:
        """Save transcription segments as VTT file."""
        try:
            with open(vtt_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for segment in segments:
                    start = segment['start']
                    end = segment['end']
                    text = segment['text'].strip()
                    
                    # Format timestamps as HH:MM:SS.mmm
                    start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}"
                    end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.3f}"
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            logger.info(f"Saved VTT file: {vtt_path}")
            
        except Exception as e:
            logger.error(f"Failed to save VTT file: {e}", exc_info=True)
            raise

    def transcribe_file(self, audio_path: Path, output_path: Path = None) -> Path:
        """Transcribe a single WAV file to VTT.
        
        Args:
            audio_path: Path to WAV file
            output_path: Optional path for VTT output. If not provided,
                        will create next to WAV file with .vtt extension.
        
        Returns:
            Path to created VTT file
        """
        try:
            logger.info(f"Starting transcription for: {audio_path}")
            
            # Determine output path
            if output_path is None:
                output_path = audio_path.with_suffix('.vtt')
            
            # Load audio file
            try:
                logger.debug(f"Loading audio file: {audio_path}")
                audio = whisper.load_audio(str(audio_path))
                logger.debug(f"Loaded audio file: {len(audio)} samples")
                
                if np.all(audio == 0):
                    logger.warning(f"Audio file contains only zeros: {audio_path}")
                    return None
                
            except Exception as e:
                logger.error(f"Failed to load audio file: {e}", exc_info=True)
                raise
            
            # Transcribe audio
            try:
                logger.debug("Starting Whisper transcription")
                result = self.whisper.transcribe(
                    audio,
                    language=self.language,
                    task="transcribe",
                    fp16=torch.cuda.is_available(),
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    beam_size=5,
                    best_of=5
                )
                logger.debug("Transcription complete")
            except Exception as e:
                logger.error(f"Failed to transcribe audio: {e}", exc_info=True)
                raise
            
            # Process segments and save VTT
            segments = []
            for segment in result["segments"]:
                text = self._preprocess_text(segment["text"])
                if text and len(text) > 2:
                    segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text
                    })
            
            self._save_vtt(output_path, segments)
            return output_path
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe_meeting(
        self,
        project_id: Optional[str],
        meeting_id: str,
        language: str = WHISPER_CONFIG["language"]
    ) -> List[Path]:
        """Transcribe all WAV files in a meeting directory."""
        try:
            # Get meeting directory
            meeting_dir = self.metadata_manager.get_meeting_dir(
                self.metadata_manager.get_project(project_id)["key"],
                meeting_id
            )
            if not meeting_dir.exists():
                raise ValueError(f"Meeting directory not found: {meeting_dir}")
            
            logger.info(f"Processing meeting directory: {meeting_dir}")
            
            # Transcribe all WAV files
            vtt_files = []
            for wav_file in meeting_dir.glob("*.wav"):
                try:
                    vtt_file = self.transcribe_file(wav_file)
                    if vtt_file:
                        vtt_files.append(vtt_file)
                except Exception as e:
                    logger.error(f"Failed to transcribe {wav_file}: {e}")
                    continue
            
            if not vtt_files:
                logger.warning("No WAV files were successfully transcribed")
                return []
            
            # Update meeting metadata
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "transcription_status": "completed",
                    "transcription_date": datetime.now().isoformat(),
                    "language": language,
                    "vtt_files": [str(f.relative_to(meeting_dir)) for f in vtt_files]
                }
            )
            
            return vtt_files
            
        except Exception as e:
            logger.error(f"Failed to transcribe meeting: {e}")
            raise

def main(
    meeting_id: str,
    project_id: Optional[str] = None,
    language: str = WHISPER_CONFIG["language"]
):
    """Transcribe meeting recordings to VTT format.
    
    Args:
        meeting_id: ID of the meeting to transcribe
        project_id: Optional project ID (uses default project if not provided)
        language: Language for transcription (default: from config)
    """
    try:
        transcriber = WhisperTranscriber(language=language)
        
        print(f"Starting transcription for meeting: {meeting_id}")
        print(f"Project: {project_id or 'default'}")
        print(f"Language: {language}")
        
        vtt_files = transcriber.transcribe_meeting(project_id, meeting_id, language)
        
        if vtt_files:
            print(f"\nCreated {len(vtt_files)} VTT files:")
            for vtt_file in vtt_files:
                print(f"  {vtt_file}")
        else:
            print("\nNo files were transcribed. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    fire.Fire(main)
