"""CLI script for transcribing WAV files to VTT format using Whisper."""
import logging
from pathlib import Path
import sys
import time
from typing import Optional, Dict, List, Set
from datetime import datetime
import fire
import torch
import whisper
import numpy as np

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

    def transcribe_file(self, audio_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Transcribe a single WAV file to VTT.
        
        Args:
            audio_path: Path to WAV file
            output_path: Optional path for VTT output. If not provided,
                        will create next to WAV file with .vtt extension.
        
        Returns:
            Path to created VTT file or None if transcription failed
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
            return None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe_meeting(
        self,
        project_id: Optional[str],
        meeting_id: str,
        language: str = WHISPER_CONFIG["language"]
    ) -> List[Path]:
        """Transcribe WAV files listed in meeting metadata.
        
        Args:
            project_id: Optional project ID (uses default if not provided)
            meeting_id: Meeting ID or directory name
            language: Language for transcription
            
        Returns:
            List of created VTT file paths
        """
        try:
            # Get project ID if not provided
            if not project_id:
                project = self.metadata_manager.get_project()
                project_id = project["key"]
            
            # Get meeting metadata to check recording files
            meeting_metadata = self.metadata_manager.get_meeting_metadata(project_id, meeting_id)
            recording_files = meeting_metadata.get("recording_files", [])
            vtt_files = meeting_metadata.get("vtt_files", [])
            
            if not recording_files:
                logger.warning("No recording files found in meeting metadata")
                self.metadata_manager.update_meeting_metadata(
                    project_id,
                    meeting_id,
                    {
                        "transcription_status": "no_audio_files",
                        "transcription_end": datetime.utcnow().isoformat()
                    }
                )
                return []
            
            # Update metadata to show transcription started
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "transcription_status": "in_progress",
                    "transcription_start": datetime.utcnow().isoformat(),
                    "language": language
                }
            )
            
            # Transcribe WAV files that don't have corresponding VTT files
            created_vtt_files = []
            for wav_path in recording_files:
                try:
                    # Convert relative path from metadata to absolute path
                    wav_file = self.metadata_manager.data_dir / wav_path
                    if not wav_file.exists():
                        logger.warning(f"WAV file not found: {wav_file}")
                        continue
                    
                    # Check if VTT already exists in metadata
                    vtt_path = wav_path.replace('.wav', '.vtt')
                    if vtt_path in vtt_files:
                        logger.info(f"VTT file already exists in metadata for {wav_file.name}")
                        created_vtt_files.append(self.metadata_manager.data_dir / vtt_path)
                        continue
                    
                    # Transcribe file
                    vtt_file = self.transcribe_file(wav_file)
                    if vtt_file:
                        created_vtt_files.append(vtt_file)
                        # Add new VTT file to metadata list
                        vtt_files.append(str(vtt_file.relative_to(self.metadata_manager.data_dir)))
                        # Update metadata with new VTT file
                        self.metadata_manager.update_meeting_metadata(
                            project_id,
                            meeting_id,
                            {"vtt_files": vtt_files}
                        )
                except Exception as e:
                    logger.error(f"Failed to transcribe {wav_path}: {e}")
                    continue
            
            # Update meeting metadata with final status
            status = "completed" if created_vtt_files else "failed"
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "transcription_status": status,
                    "transcription_end": datetime.utcnow().isoformat(),
                    "language": language,
                    "vtt_files": vtt_files
                }
            )
            
            return created_vtt_files
            
        except Exception as e:
            logger.error(f"Failed to transcribe meeting: {e}")
            # Update metadata to show failure
            if project_id and meeting_id:
                self.metadata_manager.update_meeting_metadata(
                    project_id,
                    meeting_id,
                    {
                        "transcription_status": "failed",
                        "transcription_error": str(e),
                        "transcription_end": datetime.utcnow().isoformat()
                    }
                )
            raise

def monitor_wav_files(transcriber: WhisperTranscriber, project_id: str, meeting_id: str):
    """Monitor meeting directory for new WAV files.
    
    Args:
        transcriber: WhisperTranscriber instance
        project_id: Project ID
        meeting_id: Meeting ID
    """
    meeting_dir = transcriber.metadata_manager.get_meeting_dir(project_id, meeting_id)
    processed_files: Set[Path] = set()
    
    print(f"\nMonitoring {meeting_dir} for new WAV files...")
    print("Press Ctrl+C to stop monitoring.")
    
    try:
        while True:
            # Check for new WAV files
            wav_files = set(meeting_dir.glob("*.wav"))
            new_files = wav_files - processed_files
            
            for wav_path in new_files:
                vtt_path = wav_path.with_suffix('.vtt')
                
                # Only transcribe if VTT doesn't exist
                if not vtt_path.exists():
                    logger.info(f"New WAV file detected: {wav_path}")
                    try:
                        # Get relative path for metadata
                        rel_wav_path = wav_path.relative_to(transcriber.metadata_manager.data_dir)
                        
                        # Update metadata with new WAV file
                        metadata = transcriber.metadata_manager.get_meeting_metadata(
                            project_id,
                            meeting_id
                        )
                        recording_files = metadata.get("recording_files", [])
                        if str(rel_wav_path) not in recording_files:
                            recording_files.append(str(rel_wav_path))
                            transcriber.metadata_manager.update_meeting_metadata(
                                project_id,
                                meeting_id,
                                {"recording_files": recording_files}
                            )
                        
                        # Transcribe file
                        transcriber.transcribe_file(wav_path)
                        print(f"Transcribed: {wav_path}")
                    except Exception as e:
                        logger.error(f"Failed to handle new WAV file: {e}")
                        print(f"Error processing {wav_path}: {e}")
                
                processed_files.add(wav_path)
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\nStopped monitoring.")

def select_meeting(metadata_manager: UnifiedMetadataManager, project_id: str) -> str:
    """Prompt user to select a meeting from the last 10 meetings.
    
    Args:
        metadata_manager: UnifiedMetadataManager instance
        project_id: Project ID
        
    Returns:
        Selected meeting ID
    """
    meetings = metadata_manager.get_meeting_metadata(project_id)
    
    # Sort meetings by last modified date
    sorted_meetings = sorted(
        meetings.items(),
        key=lambda x: x[1].get('last_modified', ''),
        reverse=True
    )[:10]
    
    if not sorted_meetings:
        print("No meetings found.")
        sys.exit(1)
    
    print("\nSelect a meeting (0-9):")
    for i, (meeting_id, metadata) in enumerate(sorted_meetings):
        last_modified = metadata.get('last_modified', 'Unknown date')
        if isinstance(last_modified, str):
            try:
                last_modified = datetime.fromisoformat(last_modified).strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        print(f"[{i}] {meeting_id} (Last modified: {last_modified})")
    
    while True:
        try:
            choice = input("\nEnter selection (0-9): ")
            index = int(choice)
            if 0 <= index < len(sorted_meetings):
                return sorted_meetings[index][0]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 9.")

def main(
    meeting_id: Optional[str] = None,
    project_id: Optional[str] = None,
    language: str = WHISPER_CONFIG["language"],
    monitor: bool = False
):
    """Transcribe meeting recordings to VTT format.
    
    Args:
        meeting_id: ID or directory name of the meeting to transcribe
        project_id: Optional project ID (uses default project if not provided)
        language: Language for transcription (default: from config)
        monitor: Whether to monitor the meeting directory for new WAV files
    """
    try:
        transcriber = WhisperTranscriber(language=language)
        
        # Get project ID if not provided
        if not project_id:
            project = transcriber.metadata_manager.get_project()
            project_id = project["key"]
            print(f"Using default project: {project_id}")
        
        # Get meeting ID if not provided
        if not meeting_id:
            meeting_id = select_meeting(transcriber.metadata_manager, project_id)
        
        print(f"\nStarting transcription for meeting: {meeting_id}")
        print(f"Project: {project_id}")
        print(f"Language: {language}")
        
        # Start file monitoring if requested
        if monitor:
            monitor_wav_files(transcriber, project_id, meeting_id)
            return
        
        # Otherwise, just transcribe existing files
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
