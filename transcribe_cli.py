"""CLI script for transcribing WAV/MP3 files to VTT format using Faster Whisper or AWS Transcribe."""
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional, Dict, List, Set, Protocol
from datetime import datetime, timezone
import fire
import torch
import boto3
from faster_whisper import WhisperModel
import numpy as np
import json
import uuid

from config.config import WHISPER_CONFIG, AWS_TRANSCRIBE_CONFIG, MODELS_DIR
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

def get_local_now() -> datetime:
    """Helper function to get current local time as a timezone-aware datetime object."""
    return datetime.now().astimezone()

def get_media_format(file_path: Path) -> str:
    """Determine media format from file extension."""
    ext = file_path.suffix.lower()
    if ext == '.wav':
        return 'wav'
    elif ext == '.mp3':
        return 'mp3'
    else:
        raise ValueError(f"Unsupported file format: {ext}")

class TranscriberProtocol(Protocol):
    """Protocol defining the interface for transcription classes."""
    
    def transcribe_file(self, audio_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Transcribe a single audio file to VTT."""
        ...
    
    def transcribe_meeting(
        self,
        project_id: Optional[str],
        meeting_id: str,
        language: Optional[str] = None
    ) -> List[Path]:
        """Transcribe audio files listed in meeting metadata."""
        ...

class AWSTranscriber:
    """Handles transcription of audio files using AWS Transcribe."""
    
    def __init__(self, language: Optional[str] = None):
        """Initialize AWS transcriber.
        
        Args:
            language: Language for transcription (default: None, will auto-detect)
        """
        self.language = language or 'en-US'  # AWS Transcribe requires explicit language
        self.metadata_manager = UnifiedMetadataManager()
        
        # Initialize AWS clients
        self.transcribe = boto3.client('transcribe', region_name=AWS_TRANSCRIBE_CONFIG['region'])
        self.s3 = boto3.client('s3', region_name=AWS_TRANSCRIBE_CONFIG['region'])
        
        if not AWS_TRANSCRIBE_CONFIG['input_bucket'] or not AWS_TRANSCRIBE_CONFIG['output_bucket']:
            raise ValueError("AWS S3 buckets must be configured for AWS Transcribe")
    
    def _get_s3_path(self, file_path: Path, bucket_type: str = 'input') -> tuple[str, str]:
        """Get S3 bucket and key for a file, maintaining project structure.
        
        Args:
            file_path: Local file path
            bucket_type: Either 'input' or 'output'
            
        Returns:
            Tuple of (bucket, key)
        """
        # Get relative path from data directory
        rel_path = file_path.relative_to(self.metadata_manager.data_dir)
        # Use relative path as S3 key to maintain project structure
        bucket = AWS_TRANSCRIBE_CONFIG[f'{bucket_type}_bucket']
        key = str(rel_path)
        return bucket, key
    
    def _upload_to_s3(self, file_path: Path) -> str:
        """Upload file to S3 input bucket maintaining project structure."""
        bucket, key = self._get_s3_path(file_path, 'input')
        self.s3.upload_file(str(file_path), bucket, key)
        return f"s3://{bucket}/{key}"
    
    def _convert_to_vtt(self, transcript: dict, output_path: Path) -> None:
        """Convert AWS transcript to VTT format."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                items = transcript['results']['items']
                current_segment = []
                
                for item in items:
                    if item['type'] == 'pronunciation':
                        current_segment.append(item)
                        
                        # Start new segment on punctuation or long pause
                        if len(current_segment) > 0 and (
                            float(item.get('end_time', 0)) - float(current_segment[0].get('start_time', 0)) > 5
                        ):
                            # Write segment
                            start_time = float(current_segment[0]['start_time'])
                            end_time = float(item['end_time'])
                            
                            text = ' '.join(item['alternatives'][0]['content'] 
                                          for item in current_segment)
                            
                            # Format timestamps
                            start = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}"
                            end = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}"
                            
                            f.write(f"{start} --> {end}\n")
                            f.write(f"{text}\n\n")
                            
                            current_segment = []
                            
        except Exception as e:
            logger.error(f"Failed to convert transcript to VTT: {e}")
            raise
    
    def transcribe_file(self, audio_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Transcribe a single audio file to VTT using AWS Transcribe."""
        try:
            logger.info(f"Starting AWS transcription for: {audio_path}")
            
            # Determine output path
            if output_path is None:
                output_path = audio_path.with_suffix('.vtt')
            
            # Upload file to S3
            s3_uri = self._upload_to_s3(audio_path)
            
            # Get media format
            media_format = get_media_format(audio_path)
            
            # Get output path in S3 (same structure as local, but with .json extension)
            output_bucket, output_key = self._get_s3_path(output_path.with_suffix('.json'), 'output')
            
            # Start transcription job
            job_name = f"transcribe_{uuid.uuid4()}"
            self.transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': s3_uri},
                MediaFormat=media_format,
                LanguageCode=self.language,
                OutputBucketName=output_bucket,
                OutputKey=output_key,
                Settings={
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': 10
                }
            )
            
            logger.info("Waiting for transcription job to complete...")
            while True:
                status = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                
                if job_status == 'COMPLETED':
                    logger.info("Transcription completed. Retrieving transcript...")
                    
                    # Get transcript from S3 using the same path structure
                    response = self.s3.get_object(
                        Bucket=output_bucket,
                        Key=output_key
                    )
                    transcript = json.loads(response['Body'].read().decode('utf-8'))
                    
                    logger.info("Successfully retrieved transcript from S3")
                    
                    # Convert to VTT
                    self._convert_to_vtt(transcript, output_path)
                    return output_path
                    
                elif job_status == 'FAILED':
                    failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                    raise Exception(f"Transcription job failed: {failure_reason}")
                
                logger.info(f"Not ready yet... Status: {job_status}")
                time.sleep(5)
            
        except Exception as e:
            logger.error(f"AWS transcription failed: {e}")
            return None
    
    def transcribe_meeting(
        self,
        project_id: Optional[str],
        meeting_id: str,
        language: Optional[str] = None
    ) -> List[Path]:
        """Transcribe audio files listed in meeting metadata using AWS Transcribe."""
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
                        "transcription_end": get_local_now().isoformat()
                    }
                )
                return []
            
            # Update metadata to show transcription started
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "transcription_status": "in_progress",
                    "transcription_start": get_local_now().isoformat(),
                    "language": language or self.language
                }
            )
            
            # Transcribe audio files that don't have corresponding VTT files
            created_vtt_files = []
            for audio_path in recording_files:
                try:
                    # Convert relative path from metadata to absolute path
                    audio_file = self.metadata_manager.data_dir / audio_path
                    if not audio_file.exists():
                        logger.warning(f"Audio file not found: {audio_file}")
                        continue
                    
                    # Check if VTT already exists in metadata
                    vtt_path = str(Path(audio_path).with_suffix('.vtt'))
                    if vtt_path in vtt_files:
                        logger.info(f"VTT file already exists in metadata for {audio_file.name}")
                        created_vtt_files.append(self.metadata_manager.data_dir / vtt_path)
                        continue
                    
                    # Transcribe file
                    vtt_file = self.transcribe_file(audio_file)
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
                    logger.error(f"Failed to transcribe {audio_path}: {e}")
                    continue
            
            # Update meeting metadata with final status
            status = "completed" if created_vtt_files else "failed"
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "transcription_status": status,
                    "transcription_end": get_local_now().isoformat(),
                    "language": self.language,
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
                        "transcription_end": get_local_now().isoformat()
                    }
                )
            raise

class WhisperTranscriber:
    """Handles transcription of audio files using Faster Whisper."""
    
    def __init__(self, language: Optional[str] = None):
        """Initialize transcriber.
        
        Args:
            language: Language for transcription (default: None, will auto-detect)
        """
        self.language = language
        self.metadata_manager = UnifiedMetadataManager()
        
        # Initialize Faster Whisper
        logger.info("Loading Faster Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        # Check if MODELS_DIR exists
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            logger.info(f"Created models directory: {MODELS_DIR}")
        
        try:
            logger.debug(f"Loading Faster Whisper model size: {WHISPER_CONFIG['model_size']}")
            self.model = WhisperModel(
                WHISPER_CONFIG["model_size"],
                device=self.device,
                compute_type=compute_type,
                download_root=str(MODELS_DIR)
            )
            logger.info(f"Faster Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper model: {e}", exc_info=True)
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
            logger.debug(f"Starting to save VTT file: {vtt_path}")
            with open(vtt_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for i, segment in enumerate(segments):
                    start = segment.start
                    end = segment.end
                    text = segment.text.strip()
                    
                    # Format timestamps as HH:MM:SS.mmm
                    start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}"
                    end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.3f}"
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
                    
                    if i % 10 == 0:  # Log progress every 10 segments
                        logger.debug(f"Processed {i+1}/{len(segments)} segments")
            
            logger.info(f"Saved VTT file: {vtt_path}")
            
        except Exception as e:
            logger.error(f"Failed to save VTT file: {e}", exc_info=True)
            raise

    def transcribe_file(self, audio_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Transcribe a single audio file to VTT.
        
        Args:
            audio_path: Path to audio file (WAV or MP3)
            output_path: Optional path for VTT output. If not provided,
                        will create next to audio file with .vtt extension.
        
        Returns:
            Path to created VTT file or None if transcription failed
        """
        try:
            logger.info(f"Starting transcription for: {audio_path}")
            
            # Determine output path
            if output_path is None:
                output_path = audio_path.with_suffix('.vtt')
            
            logger.debug(f"Output path set to: {output_path}")
            
            # Transcribe audio
            try:
                logger.debug("Starting Faster Whisper transcription")
                segments, info = self.model.transcribe(
                    str(audio_path),
                    language=self.language,
                    task="transcribe",
                    beam_size=5,
                    word_timestamps=True
                )
                logger.debug(f"Transcription complete. Detected language: {info.language}")
                
                # Update language if it was auto-detected
                if self.language is None:
                    self.language = info.language
                
            except Exception as e:
                logger.error(f"Failed to transcribe audio: {e}", exc_info=True)
                raise
            
            # Process segments and save VTT
            logger.debug("Processing transcribed segments")
            processed_segments = []
            for segment in segments:
                text = self._preprocess_text(segment.text)
                if text:
                    processed_segments.append(segment)
            
            logger.debug(f"Processed {len(processed_segments)} segments")
            
            logger.debug("Saving VTT file")
            self._save_vtt(output_path, processed_segments)
            logger.debug("VTT file saved successfully")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return None

    def transcribe_meeting(
        self,
        project_id: Optional[str],
        meeting_id: str,
        language: Optional[str] = None
    ) -> List[Path]:
        """Transcribe audio files listed in meeting metadata.
        
        Args:
            project_id: Optional project ID (uses default if not provided)
            meeting_id: Meeting ID or directory name
            language: Language for transcription (default: None, will auto-detect)
            
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
                        "transcription_end": get_local_now().isoformat()
                    }
                )
                return []
            
            # Update metadata to show transcription started
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "transcription_status": "in_progress",
                    "transcription_start": get_local_now().isoformat(),
                    "language": language or "auto"
                }
            )
            
            # Transcribe audio files that don't have corresponding VTT files
            created_vtt_files = []
            for audio_path in recording_files:
                try:
                    # Convert relative path from metadata to absolute path
                    audio_file = self.metadata_manager.data_dir / audio_path
                    if not audio_file.exists():
                        logger.warning(f"Audio file not found: {audio_file}")
                        continue
                    
                    # Check if VTT already exists in metadata
                    vtt_path = str(Path(audio_path).with_suffix('.vtt'))
                    if vtt_path in vtt_files:
                        logger.info(f"VTT file already exists in metadata for {audio_file.name}")
                        created_vtt_files.append(self.metadata_manager.data_dir / vtt_path)
                        continue
                    
                    # Transcribe file
                    vtt_file = self.transcribe_file(audio_file)
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
                    logger.error(f"Failed to transcribe {audio_path}: {e}")
                    continue
            
            # Update meeting metadata with final status
            status = "completed" if created_vtt_files else "failed"
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "transcription_status": status,
                    "transcription_end": get_local_now().isoformat(),
                    "language": self.language or "auto",
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
                        "transcription_end": get_local_now().isoformat()
                    }
                )
            raise

def get_transcriber(language: Optional[str] = None) -> TranscriberProtocol:
    """Get appropriate transcriber based on configuration."""
    if AWS_TRANSCRIBE_CONFIG['enabled']:
        return AWSTranscriber(language=language)
    return WhisperTranscriber(language=language)

def monitor_audio_files(transcriber: TranscriberProtocol, project_id: str, meeting_id: str):
    """Monitor meeting directory for new audio files (WAV/MP3).
    
    Args:
        transcriber: Transcriber instance
        project_id: Project ID
        meeting_id: Meeting ID
    """
    meeting_dir = transcriber.metadata_manager.get_meeting_dir(project_id, meeting_id)
    processed_files: Set[Path] = set()
    
    print(f"\nMonitoring {meeting_dir} for new audio files...")
    print("Press Ctrl+C to stop monitoring.")
    
    try:
        while True:
            # Check for new audio files (both WAV and MP3)
            audio_files = set(meeting_dir.glob("*.wav")) | set(meeting_dir.glob("*.mp3"))
            new_files = audio_files - processed_files
            
            for audio_path in new_files:
                vtt_path = audio_path.with_suffix('.vtt')
                
                # Only transcribe if VTT doesn't exist
                if not vtt_path.exists():
                    logger.info(f"New audio file detected: {audio_path}")
                    try:
                        # Get relative path for metadata
                        rel_audio_path = audio_path.relative_to(transcriber.metadata_manager.data_dir)
                        
                        # Update metadata with new audio file
                        metadata = transcriber.metadata_manager.get_meeting_metadata(
                            project_id,
                            meeting_id
                        )
                        recording_files = metadata.get("recording_files", [])
                        if str(rel_audio_path) not in recording_files:
                            recording_files.append(str(rel_audio_path))
                            transcriber.metadata_manager.update_meeting_metadata(
                                project_id,
                                meeting_id,
                                {"recording_files": recording_files}
                            )
                        
                        # Transcribe file
                        transcriber.transcribe_file(audio_path)
                        print(f"Transcribed: {audio_path}")
                    except Exception as e:
                        logger.error(f"Failed to handle new audio file: {e}")
                        print(f"Error processing {audio_path}: {e}")
                
                processed_files.add(audio_path)
            
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
    language: Optional[str] = None,
    monitor: bool = False,
    use_aws: Optional[bool] = None
):
    """Transcribe meeting recordings to VTT format.
    
    Args:
        meeting_id: ID or directory name of the meeting to transcribe
        project_id: Optional project ID (uses default project if not provided)
        language: Language for transcription (default: from config)
        monitor: Whether to monitor the meeting directory for new audio files
        use_aws: Override to explicitly use AWS Transcribe (True) or Whisper (False).
                If not provided, uses the value from AWS_TRANSCRIBE_CONFIG.
    """
    try:
        # Override AWS config if explicitly specified
        if use_aws is not None:
            AWS_TRANSCRIBE_CONFIG['enabled'] = str(use_aws).lower() == 'true'
            
        transcriber = get_transcriber(language=language)
        
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
        print(f"Using {'AWS Transcribe' if AWS_TRANSCRIBE_CONFIG['enabled'] else 'Whisper'}")
        
        # Start file monitoring if requested
        if monitor:
            monitor_audio_files(transcriber, project_id, meeting_id)
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
