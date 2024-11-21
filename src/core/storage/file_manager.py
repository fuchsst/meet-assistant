"""File management functionality for the Meeting Assistant."""
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

from config.config import MEETINGS_DIR, FILE_PATTERNS

logger = logging.getLogger(__name__)

class FileManager:
    """Handles file operations and directory management."""

    def __init__(self):
        """Initialize the file manager."""
        self.meetings_dir = MEETINGS_DIR
        self.meetings_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.meetings_dir / FILE_PATTERNS["metadata"]

    def generate_meeting_id(self) -> str:
        """Generate a unique meeting ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"meeting_{timestamp}_{unique_id}"

    def create_meeting_directory(self, meeting_id: str) -> Path:
        """Create directory structure for a new meeting."""
        meeting_dir = self.meetings_dir / meeting_id
        meeting_dir.mkdir(exist_ok=True)
        return meeting_dir

    def save_metadata(self, meeting_id: str, metadata: Dict[str, Any]) -> None:
        """Save or update meeting metadata."""
        try:
            # Load existing metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {"meetings": {}}

            # Update metadata for this meeting
            all_metadata["meetings"][meeting_id] = {
                **metadata,
                "last_modified": datetime.now().isoformat()
            }

            # Save updated metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, indent=2)

            logger.info(f"Updated metadata for meeting {meeting_id}")

        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise

    def get_metadata(self, meeting_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a specific meeting or all meetings."""
        try:
            if not self.metadata_file.exists():
                return {"meetings": {}}

            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)

            if meeting_id:
                return all_metadata["meetings"].get(meeting_id, {})
            return all_metadata

        except Exception as e:
            logger.error(f"Failed to read metadata: {str(e)}")
            raise

    def save_transcript(self, meeting_id: str, text: str, mode: str = 'a') -> None:
        """Save transcript text."""
        meeting_dir = self.meetings_dir / meeting_id
        transcript_file = meeting_dir / FILE_PATTERNS["transcript"]

        try:
            with open(transcript_file, mode, encoding='utf-8') as f:
                f.write(f"{text}\n")

            logger.debug(f"Saved transcript for meeting {meeting_id}")

        except Exception as e:
            logger.error(f"Failed to save transcript: {str(e)}")
            raise

    def save_analysis(self, meeting_id: str, analysis: Dict[str, Any]) -> None:
        """Save analysis results."""
        meeting_dir = self.meetings_dir / meeting_id
        analysis_file = meeting_dir / FILE_PATTERNS["analysis"]

        try:
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2)

            logger.info(f"Saved analysis for meeting {meeting_id}")

        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            raise

    def save_audio(self, meeting_id: str, audio_data: bytes) -> None:
        """Save audio recording."""
        meeting_dir = self.meetings_dir / meeting_id
        audio_file = meeting_dir / FILE_PATTERNS["audio"]

        try:
            with open(audio_file, 'wb') as f:
                f.write(audio_data)

            logger.info(f"Saved audio for meeting {meeting_id}")

        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}")
            raise

    def create_backup(self, meeting_id: str) -> Path:
        """Create a backup of meeting data."""
        meeting_dir = self.meetings_dir / meeting_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.meetings_dir / "backups" / f"{meeting_id}_{timestamp}"

        try:
            # Create backups directory if it doesn't exist
            backup_dir.parent.mkdir(parents=True, exist_ok=True)

            # Copy meeting directory to backup
            shutil.copytree(meeting_dir, backup_dir)

            logger.info(f"Created backup of meeting {meeting_id} at {backup_dir}")
            return backup_dir

        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise

    def list_meetings(self) -> Dict[str, Dict[str, Any]]:
        """List all meetings with their metadata."""
        return self.get_metadata()["meetings"]

    def get_meeting_files(self, meeting_id: str) -> Dict[str, Path]:
        """Get paths to all files associated with a meeting."""
        meeting_dir = self.meetings_dir / meeting_id
        return {
            "audio": meeting_dir / FILE_PATTERNS["audio"],
            "transcript": meeting_dir / FILE_PATTERNS["transcript"],
            "analysis": meeting_dir / FILE_PATTERNS["analysis"]
        }

    def delete_meeting(self, meeting_id: str, backup: bool = True) -> None:
        """Delete a meeting and optionally create a backup."""
        meeting_dir = self.meetings_dir / meeting_id

        try:
            # Create backup if requested
            if backup:
                self.create_backup(meeting_id)

            # Remove meeting directory
            shutil.rmtree(meeting_dir)

            # Update metadata
            metadata = self.get_metadata()
            if meeting_id in metadata["meetings"]:
                del metadata["meetings"][meeting_id]
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Deleted meeting {meeting_id}")

        except Exception as e:
            logger.error(f"Failed to delete meeting: {str(e)}")
            raise
