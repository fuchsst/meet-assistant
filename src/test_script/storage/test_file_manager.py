"""Test script for FileManager functionality."""
import json
import logging
from pathlib import Path
import shutil

from config.config import DATA_DIR
from file_manager import FileManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_file_operations():
    """Test basic file operations."""
    # Initialize
    manager = FileManager()
    test_data_dir = DATA_DIR / "test_file_manager"
    
    try:
        # Test 1: Meeting Creation
        logger.info("Testing meeting creation...")
        meeting_id = manager.generate_meeting_id()
        meeting_dir = manager.create_meeting_directory(meeting_id)
        
        assert meeting_dir.exists(), "Meeting directory was not created"
        logger.info(f"Created meeting directory: {meeting_dir}")

        # Test 2: Metadata Management
        logger.info("\nTesting metadata management...")
        test_metadata = {
            "title": "Test Meeting",
            "participants": ["Alice", "Bob"],
            "date": "2024-03-19",
            "duration": 3600
        }
        
        manager.save_metadata(meeting_id, test_metadata)
        retrieved_metadata = manager.get_metadata(meeting_id)
        
        assert retrieved_metadata["title"] == test_metadata["title"], "Metadata mismatch"
        logger.info("Metadata saved and retrieved successfully")

        # Test 3: File Saving
        logger.info("\nTesting file saving...")
        
        # Test transcript
        test_transcript = "[00:00:00] This is a test transcript."
        manager.save_transcript(meeting_id, test_transcript)
        
        transcript_file = meeting_dir / "transcript.txt"
        assert transcript_file.exists(), "Transcript file was not created"
        
        # Test analysis
        test_analysis = {
            "summary": "Test meeting summary",
            "action_items": ["Task 1", "Task 2"]
        }
        manager.save_analysis(meeting_id, test_analysis)
        
        analysis_file = meeting_dir / "analysis.json"
        assert analysis_file.exists(), "Analysis file was not created"
        
        # Test audio
        test_audio = b"dummy audio data"
        manager.save_audio(meeting_id, test_audio)
        
        audio_file = meeting_dir / "audio.wav"
        assert audio_file.exists(), "Audio file was not created"
        
        logger.info("All files saved successfully")

        # Test 4: Backup Creation
        logger.info("\nTesting backup functionality...")
        backup_dir = manager.create_backup(meeting_id)
        
        assert backup_dir.exists(), "Backup directory was not created"
        assert (backup_dir / "transcript.txt").exists(), "Backup missing transcript"
        assert (backup_dir / "analysis.json").exists(), "Backup missing analysis"
        assert (backup_dir / "audio.wav").exists(), "Backup missing audio"
        
        logger.info(f"Backup created at: {backup_dir}")

        # Test 5: Meeting Listing
        logger.info("\nTesting meeting listing...")
        meetings = manager.list_meetings()
        
        assert meeting_id in meetings, "Meeting not found in listing"
        logger.info(f"Found {len(meetings)} meetings")

        # Test 6: File Retrieval
        logger.info("\nTesting file path retrieval...")
        files = manager.get_meeting_files(meeting_id)
        
        assert all(path.exists() for path in files.values()), "Some meeting files are missing"
        logger.info("All meeting files are accessible")

        # Test 7: Meeting Deletion
        logger.info("\nTesting meeting deletion...")
        manager.delete_meeting(meeting_id, backup=True)
        
        assert not meeting_dir.exists(), "Meeting directory still exists after deletion"
        logger.info("Meeting deleted successfully")

        logger.info("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        logger.info("Test cleanup completed")

if __name__ == "__main__":
    test_file_operations()
