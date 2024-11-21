"""Test suite for file management functionality."""
import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime

from core.storage.file_manager import FileManager

def test_file_manager_initialization(test_dir):
    """Test FileManager initialization."""
    manager = FileManager()
    assert manager is not None
    assert manager.meetings_dir.exists()
    assert manager.metadata_file.name == "metadata.json"

def test_meeting_id_generation():
    """Test meeting ID generation."""
    manager = FileManager()
    
    # Generate multiple IDs
    ids = [manager.generate_meeting_id() for _ in range(5)]
    
    # Verify uniqueness
    assert len(set(ids)) == len(ids), "Meeting IDs should be unique"
    
    # Verify format (meeting_YYYYMMDD_HHMMSS_uuid)
    for meeting_id in ids:
        assert meeting_id.startswith("meeting_")
        parts = meeting_id.split("_")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # UUID part

def test_directory_creation(test_dir):
    """Test meeting directory creation."""
    manager = FileManager()
    meeting_id = manager.generate_meeting_id()
    
    # Create directory
    meeting_dir = manager.create_meeting_directory(meeting_id)
    
    # Verify directory exists
    assert meeting_dir.exists()
    assert meeting_dir.is_dir()
    assert meeting_dir.name == meeting_id

def test_metadata_management(test_dir):
    """Test metadata handling."""
    manager = FileManager()
    meeting_id = manager.generate_meeting_id()
    
    # Test metadata creation
    test_metadata = {
        "title": "Test Meeting",
        "participants": ["Alice", "Bob"],
        "date": datetime.now().isoformat()
    }
    
    manager.save_metadata(meeting_id, test_metadata)
    
    # Verify metadata was saved
    saved_metadata = manager.get_metadata(meeting_id)
    assert saved_metadata["title"] == test_metadata["title"]
    assert saved_metadata["participants"] == test_metadata["participants"]
    
    # Test metadata update
    updated_metadata = {**test_metadata, "title": "Updated Title"}
    manager.save_metadata(meeting_id, updated_metadata)
    
    # Verify update
    saved_metadata = manager.get_metadata(meeting_id)
    assert saved_metadata["title"] == "Updated Title"

def test_transcript_handling(test_dir, mock_transcription):
    """Test transcript file operations."""
    manager = FileManager()
    meeting_id = manager.generate_meeting_id()
    manager.create_meeting_directory(meeting_id)
    
    # Test saving transcript
    manager.save_transcript(meeting_id, mock_transcription)
    
    # Verify transcript file
    meeting_files = manager.get_meeting_files(meeting_id)
    assert meeting_files["transcript"].exists()
    
    # Verify content
    with open(meeting_files["transcript"], 'r', encoding='utf-8') as f:
        content = f.read().strip()
        assert content == mock_transcription

def test_analysis_handling(test_dir, mock_analysis_result):
    """Test analysis file operations."""
    manager = FileManager()
    meeting_id = manager.generate_meeting_id()
    manager.create_meeting_directory(meeting_id)
    
    # Save analysis
    manager.save_analysis(meeting_id, mock_analysis_result)
    
    # Verify analysis file
    meeting_files = manager.get_meeting_files(meeting_id)
    assert meeting_files["analysis"].exists()
    
    # Verify content
    with open(meeting_files["analysis"], 'r', encoding='utf-8') as f:
        saved_analysis = json.load(f)
        assert saved_analysis == mock_analysis_result

def test_backup_functionality(test_dir, mock_meeting_dir):
    """Test backup creation and restoration."""
    manager = FileManager()
    meeting_id = mock_meeting_dir.name
    
    # Create backup
    backup_dir = manager.create_backup(meeting_id)
    
    # Verify backup
    assert backup_dir.exists()
    assert (backup_dir / "audio.wav").exists()
    assert (backup_dir / "transcript.txt").exists()
    assert (backup_dir / "analysis.json").exists()
    
    # Verify backup content matches original
    original_files = list(mock_meeting_dir.glob("*"))
    backup_files = list(backup_dir.glob("*"))
    assert len(original_files) == len(backup_files)

def test_meeting_listing(test_dir):
    """Test listing meetings."""
    manager = FileManager()
    
    # Create test meetings
    meeting_ids = []
    for i in range(3):
        meeting_id = manager.generate_meeting_id()
        meeting_ids.append(meeting_id)
        manager.create_meeting_directory(meeting_id)
        manager.save_metadata(meeting_id, {
            "title": f"Test Meeting {i}",
            "date": datetime.now().isoformat()
        })
    
    # List meetings
    meetings = manager.list_meetings()
    
    # Verify all meetings are listed
    assert len(meetings) >= len(meeting_ids)
    for meeting_id in meeting_ids:
        assert meeting_id in meetings

def test_file_cleanup(test_dir):
    """Test meeting deletion."""
    manager = FileManager()
    meeting_id = manager.generate_meeting_id()
    manager.create_meeting_directory(meeting_id)
    
    # Add some test files
    manager.save_metadata(meeting_id, {"title": "Test Meeting"})
    manager.save_transcript(meeting_id, "Test transcript")
    
    # Delete meeting
    manager.delete_meeting(meeting_id, backup=True)
    
    # Verify meeting directory is removed
    meeting_dir = manager.meetings_dir / meeting_id
    assert not meeting_dir.exists()
    
    # Verify backup was created
    backup_dirs = list(manager.meetings_dir.glob(f"backups/{meeting_id}_*"))
    assert len(backup_dirs) > 0

@pytest.mark.integration
def test_complete_meeting_lifecycle(test_dir):
    """Test complete meeting lifecycle."""
    manager = FileManager()
    
    # Create meeting
    meeting_id = manager.generate_meeting_id()
    manager.create_meeting_directory(meeting_id)
    
    # Add metadata
    metadata = {
        "title": "Lifecycle Test Meeting",
        "participants": ["Alice", "Bob"],
        "date": datetime.now().isoformat()
    }
    manager.save_metadata(meeting_id, metadata)
    
    # Add transcript
    manager.save_transcript(meeting_id, "Test transcript")
    
    # Add analysis
    analysis = {
        "summary": "Test summary",
        "action_items": ["Test action"]
    }
    manager.save_analysis(meeting_id, analysis)
    
    # Create backup
    backup_dir = manager.create_backup(meeting_id)
    
    # Delete meeting
    manager.delete_meeting(meeting_id, backup=True)
    
    # Verify cleanup
    assert not (manager.meetings_dir / meeting_id).exists()
    assert backup_dir.exists()

@pytest.mark.performance
def test_file_operation_performance(test_dir):
    """Test performance of file operations."""
    manager = FileManager()
    meeting_id = manager.generate_meeting_id()
    manager.create_meeting_directory(meeting_id)
    
    import time
    
    # Test metadata operation speed
    start_time = time.time()
    for i in range(100):
        manager.save_metadata(meeting_id, {"test": f"data_{i}"})
    metadata_time = time.time() - start_time
    
    # Test transcript operation speed
    start_time = time.time()
    for i in range(100):
        manager.save_transcript(meeting_id, f"test_{i}")
    transcript_time = time.time() - start_time
    
    # Verify performance
    assert metadata_time < 5.0, "Metadata operations too slow"
    assert transcript_time < 5.0, "Transcript operations too slow"
