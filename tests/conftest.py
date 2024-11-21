"""Test configuration and fixtures for Meeting Assistant."""
import pytest
import tempfile
import shutil
from pathlib import Path
import wave
import numpy as np
from config.config import VOICE_CONFIG

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_audio_file(test_dir):
    """Create a test WAV file."""
    audio_file = test_dir / "test_audio.wav"
    
    # Generate 1 second of audio data (sine wave)
    sample_rate = 16000
    duration = 1  # seconds
    t = np.linspace(0, duration, duration * sample_rate)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    # Create WAV file
    with wave.open(str(audio_file), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    yield audio_file

@pytest.fixture(scope="session")
def test_transcript_file(test_dir):
    """Create a test transcript file."""
    transcript_file = test_dir / "test_transcript.txt"
    transcript_content = """
    [00:00:00] Alice: Let's begin the meeting.
    [00:00:10] Bob: I'll present the project status.
    [00:00:20] Charlie: Great, let's discuss the timeline.
    """
    transcript_file.write_text(transcript_content)
    yield transcript_file

@pytest.fixture(scope="session")
def test_metadata_file(test_dir):
    """Create a test metadata file."""
    metadata_file = test_dir / "metadata.json"
    metadata_content = {
        "meetings": {
            "meeting_20240319_001": {
                "title": "Test Meeting",
                "date": "2024-03-19",
                "participants": ["Alice", "Bob", "Charlie"],
                "duration": 3600
            }
        }
    }
    import json
    metadata_file.write_text(json.dumps(metadata_content, indent=2))
    yield metadata_file

@pytest.fixture(scope="session")
def test_analysis_file(test_dir):
    """Create a test analysis file."""
    analysis_file = test_dir / "analysis.json"
    analysis_content = {
        "summary": "Test meeting discussion about project status and timeline.",
        "key_points": [
            "Project status presented",
            "Timeline discussed"
        ],
        "action_items": [
            {
                "task": "Update project timeline",
                "assignee": "Bob",
                "due_date": "2024-03-26"
            }
        ],
        "decisions": [
            "Approved timeline changes",
            "Scheduled follow-up meeting"
        ]
    }
    import json
    analysis_file.write_text(json.dumps(analysis_content, indent=2))
    yield analysis_file

@pytest.fixture(scope="function")
def mock_meeting_dir(test_dir, test_audio_file, test_transcript_file, test_analysis_file):
    """Create a mock meeting directory structure."""
    meeting_dir = test_dir / "meetings" / "test_meeting"
    meeting_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy test files to meeting directory
    shutil.copy(test_audio_file, meeting_dir / "audio.wav")
    shutil.copy(test_transcript_file, meeting_dir / "transcript.txt")
    shutil.copy(test_analysis_file, meeting_dir / "analysis.json")
    
    yield meeting_dir
    
    # Cleanup
    if meeting_dir.exists():
        shutil.rmtree(meeting_dir)

@pytest.fixture(scope="function")
def mock_audio_data():
    """Generate mock audio data."""
    sample_rate = 16000
    duration = 0.1  # seconds
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return audio_data.tobytes()

@pytest.fixture(scope="function")
def mock_transcription():
    """Generate mock transcription data."""
    return "[00:00:00] This is a test transcription."

@pytest.fixture(scope="function")
def mock_analysis_result():
    """Generate mock analysis result."""
    return {
        "summary": "Test analysis summary",
        "key_points": ["Point 1", "Point 2"],
        "action_items": [
            {"task": "Test task", "assignee": "Test user"}
        ],
        "decisions": ["Test decision"]
    }

@pytest.fixture(scope="session")
def test_voice_samples(test_dir):
    """Create test voice sample files."""
    voice_dir = test_dir / "voice_samples"
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample audio data
    sample_rate = VOICE_CONFIG["sample_rate"]
    duration = 1  # seconds
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    # Create voice sample files
    voice_files = {}
    for voice_id in VOICE_CONFIG["voices"]:
        voice_file = voice_dir / f"{voice_id}.wav"
        with wave.open(str(voice_file), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        voice_files[voice_id] = voice_file
    
    yield voice_files
    
    # Cleanup
    if voice_dir.exists():
        shutil.rmtree(voice_dir)

@pytest.fixture(scope="function")
def mock_voice_config(test_voice_samples):
    """Create a mock voice configuration."""
    config = {
        "sample_rate": VOICE_CONFIG["sample_rate"],
        "speed": VOICE_CONFIG["speed"],
        "model_type": VOICE_CONFIG["model_type"],
        "vocoder": VOICE_CONFIG["vocoder"],
        "remove_silence": VOICE_CONFIG["remove_silence"],
        "voices": {}
    }
    
    # Add voice configurations with test sample paths
    for voice_id, voice_info in VOICE_CONFIG["voices"].items():
        config["voices"][voice_id] = {
            "name": voice_info["name"],
            "ref_audio": str(test_voice_samples[voice_id]),
            "ref_text": voice_info["ref_text"],
            "description": voice_info["description"]
        }
    
    return config

@pytest.fixture(scope="function")
def mock_streamlit_session():
    """Create a mock Streamlit session state."""
    class MockSessionState:
        def __init__(self):
            self.selected_voice = VOICE_CONFIG.get("default_voice", "nature")
    
    return MockSessionState()
