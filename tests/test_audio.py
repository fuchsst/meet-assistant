"""Test suite for audio recording functionality."""
import pytest
import wave
import numpy as np
from pathlib import Path
import tempfile

from core.audio.recorder import AudioRecorder
from core.audio.audio_monitor import AudioMonitor
from core.audio.voice_generator import VoiceGenerator
from config.config import VOICE_CONFIG

def test_audio_recorder_initialization():
    """Test AudioRecorder initialization."""
    recorder = AudioRecorder()
    assert recorder is not None
    assert recorder.stream is None
    assert not recorder.is_recording
    assert len(recorder.frames) == 0

def test_recording_start_stop(test_dir):
    """Test starting and stopping recording."""
    recorder = AudioRecorder()
    
    # Start recording
    recorder.start_recording()
    assert recorder.is_recording
    assert recorder.stream is not None
    assert recorder.stream.is_active()
    
    # Stop recording
    recorder.stop_recording()
    assert not recorder.is_recording
    assert recorder.stream is None

def test_save_recording(test_dir):
    """Test saving recording to file."""
    recorder = AudioRecorder()
    output_file = test_dir / "test_recording.wav"
    
    # Record some audio
    recorder.start_recording()
    import time
    time.sleep(1)  # Record for 1 second
    recorder.stop_recording()
    
    # Save recording
    recorder.save_recording(output_file)
    
    # Verify file exists and contains valid audio
    assert output_file.exists()
    with wave.open(str(output_file), 'rb') as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() > 0

def test_audio_validation(mock_audio_data):
    """Test audio validation functionality."""
    recorder = AudioRecorder()
    
    # Test with valid audio
    recorder.frames = [mock_audio_data]
    assert recorder.validate_audio()
    
    # Test with silence
    silent_data = np.zeros(16000, dtype=np.int16).tobytes()
    recorder.frames = [silent_data]
    assert not recorder.validate_audio()
    
    # Test with clipping
    clipping_data = np.ones(16000, dtype=np.int16) * 32767
    recorder.frames = [clipping_data.tobytes()]
    assert not recorder.validate_audio()

def test_audio_monitor_initialization():
    """Test AudioMonitor initialization."""
    monitor = AudioMonitor()
    assert monitor is not None
    assert not monitor.is_monitoring
    assert monitor.input_stream is None
    assert len(monitor.input_frames) == 0

def test_monitor_start_stop():
    """Test starting and stopping audio monitoring."""
    monitor = AudioMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    assert monitor.is_monitoring
    assert monitor.input_stream is not None
    assert monitor.input_stream.is_active()
    
    # Stop monitoring
    monitor.stop_monitoring()
    assert not monitor.is_monitoring
    assert monitor.input_stream is None

def test_silence_detection(mock_audio_data):
    """Test silence detection in audio monitor."""
    monitor = AudioMonitor()
    
    # Test with normal audio
    assert not monitor._is_silence(mock_audio_data)
    
    # Test with silence
    silent_data = np.zeros(16000, dtype=np.int16).tobytes()
    assert monitor._is_silence(silent_data)

@pytest.mark.integration
def test_audio_processing_pipeline(test_dir, mock_audio_data):
    """Test complete audio processing pipeline."""
    recorder = AudioRecorder()
    monitor = AudioMonitor()
    output_file = test_dir / "pipeline_test.wav"
    
    # Record and monitor
    recorder.start_recording()
    monitor.start_monitoring()
    
    # Simulate audio processing
    import time
    time.sleep(1)
    
    # Stop recording and monitoring
    recorder.stop_recording()
    monitor.stop_monitoring()
    
    # Save and verify
    recorder.save_recording(output_file)
    assert output_file.exists()

def test_voice_generator_initialization():
    """Test VoiceGenerator initialization."""
    # Test default initialization
    generator = VoiceGenerator()
    assert generator is not None
    assert not generator.is_playing
    assert generator.current_audio is None
    assert generator.voice_id in VOICE_CONFIG["voices"]
    
    # Test initialization with specific voice
    generator = VoiceGenerator(voice_id="nature")
    assert generator.voice_id == "nature"
    assert generator.voice_config == VOICE_CONFIG["voices"]["nature"]

def test_voice_switching():
    """Test switching between different voices."""
    generator = VoiceGenerator(voice_id="nature")
    
    # Switch to morgan_freeman voice
    generator.set_voice("morgan_freeman")
    assert generator.voice_id == "morgan_freeman"
    assert generator.voice_config == VOICE_CONFIG["voices"]["morgan_freeman"]
    
    # Switch to nature voice
    generator.set_voice("nature")
    assert generator.voice_id == "nature"
    assert generator.voice_config == VOICE_CONFIG["voices"]["nature"]
    
    # Test invalid voice
    with pytest.raises(ValueError):
        generator.set_voice("nonexistent_voice")

def test_voice_generation():
    """Test voice generation functionality."""
    generator = VoiceGenerator()
    test_text = "This is a test of voice generation."
    
    # Test generating speech to memory
    generator.generate_speech(test_text)
    assert generator.current_audio is not None
    assert isinstance(generator.current_audio, np.ndarray)
    
    # Test generating speech to file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            output_path = generator.generate_speech(test_text, temp_path)
            assert output_path == temp_path
            assert temp_path.exists()
            
            # Verify the generated audio file
            with wave.open(str(temp_path), 'rb') as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == 2
                assert wav_file.getframerate() == VOICE_CONFIG["sample_rate"]
                assert wav_file.getnframes() > 0
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()

def test_voice_playback_control():
    """Test voice playback controls."""
    generator = VoiceGenerator()
    test_text = "Testing playback controls."
    
    # Generate audio in memory
    generator.generate_speech(test_text)
    assert generator.current_audio is not None
    
    # Start playback
    import threading
    playback_thread = threading.Thread(target=generator.play_audio)
    playback_thread.start()
    
    # Verify playing state
    import time
    time.sleep(0.1)
    assert generator.is_playing
    
    # Stop playback
    generator.stop_playback()
    assert not generator.is_playing
    
    # Wait for thread to complete
    playback_thread.join(timeout=1)

@pytest.mark.integration
def test_voice_preview():
    """Test voice preview functionality."""
    generator = VoiceGenerator()
    
    # Test preview with default text
    generator.preview_voice()
    assert generator.current_audio is not None
    
    # Test preview with custom text
    custom_text = "This is a custom preview message."
    generator.preview_voice(custom_text)
    assert generator.current_audio is not None

@pytest.mark.integration
def test_voice_config_changes():
    """Test voice configuration changes."""
    generator = VoiceGenerator()
    test_text = "Testing voice configuration."
    
    # Test with different voices
    for voice_id in VOICE_CONFIG["voices"]:
        generator.set_voice(voice_id)
        assert generator.voice_id == voice_id
        generator.generate_speech(test_text)
        assert generator.current_audio is not None
    
    # Test speed changes
    original_speed = generator.speed
    generator.speed = 1.5
    assert generator.speed == 1.5
    generator.generate_speech(test_text)
    assert generator.current_audio is not None
    generator.speed = original_speed
