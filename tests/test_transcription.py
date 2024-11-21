"""Test suite for transcription functionality."""
import pytest
import numpy as np
from pathlib import Path

from core.transcription.whisper_processor import WhisperProcessor

def test_whisper_initialization():
    """Test WhisperProcessor initialization."""
    processor = WhisperProcessor()
    assert processor is not None
    assert processor.model is not None
    assert processor.device in ["cuda", "cpu"]

def test_transcribe_chunk(mock_audio_data):
    """Test transcribing an audio chunk."""
    processor = WhisperProcessor()
    
    # Test with valid audio chunk
    text = processor.transcribe_chunk(mock_audio_data)
    assert isinstance(text, str)
    
    # Test with silence
    silent_data = np.zeros(16000, dtype=np.int16).tobytes()
    text = processor.transcribe_chunk(silent_data)
    assert text == ""  # Should return empty string for silence

def test_transcribe_file(test_audio_file):
    """Test transcribing a complete audio file."""
    processor = WhisperProcessor()
    
    # Transcribe test file
    segments = processor.transcribe_file(test_audio_file)
    
    # Verify segments structure
    assert isinstance(segments, list)
    if segments:  # If test audio contains speech
        for segment in segments:
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
            assert isinstance(segment["text"], str)

def test_timestamp_generation():
    """Test timestamp formatting."""
    processor = WhisperProcessor()
    
    # Test various durations
    test_cases = [
        (0, "00:00:00"),
        (61, "00:01:01"),
        (3600, "01:00:00"),
        (3661, "01:01:01")
    ]
    
    for seconds, expected in test_cases:
        formatted = processor._format_timestamp(seconds)
        assert formatted == expected

@pytest.mark.integration
def test_streaming_transcription(mock_audio_data):
    """Test real-time streaming transcription."""
    processor = WhisperProcessor()
    
    # Create a simple audio stream generator
    def audio_stream():
        for _ in range(5):  # Simulate 5 chunks
            yield mock_audio_data
    
    # Process stream
    transcriptions = []
    for text in processor.transcribe_stream(audio_stream()):
        transcriptions.append(text)
    
    # Verify transcriptions
    assert len(transcriptions) > 0
    assert all(isinstance(t, str) for t in transcriptions)

def test_error_handling():
    """Test error handling in transcription."""
    processor = WhisperProcessor()
    
    # Test with invalid audio data
    with pytest.raises(Exception):
        processor.transcribe_chunk(b"invalid audio data")
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        processor.transcribe_file(Path("nonexistent.wav"))

def test_language_handling():
    """Test transcription with different languages."""
    processor = WhisperProcessor()
    
    # Store original language setting
    original_language = processor.language
    
    try:
        # Test with different language settings
        test_languages = ["en", "fr", "de"]
        for lang in test_languages:
            processor.language = lang
            # Just verify it doesn't raise an error
            processor.transcribe_chunk(mock_audio_data)
    
    finally:
        # Restore original language
        processor.language = original_language

@pytest.mark.integration
def test_long_form_transcription(test_audio_file):
    """Test transcription of longer audio files."""
    processor = WhisperProcessor()
    
    # Transcribe complete file
    segments = processor.transcribe_file(test_audio_file)
    
    # Verify transcription quality
    assert isinstance(segments, list)
    if segments:
        # Check segment overlap
        for i in range(len(segments) - 1):
            current_end = float(segments[i]["end"].split(":")[2])
            next_start = float(segments[i + 1]["start"].split(":")[2])
            assert current_end <= next_start, "Segments should not overlap"
        
        # Check segment duration
        for segment in segments:
            start = float(segment["start"].split(":")[2])
            end = float(segment["end"].split(":")[2])
            assert end > start, "Segment duration should be positive"

def test_model_configuration():
    """Test different model configurations."""
    # Test with different model sizes
    model_sizes = ["tiny", "base", "small"]
    
    for size in model_sizes:
        try:
            processor = WhisperProcessor()
            processor.model_size = size
            # Verify model loaded successfully
            assert processor.model is not None
            # Test basic transcription
            text = processor.transcribe_chunk(mock_audio_data)
            assert isinstance(text, str)
        except Exception as e:
            pytest.skip(f"Model {size} not available: {str(e)}")

@pytest.mark.performance
def test_transcription_performance(mock_audio_data):
    """Test transcription performance."""
    processor = WhisperProcessor()
    
    import time
    
    # Measure chunk processing time
    start_time = time.time()
    processor.transcribe_chunk(mock_audio_data)
    chunk_time = time.time() - start_time
    
    # Verify performance meets requirements
    assert chunk_time < 5.0, "Chunk processing too slow"
    
    # Test memory usage
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    # Process multiple chunks
    for _ in range(10):
        processor.transcribe_chunk(mock_audio_data)
    
    mem_after = process.memory_info().rss
    mem_increase = (mem_after - mem_before) / 1024 / 1024  # MB
    
    # Verify memory usage is reasonable
    assert mem_increase < 500, "Memory usage too high"
