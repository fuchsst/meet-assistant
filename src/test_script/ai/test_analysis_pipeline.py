"""Test script for AnalysisPipeline functionality."""
import logging
from pathlib import Path
import json
import wave
import numpy as np

from config.config import DATA_DIR
from core.storage.file_manager import FileManager
from analysis_pipeline import AnalysisPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio(filepath: Path, duration_seconds: int = 5) -> None:
    """Create a test WAV file with sample audio data."""
    # Generate sample audio data (sine wave)
    sample_rate = 16000
    t = np.linspace(0, duration_seconds, duration_seconds * sample_rate)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # Create WAV file
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

def test_analysis_pipeline():
    """Test the complete analysis pipeline."""
    # Initialize
    pipeline = AnalysisPipeline()
    file_manager = FileManager()
    test_data_dir = DATA_DIR / "test_analysis"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Test 1: Setup Test Meeting
        logger.info("Setting up test meeting...")
        meeting_id = file_manager.generate_meeting_id()
        meeting_dir = file_manager.create_meeting_directory(meeting_id)
        
        # Create test audio file
        audio_file = meeting_dir / "audio.wav"
        create_test_audio(audio_file)
        assert audio_file.exists(), "Test audio file not created"
        
        # Create test transcript
        transcript = """
        [00:00:00] Alice: Let's begin our project planning meeting.
        [00:00:10] Bob: I've prepared the timeline for Q2.
        [00:00:20] Alice: Great. We need to allocate resources.
        [00:00:30] Charlie: I suggest we add two developers.
        [00:00:40] Alice: Agreed. Let's document that decision.
        [00:00:50] Bob: I'll update the project plan accordingly.
        """
        file_manager.save_transcript(meeting_id, transcript)
        
        logger.info("Test meeting setup completed")

        # Test 2: Complete Meeting Processing
        logger.info("\nTesting complete meeting processing...")
        analysis = pipeline.process_meeting(meeting_id)
        
        assert "summary" in analysis, "Missing summary in analysis"
        assert "action_items" in analysis, "Missing action items in analysis"
        assert "decisions" in analysis, "Missing decisions in analysis"
        
        logger.info("Complete meeting processing successful")

        # Test 3: Real-time Segment Processing
        logger.info("\nTesting real-time segment processing...")
        with wave.open(str(audio_file), 'rb') as wav_file:
            chunk_size = wav_file.getframerate()  # 1 second of audio
            audio_data = wav_file.readframes(chunk_size)
            
            result = pipeline.process_segment(meeting_id, audio_data)
            assert "transcript" in result, "Missing transcript in segment result"
            assert "analysis" in result, "Missing analysis in segment result"
        
        logger.info("Real-time segment processing successful")

        # Test 4: Summary Generation
        logger.info("\nTesting summary generation...")
        summary = pipeline.generate_summary(meeting_id)
        
        assert "summary" in summary, "Missing summary"
        assert "key_points" in summary, "Missing key points"
        assert "duration" in summary, "Missing duration"
        
        logger.info("Summary generation successful")

        # Test 5: Task Extraction
        logger.info("\nTesting task extraction...")
        tasks = pipeline.extract_tasks(meeting_id)
        
        assert isinstance(tasks, list), "Tasks should be a list"
        logger.info(f"Extracted {len(tasks)} tasks")

        # Test 6: Decision Identification
        logger.info("\nTesting decision identification...")
        decisions = pipeline.identify_decisions(meeting_id)
        
        assert isinstance(decisions, list), "Decisions should be a list"
        logger.info(f"Identified {len(decisions)} decisions")

        # Test 7: Metadata Updates
        logger.info("\nTesting metadata updates...")
        metadata = file_manager.get_metadata(meeting_id)
        
        assert "last_analyzed" in metadata, "Missing analysis timestamp"
        assert "summary_length" in metadata, "Missing summary length"
        assert "action_items_count" in metadata, "Missing action items count"
        
        logger.info("Metadata updates verified")

        logger.info("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if test_data_dir.exists():
            import shutil
            shutil.rmtree(test_data_dir)
        logger.info("Test cleanup completed")

if __name__ == "__main__":
    test_analysis_pipeline()
