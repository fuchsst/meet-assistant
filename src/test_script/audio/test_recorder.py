"""Test script for AudioRecorder functionality."""
import time
from pathlib import Path
import logging

from config.config import DATA_DIR
from recorder import AudioRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_recording():
    """Test basic recording functionality."""
    recorder = AudioRecorder()
    output_dir = DATA_DIR / "test_recordings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test_recording.wav"

    try:
        # Start recording
        logger.info("Starting recording... (will record for 5 seconds)")
        recorder.start_recording()
        
        # Record for 5 seconds
        time.sleep(5)
        
        # Stop recording
        logger.info("Stopping recording...")
        recorder.stop_recording()
        
        # Validate audio
        logger.info("Validating audio...")
        if recorder.validate_audio():
            logger.info("Audio validation passed")
        else:
            logger.warning("Audio validation failed")
        
        # Save recording
        logger.info(f"Saving recording to {output_file}")
        recorder.save_recording(output_file)
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if recorder.stream:
            recorder.stop_recording()

if __name__ == "__main__":
    test_recording()
