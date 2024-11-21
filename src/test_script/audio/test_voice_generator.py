"""Test script for VoiceGenerator functionality."""
import logging
from pathlib import Path
import time

from config.config import DATA_DIR, VOICE_CONFIG
from voice_generator import VoiceGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_voice_generator():
    """Test voice generation and playback functionality."""
    generator = VoiceGenerator()
    test_data_dir = DATA_DIR / "test_voice"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Test 1: Basic Speech Generation
        logger.info("Testing basic speech generation...")
        test_text = "This is a test of the voice generation system."
        output_file = test_data_dir / "test_speech.wav"
        
        generator.generate_speech(test_text, output_file)
        assert output_file.exists(), "Speech file was not created"
        
        logger.info("Basic speech generation successful")

        # Test 2: In-Memory Generation
        logger.info("\nTesting in-memory speech generation...")
        generator.generate_speech(test_text)
        assert generator.current_audio is not None, "In-memory audio not generated"
        
        logger.info("In-memory generation successful")

        # Test 3: Audio Playback
        logger.info("\nTesting audio playback...")
        logger.info("Playing from file (3 seconds)...")
        generator.play_audio(output_file)
        
        logger.info("Playing from memory (3 seconds)...")
        generator.play_audio()
        
        logger.info("Audio playback successful")

        # Test 4: Generate and Play
        logger.info("\nTesting generate and play...")
        generator.generate_and_play("This is a direct generation and playback test.")
        
        logger.info("Generate and play successful")

        # Test 5: Voice Sample Creation
        logger.info("\nTesting voice sample creation...")
        sample_text = "This is a reference voice sample for configuration."
        sample_file = test_data_dir / "voice_sample.wav"
        
        generator.create_voice_sample(sample_text, sample_file)
        assert sample_file.exists(), "Voice sample was not created"
        
        logger.info("Voice sample creation successful")

        # Test 6: Voice Preview
        logger.info("\nTesting voice preview...")
        generator.preview_voice("This is a preview of the voice settings.")
        
        logger.info("Voice preview successful")

        # Test 7: Playback Control
        logger.info("\nTesting playback control...")
        
        # Start playback in background
        generator.generate_speech("This is a long text that we will interrupt.")
        generator.play_audio()
        
        # Wait briefly then stop
        time.sleep(1)
        generator.stop_playback()
        assert not generator.is_playing, "Playback was not stopped"
        
        logger.info("Playback control successful")

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

def test_voice_configurations():
    """Test different voice configurations."""
    generator = VoiceGenerator()
    test_data_dir = DATA_DIR / "test_voice_config"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Test different speeds
        logger.info("Testing different speech speeds...")
        speeds = [0.8, 1.0, 1.2]
        test_text = "Testing different speech speeds."
        
        for speed in speeds:
            generator.speed = speed
            logger.info(f"\nTesting speed {speed}...")
            generator.generate_and_play(test_text)
            time.sleep(1)

        # Reset speed
        generator.speed = VOICE_CONFIG["speed"]
        
        logger.info("Speed tests completed")

        # Test different speakers (if available)
        logger.info("\nTesting available speakers...")
        speakers = ["en_speaker_0", "en_speaker_1"]  # Example speakers
        test_text = "Testing different speaker voices."
        
        for speaker in speakers:
            try:
                generator.speaker = speaker
                logger.info(f"\nTesting speaker {speaker}...")
                generator.generate_and_play(test_text)
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Speaker {speaker} not available: {str(e)}")

        # Reset speaker
        generator.speaker = VOICE_CONFIG["speaker"]
        
        logger.info("Speaker tests completed")

        logger.info("\nAll configuration tests completed successfully!")

    except Exception as e:
        logger.error(f"Configuration test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if test_data_dir.exists():
            import shutil
            shutil.rmtree(test_data_dir)
        logger.info("Test cleanup completed")

if __name__ == "__main__":
    logger.info("Running voice generator tests...")
    test_voice_generator()
    logger.info("\nRunning voice configuration tests...")
    test_voice_configurations()
