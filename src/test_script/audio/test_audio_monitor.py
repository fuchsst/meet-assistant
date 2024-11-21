"""Test script for AudioMonitor functionality."""
import logging
import time
from pathlib import Path
import wave
import numpy as np

from config.config import DATA_DIR
from audio_monitor import AudioMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio(filepath: Path, duration_seconds: int = 5, include_silence: bool = True) -> None:
    """Create a test WAV file with sample audio data and optional silence."""
    sample_rate = 16000
    t = np.linspace(0, duration_seconds, duration_seconds * sample_rate)
    
    if include_silence:
        # Create audio with alternating sound and silence
        audio_data = np.zeros(len(t), dtype=np.int16)
        # Add sound in the first and third seconds
        audio_data[:sample_rate] = (np.sin(2 * np.pi * 440 * t[:sample_rate]) * 32767).astype(np.int16)
        audio_data[2*sample_rate:3*sample_rate] = (np.sin(2 * np.pi * 440 * t[:sample_rate]) * 32767).astype(np.int16)
    else:
        # Create continuous audio
        audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

def test_audio_monitor():
    """Test AudioMonitor functionality."""
    monitor = AudioMonitor()
    test_data_dir = DATA_DIR / "test_monitor"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Create test files
    continuous_audio = test_data_dir / "continuous.wav"
    silence_audio = test_data_dir / "with_silence.wav"
    create_test_audio(continuous_audio, duration_seconds=5, include_silence=False)
    create_test_audio(silence_audio, duration_seconds=5, include_silence=True)

    try:
        # Test 1: Silence Detection
        logger.info("Testing silence detection...")
        
        # Test with silence
        with wave.open(str(silence_audio), 'rb') as wav_file:
            chunk = wav_file.readframes(wav_file.getnframes())
            is_silence = monitor._is_silence(chunk[16000:32000])  # Check second second (should be silence)
            assert is_silence, "Failed to detect silence"
            
            is_sound = not monitor._is_silence(chunk[:16000])  # Check first second (should be sound)
            assert is_sound, "Incorrectly detected silence in audio"
        
        logger.info("Silence detection working correctly")

        # Test 2: Chunk Processing
        logger.info("\nTesting chunk processing...")
        transcriptions = []
        
        def callback(text: str):
            transcriptions.append(text)
            logger.info(f"Transcribed: {text}")

        # Process continuous audio
        with wave.open(str(continuous_audio), 'rb') as wav_file:
            chunk = wav_file.readframes(wav_file.getnframes())
            monitor._handle_chunk([chunk], callback)
        
        assert len(transcriptions) > 0, "No transcriptions generated"
        logger.info("Chunk processing successful")

        # Test 3: Audio Saving
        logger.info("\nTesting audio saving...")
        test_chunk = test_data_dir / "test_chunk.wav"
        
        with wave.open(str(continuous_audio), 'rb') as wav_file:
            chunk = wav_file.readframes(wav_file.getnframes())
            monitor.save_chunk(chunk, test_chunk)
        
        assert test_chunk.exists(), "Failed to save audio chunk"
        logger.info("Audio saving successful")

        # Test 4: Monitoring Start/Stop
        logger.info("\nTesting monitor control...")
        
        # Start monitoring
        monitor.start_monitoring(callback)
        assert monitor.is_monitoring, "Monitor failed to start"
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring, "Monitor failed to stop"
        
        logger.info("Monitor control successful")

        logger.info("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        monitor.stop_monitoring()
        if test_data_dir.exists():
            import shutil
            shutil.rmtree(test_data_dir)
        logger.info("Test cleanup completed")

if __name__ == "__main__":
    test_audio_monitor()
