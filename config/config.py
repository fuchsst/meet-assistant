"""Configuration settings for the Meeting Assistant."""
import os
from pathlib import Path
import json

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MEETINGS_DIR = DATA_DIR / "meetings"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Create necessary directories
for directory in [DATA_DIR, MEETINGS_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Audio device settings file
AUDIO_DEVICES_CONFIG = DATA_DIR / "audio_devices.json"

# Default audio device settings
DEFAULT_AUDIO_DEVICES = {
    "input_devices": [],  # List of input device indices to record from
    "output_devices": [], # List of output device names for loopback recording
    "record_output": True # Flag to control output recording
}

# Load or create audio device settings
if AUDIO_DEVICES_CONFIG.exists():
    try:
        with open(AUDIO_DEVICES_CONFIG, 'r') as f:
            AUDIO_DEVICES = json.load(f)
            # Ensure record_output exists in loaded config
            if 'record_output' not in AUDIO_DEVICES:
                AUDIO_DEVICES['record_output'] = True
    except Exception:
        AUDIO_DEVICES = DEFAULT_AUDIO_DEVICES
else:
    AUDIO_DEVICES = DEFAULT_AUDIO_DEVICES

# Audio settings
AUDIO_CONFIG = {
    "format": "wav",
    "channels": 1,  # Mono audio for better transcription
    "rate": 16000,  # Required sample rate for Whisper
    "chunk": 1024,  # Reduced chunk size for better overlap handling
    "sample_width": 2,  # 16-bit
    "silence_threshold": 100,  # Lowered to better detect speech
    "silence_duration": 0.5,   # Reduced to capture speech more quickly
    "max_chunk_duration": 30.0, # Matched to Whisper's chunk_length
    # Audio gain settings
    "min_gain_db": 0.0,    # Minimum gain in decibels
    "max_gain_db": 30.0,   # Maximum gain in decibels
    "default_gain_db": 15.0, # Increased for better speech detection
    # Device settings
    "devices": AUDIO_DEVICES
}


# Whisper settings
WHISPER_CONFIG = {
    "model_size": "medium",  # options: tiny, base, small, medium, large, turbo
    "language": "en",  # Changed to auto-detect for better flexibility
    "task": "transcribe",
    "chunk_length": 30,  # seconds per chunk for better transcription
    "min_speech_duration": 0.3,  # Reduced to catch shorter utterances
    # Available languages with their display names
    "available_languages": {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "pl": "Polish",
        "ja": "Japanese",
        "zh": "Chinese",
        "ko": "Korean",
        "ru": "Russian",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "ar": "Arabic",
        "hi": "Hindi",
        "th": "Thai",
        "id": "Indonesian"
    }
}

# File patterns
FILE_PATTERNS = {
    "audio": "audio.wav",
    "transcript": "transcript.txt",
    "analysis": "analysis.json",
    "metadata": "metadata.json",
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "meeting_assistant.log"),
            "formatter": "json",
            "level": "DEBUG"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True
        }
    }
}

# UI settings
UI_CONFIG = {
    "page_title": "Meeting Assistant",
    "page_icon": "👤",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Voice settings
VOICE_CONFIG = {
    # Common settings
    "sample_rate": 24000,  # F5-TTS uses 24kHz sample rate
    "speed": 1.0,
    "model_type": "F5-TTS",  # F5-TTS or E2-TTS
    "vocoder": "vocos",  # vocos or bigvgan
    "remove_silence": True,  # Remove silence from generated audio
    
    # Available voices
    "voices": {
        "nature": {
            "name": "Nature Voice",
            "ref_audio": str(MODELS_DIR / "voice_samples" / "nature.wav"),
            "ref_text": "Some call me nature, others call me mother nature.",
            "description": "A calm and soothing nature-themed voice"
        },
        "morgan_freeman": {
            "name": "Morgan Freeman",
            "ref_audio": str(MODELS_DIR / "voice_samples" / "morgan_freeman.mp3"),
            "ref_text": "It tells me what's right and what's not. When to leave and where to go. It's not Shakespear. It does not speak in memorable lines. My inner voice always gives it to me straight!",
            "description": "The warm and friendly narator voice of Morgan Freeman"
        }
    },
    
    # Default voice to use
    "default_voice": "morgan_freeman"
}

# Analysis settings
ANALYSIS_CONFIG = {
    "min_segment_length": 10,  # minimum seconds for a topic segment
    "max_summary_tokens": 500,
    "max_context_window": 4000,
    "temperature": 0.7,
}

# Cache settings
CACHE_CONFIG = {
    "ttl": 3600,  # Time to live in seconds
    "max_size": 100,  # Maximum number of items in cache
}

# LLM settings
LLM_CONFIG = {
    "provider": "anthropic",  # The LLM provider to use
    "model": "claude-3-5-sonnet-20241022",  # The model to use
    "temperature": 0.7,  # Controls randomness in responses
    "max_tokens": 1000,  # Maximum length of generated responses
    "top_p": 0.9,  # Controls diversity in responses
    "frequency_penalty": 0.0,  # Reduces repetition of token sequences
    "presence_penalty": 0.0,  # Reduces repetition of topics
}
