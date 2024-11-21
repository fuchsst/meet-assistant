"""Error handling utilities for Meeting Assistant."""
import logging
import traceback
from typing import Optional, Type, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

class AppError(Exception):
    """Base exception class for application errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class AudioError(AppError):
    """Audio-related errors."""
    pass

class TranscriptionError(AppError):
    """Transcription-related errors."""
    pass

class StorageError(AppError):
    """File storage and management errors."""
    pass

class AnalysisError(AppError):
    """Analysis-related errors."""
    pass

class ValidationError(AppError):
    """Data validation errors."""
    pass

class ConfigurationError(AppError):
    """Configuration-related errors."""
    pass

# Error messages for common scenarios
ERROR_MESSAGES = {
    # Audio errors
    "AUDIO_DEVICE_NOT_FOUND": "No audio device found. Please check your microphone connection.",
    "AUDIO_PERMISSION_DENIED": "Unable to access microphone. Please check permissions.",
    "AUDIO_RECORDING_FAILED": "Failed to record audio. Please try again.",
    "AUDIO_PLAYBACK_FAILED": "Failed to play audio. Please check your speakers.",
    
    # Transcription errors
    "TRANSCRIPTION_FAILED": "Failed to transcribe audio. Please try again.",
    "MODEL_LOADING_FAILED": "Failed to load Whisper model. Please check your installation.",
    "INVALID_AUDIO_FORMAT": "Invalid audio format. Please use WAV format.",
    
    # Storage errors
    "FILE_NOT_FOUND": "File not found. Please check the file path.",
    "PERMISSION_DENIED": "Permission denied. Please check file permissions.",
    "STORAGE_FULL": "Not enough storage space. Please free up some space.",
    "BACKUP_FAILED": "Failed to create backup. Please try again.",
    
    # Analysis errors
    "ANALYSIS_FAILED": "Failed to analyze meeting. Please try again.",
    "INVALID_TRANSCRIPT": "Invalid transcript format. Please check the input.",
    "LLM_ERROR": "Error communicating with language model. Please try again.",
    
    # Validation errors
    "INVALID_MEETING_ID": "Invalid meeting ID. Please check the input.",
    "INVALID_METADATA": "Invalid meeting metadata. Please check the input.",
    "INVALID_CONFIG": "Invalid configuration. Please check settings.",
    
    # System errors
    "SYSTEM_ERROR": "A system error occurred. Please try again.",
    "NETWORK_ERROR": "Network error. Please check your connection.",
    "RESOURCE_EXHAUSTED": "System resources exhausted. Please try again later."
}

def get_user_message(error: Exception) -> str:
    """Get user-friendly error message."""
    if isinstance(error, AppError):
        return error.message
    
    # Map exception types to error messages
    error_type = type(error).__name__
    if error_type in ERROR_MESSAGES:
        return ERROR_MESSAGES[error_type]
    
    # Default error message
    return "An unexpected error occurred. Please try again."

def handle_error(error: Exception, log_level: str = "error") -> Dict[str, Any]:
    """Handle error and return structured error information."""
    # Get error details
    error_type = type(error).__name__
    error_message = str(error)
    stack_trace = traceback.format_exc()
    
    # Get user-friendly message
    user_message = get_user_message(error)
    
    # Log error
    log_func = getattr(logger, log_level)
    log_func(
        f"{error_type}: {error_message}",
        extra={
            "error_type": error_type,
            "stack_trace": stack_trace,
            "user_message": user_message
        }
    )
    
    # Return structured error info
    return {
        "error_type": error_type,
        "message": user_message,
        "details": error.details if isinstance(error, AppError) else None
    }

def error_handler(error_type: Type[AppError], log_level: str = "error"):
    """Decorator for handling errors in functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Convert to application error if needed
                if not isinstance(e, AppError):
                    e = error_type(str(e))
                
                # Handle error
                return handle_error(e, log_level)
        return wrapper
    return decorator

# Example usage:
# @error_handler(AudioError)
# def start_recording():
#     ...

# @error_handler(TranscriptionError)
# def transcribe_audio():
#     ...

def validate_meeting_id(meeting_id: str) -> None:
    """Validate meeting ID format."""
    if not meeting_id or not meeting_id.startswith("meeting_"):
        raise ValidationError(
            ERROR_MESSAGES["INVALID_MEETING_ID"],
            details={"meeting_id": meeting_id}
        )

def validate_metadata(metadata: Dict[str, Any]) -> None:
    """Validate meeting metadata."""
    required_fields = ["title", "date"]
    missing_fields = [field for field in required_fields if field not in metadata]
    
    if missing_fields:
        raise ValidationError(
            ERROR_MESSAGES["INVALID_METADATA"],
            details={"missing_fields": missing_fields}
        )

def validate_audio_data(audio_data: bytes, min_length: int = 1024) -> None:
    """Validate audio data."""
    if not audio_data or len(audio_data) < min_length:
        raise AudioError(
            ERROR_MESSAGES["INVALID_AUDIO_FORMAT"],
            details={"data_length": len(audio_data) if audio_data else 0}
        )

def validate_transcript(transcript: str) -> None:
    """Validate transcript format."""
    if not transcript or not any(line.strip() for line in transcript.split('\n')):
        raise ValidationError(
            ERROR_MESSAGES["INVALID_TRANSCRIPT"],
            details={"transcript_length": len(transcript) if transcript else 0}
        )
