# Meeting Assistant API Documentation

## Core Components

### Audio Processing

#### AudioRecorder
```python
class AudioRecorder:
    """Handles audio recording functionality using PyAudio."""
    
    def start_recording(self) -> None:
        """Start recording audio."""
        
    def stop_recording(self) -> None:
        """Stop recording audio."""
        
    def save_recording(self, filepath: Path) -> None:
        """Save the recorded audio to a WAV file."""
        
    def validate_audio(self) -> bool:
        """Validate the recorded audio data."""
```

#### AudioMonitor
```python
class AudioMonitor:
    """Monitors and transcribes audio input and output streams."""
    
    def start_monitoring(self, transcription_callback: Optional[Callable] = None) -> None:
        """Start monitoring audio streams."""
        
    def stop_monitoring(self) -> None:
        """Stop monitoring audio streams."""
        
    def _is_silence(self, audio_data: bytes) -> bool:
        """Detect if an audio chunk is silence."""
```

#### VoiceGenerator
```python
class VoiceGenerator:
    """Handles text-to-speech generation and playback."""
    
    def generate_speech(self, text: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Generate speech from text."""
        
    def play_audio(self, audio_path: Optional[Path] = None) -> None:
        """Play generated audio."""
        
    def generate_and_play(self, text: str) -> None:
        """Generate speech and play it immediately."""
```

### Transcription

#### WhisperProcessor
```python
class WhisperProcessor:
    """Handles real-time transcription using Whisper."""
    
    def transcribe_chunk(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe an audio chunk in real-time."""
        
    def transcribe_stream(self, audio_stream: Generator[bytes, None, None]) -> Generator[str, None, None]:
        """Transcribe an audio stream in real-time."""
        
    def transcribe_file(self, audio_path: Path) -> list[dict]:
        """Transcribe an entire audio file with detailed segments."""
```

### File Management

#### FileManager
```python
class FileManager:
    """Handles file operations and directory management."""
    
    def generate_meeting_id(self) -> str:
        """Generate a unique meeting ID."""
        
    def create_meeting_directory(self, meeting_id: str) -> Path:
        """Create directory structure for a new meeting."""
        
    def save_metadata(self, meeting_id: str, metadata: Dict[str, Any]) -> None:
        """Save or update meeting metadata."""
        
    def get_metadata(self, meeting_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a specific meeting or all meetings."""
        
    def create_backup(self, meeting_id: str) -> Path:
        """Create a backup of meeting data."""
```

### AI Analysis

#### LLMClient
```python
class LLMClient:
    """Handles LLM-based text analysis using CrewAI."""
    
    def analyze_transcript(self, transcript: str, max_retries: int = 3) -> Dict[str, Any]:
        """Analyze meeting transcript using multiple agents."""
        
    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate the structure and content of the analysis."""
```

#### AnalysisPipeline
```python
class AnalysisPipeline:
    """Orchestrates the meeting analysis process."""
    
    def process_meeting(self, meeting_id: str) -> Dict[str, Any]:
        """Process a complete meeting recording."""
        
    def process_segment(self, meeting_id: str, audio_segment: bytes) -> Dict[str, Any]:
        """Process a single audio segment in real-time."""
        
    def generate_summary(self, meeting_id: str) -> Dict[str, Any]:
        """Generate a meeting summary."""
```

### UI Components

#### RecordingInterface
```python
class RecordingInterface:
    """Handles the recording interface and controls."""
    
    def render_metadata_form(self, meeting_id: str) -> None:
        """Render and handle meeting metadata form."""
        
    def render_recording_controls(self, meeting_id: str, monitor: Optional[AudioMonitor] = None, is_recording: bool = False) -> Dict[str, Any]:
        """Render recording control buttons and status."""
        
    def render_transcript_display(self, transcript: str) -> None:
        """Render real-time transcript display."""
```

#### AnalysisInterface
```python
class AnalysisInterface:
    """Handles the analysis interface and controls."""
    
    def render_analysis_controls(self, meeting_id: str) -> None:
        """Render analysis control buttons and results."""
        
    def render_custom_task(self, meeting_id: str) -> None:
        """Render custom task input and execution."""
```

## Configuration

### Audio Configuration
```python
AUDIO_CONFIG = {
    "format": "wav",      # Audio format
    "channels": 1,        # Mono audio
    "rate": 16000,        # Sample rate in Hz
    "chunk": 1024,        # Buffer size
    "sample_width": 2     # 16-bit audio
}
```

### Whisper Configuration
```python
WHISPER_CONFIG = {
    "model_size": "medium",  # Whisper model size
    "language": "en",        # Default language
    "task": "transcribe"     # Transcription task
}
```

### Voice Configuration
```python
VOICE_CONFIG = {
    "speaker": "en_speaker_0",  # Default voice
    "sample_rate": 16000,       # Sample rate in Hz
    "speed": 1.0               # Speech speed
}
```

## Error Handling

All components implement comprehensive error handling:

1. Audio Errors:
   - `AudioDeviceError`: Audio device not available
   - `RecordingError`: Recording operation failed
   - `PlaybackError`: Audio playback failed

2. File Errors:
   - `FileNotFoundError`: Required file not found
   - `PermissionError`: File access denied
   - `BackupError`: Backup operation failed

3. Analysis Errors:
   - `TranscriptionError`: Transcription failed
   - `AnalysisError`: Analysis operation failed
   - `ValidationError`: Analysis validation failed

## Best Practices

1. Resource Management:
   - Always close audio streams
   - Release file handles promptly
   - Clean up temporary files

2. Error Handling:
   - Use try-except blocks
   - Provide meaningful error messages
   - Implement proper cleanup

3. Performance:
   - Optimize buffer sizes
   - Use appropriate model sizes
   - Implement caching where appropriate

## Examples

### Recording Example
```python
recorder = AudioRecorder()
recorder.start_recording()
# Record for some time
recorder.stop_recording()
recorder.save_recording(Path("meeting.wav"))
```

### Analysis Example
```python
pipeline = AnalysisPipeline()
analysis = pipeline.process_meeting("meeting_id")
summary = pipeline.generate_summary("meeting_id")
```

### Voice Generation Example
```python
generator = VoiceGenerator()
generator.generate_and_play("Meeting summary complete.")
