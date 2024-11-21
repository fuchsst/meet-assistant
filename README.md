# Meeting Assistant

An intelligent meeting assistant that records, transcribes, and analyzes meetings in real-time using local Whisper transcription and F5-TTS for voice responses.

## Features

- **Real-time Audio Processing**
  - Live meeting recording with PyAudio
  - Local Whisper-powered transcription
  - Automatic transcript generation with timestamps
  - Audio input/output monitoring

- **AI-Powered Analysis**
  - Meeting summaries
  - Automatic task extraction
  - Key decision tracking
  - Custom analysis queries

- **Interactive Interface**
  - Streamlit-based user interface
  - Real-time transcript display
  - One-click analysis generation
  - Task management system

- **Voice Capabilities**
  - Local F5-TTS integration
  - Configurable voice settings
  - Audio playback controls
  - Voice response for analysis results

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for Whisper)
- Working microphone and speakers
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/meeting-assistant.git
cd meeting-assistant
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download Whisper model (will be done automatically on first run, but you can pre-download):
```bash
# Download medium model (recommended)
python -c "import whisper; whisper.load_model('medium')"
```

## Configuration

The application can be configured through `config/config.py`:

```python
# Audio settings
AUDIO_CONFIG = {
    "format": "wav",
    "channels": 1,
    "rate": 16000,
    "chunk": 1024,
    "sample_width": 2,
}

# Whisper settings
WHISPER_CONFIG = {
    "model_size": "medium",
    "language": "en",
    "task": "transcribe",
}

# Voice settings
VOICE_CONFIG = {
    "speaker": "en_speaker_0",
    "sample_rate": 16000,
    "speed": 1.0,
}
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Create a New Meeting:
   - Click "New Meeting" in the sidebar
   - Fill in meeting details (title, participants)
   - Click "Start Recording" when ready

3. During the Meeting:
   - View real-time transcription
   - Monitor audio levels
   - Track meeting duration

4. Analysis Features:
   - Generate meeting minutes
   - Extract action items
   - Identify key decisions
   - Ask custom analysis questions

5. Export Options:
   - Copy to clipboard
   - Download as markdown
   - Save audio recordings

## Voice Commands

You can interact with the assistant using voice commands:

1. Custom Analysis:
   - Click the microphone icon
   - Ask your question (e.g., "Summarize the key points about the project timeline")
   - The assistant will respond with both text and voice

2. Task Management:
   - Use voice commands to create tasks
   - Ask for task summaries
   - Update task status

## Development

### Project Structure
```
meeting_assistant/
├── app.py              # Main Streamlit application
├── config/            # Configuration files
├── src/
│   ├── core/          # Core functionality
│   │   ├── audio/     # Audio processing
│   │   ├── ai/        # AI components
│   │   └── storage/   # File management
│   └── pages/         # UI components
└── tests/             # Test suites
```

### Running Tests

Run the complete test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_audio.py        # Audio tests
pytest tests/test_transcription.py # Transcription tests
pytest tests/test_analysis.py     # Analysis tests
pytest -m integration            # Integration tests
pytest -m performance           # Performance tests
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Troubleshooting

### Common Issues

1. Audio Recording:
   - Check microphone permissions
   - Verify PyAudio installation
   - Test audio input levels

2. Transcription:
   - Ensure Whisper model is downloaded
   - Check GPU availability
   - Verify audio quality

3. Analysis:
   - Check file permissions
   - Verify file structure
   - Monitor memory usage

### Error Messages

- `PortAudioError`: Check audio device connections
- `FileNotFoundError`: Verify file paths and permissions
- `CUDA out of memory`: Reduce model size or batch size

## Performance Tips

1. GPU Optimization:
   - Use CUDA-capable GPU
   - Monitor GPU memory usage
   - Adjust model size if needed

2. Audio Processing:
   - Optimize chunk size
   - Monitor buffer overflow
   - Handle silence efficiently

3. File Management:
   - Regular cleanup of old files
   - Implement backup strategy
   - Monitor disk usage

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Whisper team
- F5-TTS developers
- Streamlit community
- CrewAI framework

## Support

- Create an issue for bugs
- Join our Discord community
- Check the FAQ in the wiki

## Version History

- 1.0.0
  - Initial release
  - Core functionality implementation
  - Basic UI and analysis features
