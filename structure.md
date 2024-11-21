# AI Meeting Assistant Project Specification

## 1. Project Overview
The Meeting Assistant is a Streamlit-based application that records meetings, transcribes them using local Whisper, stores data in JSON files, and leverages AI for analysis, with F5-TTS for voice responses.

## 2. Core Components

### 2.1 Audio Recording & Processing
- **Audio Capture Module**
  - Real-time audio stream capture using PyAudio
  - Buffer management for continuous streaming
  - Audio format: WAV (16kHz, 16-bit, mono)

- **Local Speech-to-Text Pipeline**
  - Local Whisper model implementation
  - Model: `whisper-medium` or `whisper-large-v3`
  - Streaming buffer management
  - Timestamp preservation

### 2.2 Data Management
- **Storage Structure**
  ```
  data/
  ├── meetings/
  │   ├── metadata.json       # Global meetings index
  │   ├── meeting_[id]/
  │   │   ├── audio.wav      # Raw audio recording
  │   │   ├── transcript.txt  # Plain text transcript
  │   │   └── analysis.json  # AI-generated insights and tasks
  ```

- **Metadata JSON Structure**
  ```json
  {
    "meetings": {
      "meeting_20240319_001": {
        "date": "2024-03-19",
        "start_time": "14:30:00",
        "duration": "3600",
        "participants": ["Alice", "Bob"],
        "title": "Weekly Planning",
        "status": "completed"
      }
    }
  }
  ```

- **Analysis JSON Structure**
  ```json
  {
    "summary": {
      "key_points": [
        "Discussed Q1 objectives",
        "Reviewed project timeline"
      ],
      "decisions": [
        "Approved budget increase",
        "Scheduled follow-up meeting"
      ]
    },
    "action_items": [
      {
        "task": "Prepare budget report",
        "assignee": "Alice",
        "due_date": "2024-03-26",
        "status": "pending",
        "context": "Reference transcript timestamp 00:15:30"
      }
    ],
    "topics": [
      {
        "name": "Budget Discussion",
        "timestamp_start": "00:10:00",
        "timestamp_end": "00:25:30"
      }
    ]
  }
  ```

### 2.3 File Operations
```python
class MeetingStorage:
    def __init__(self, base_path="data/meetings"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def create_meeting(self, meeting_id):
        """Create directory structure for new meeting"""
        meeting_dir = self.base_path / f"meeting_{meeting_id}"
        meeting_dir.mkdir(exist_ok=True)
        return meeting_dir
        
    def save_transcript(self, meeting_id, text, mode="a"):
        """Append or write to transcript file"""
        transcript_path = self.base_path / f"meeting_{meeting_id}/transcript.txt"
        with open(transcript_path, mode, encoding='utf-8') as f:
            f.write(f"{text}\n")
            
    def read_transcript(self, meeting_id):
        """Read full transcript"""
        transcript_path = self.base_path / f"meeting_{meeting_id}/transcript.txt"
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def update_metadata(self, meeting_id, metadata):
        """Update meeting metadata"""
        metadata_path = self.base_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"meetings": {}}
            
        data["meetings"][meeting_id] = metadata
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def save_analysis(self, meeting_id, analysis):
        """Save AI analysis results"""
        analysis_path = self.base_path / f"meeting_{meeting_id}/analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
```

### 2.4 Real-time Processing Pipeline
```python
class MeetingProcessor:
    def __init__(self):
        self.storage = MeetingStorage()
        self.whisper = WhisperProcessor()
        self.llm = LLMClient()
        
    async def process_audio_chunk(self, meeting_id, audio_chunk):
        # Process audio with Whisper
        text = await self.whisper.transcribe_chunk(audio_chunk)
        
        # Save transcript chunk
        if text.strip():
            self.storage.save_transcript(
                meeting_id,
                f"[{datetime.now().strftime('%H:%M:%S')}] {text}",
                mode="a"
            )
            
        return text
        
    async def generate_analysis(self, meeting_id):
        # Read full transcript
        transcript = self.storage.read_transcript(meeting_id)
        
        # Generate analysis with LLM
        analysis = await self.llm.analyze_meeting(transcript)
        
        # Save analysis results
        self.storage.save_analysis(meeting_id, analysis)
        
        return analysis
```

### 2.5 Streamlit Interface
```python
def render_meeting_page():
    st.title("AI Meeting Assistant")
    
    # Meeting controls
    if st.button("Start New Meeting"):
        meeting_id = generate_meeting_id()
        st.session_state.meeting_id = meeting_id
        st.session_state.processor = MeetingProcessor()
        
    # Real-time transcript display
    if 'transcript' in st.session_state:
        st.text_area("Live Transcript", 
                    value=st.session_state.transcript,
                    height=300)
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate Minutes"):
            transcript = storage.read_transcript(st.session_state.meeting_id)
            minutes = generate_minutes(transcript)
            st.write(minutes)
            
    with col2:
        if st.button("Extract Tasks"):
            analysis = storage.read_analysis(st.session_state.meeting_id)
            st.write(analysis["action_items"])
            
    with col3:
        if st.button("Summary"):
            analysis = storage.read_analysis(st.session_state.meeting_id)
            st.write(analysis["summary"])
```

## 3. Technical Stack
- **Core Technologies**
  - Python 3.9+
  - Streamlit
  - PyAudio
  - Local Whisper
  - CrewAI
  - F5-TTS (local)

- **Dependencies**
  ```
  streamlit>=1.40.1
  pyaudio>=0.2.14
  openai-whisper>=20240930
  crewai>=0.80.0
  f5-tts>=0.1.0
  python-json-logger>=2.0.7
  fire>=0.7.0
  ```

## 4. Implementation Phases

### Phase 1: Core Recording
1. Audio capture and file management
2. Real-time Whisper transcription
3. Basic text file storage
4. Simple metadata tracking

### Phase 2: Analysis
1. LLM integration
2. Transcript analysis
3. Task extraction
4. JSON storage for analysis results

### Phase 3: UI Development
1. Real-time transcript display
2. Meeting controls
3. Analysis viewing
4. Task management

### Phase 4: Enhancement
1. Export functionality
2. Meeting templates
3. Voice responses
4. Search capabilities

## 5. File Structure
```
meeting_assistant/
├── app/               # Streamlit application
├── core/             # Core functionality
│   ├── audio/        # Audio recording and processing
│   ├── transcription/# Whisper integration
│   └── storage/      # File management
├── ai/               # AI components
│   ├── llm_client.py
│   ├── crew/
│   └── tts/
└── data/             # Meeting data storage
    └── meetings/
```

## 6. Security & Performance
- Local file encryption for sensitive data
- Environment variables management
- Caching for frequent LLM queries
- Audio compression for storage efficiency
- Regular JSON file backups

## 7. Future Enhancements
- Offline mode with cached LLM responses
- Custom Whisper model fine-tuning
- Meeting templates and presets
- Batch processing for long recordings
- Multi-speaker detection
