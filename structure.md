# AI Meeting Assistant Documentation

## 1. Overview
The Meeting Assistant is a comprehensive tool for recording, transcribing, and analyzing meetings. It features integration with Confluence and Jira, voice generation capabilities, and AI-powered analysis.

## 2. Features

### 2.1 Core Features
- Real-time meeting recording with configurable audio devices
- Local speech-to-text using Whisper
- AI-powered meeting analysis and summarization
- Integration with Confluence and Jira
- Voice response generation using F5-TTS
- Project-based organization
- Document and resource management

### 2.2 User Interface
- Project and meeting navigation
- Real-time transcription display
- Task management interface
- Meeting analysis visualization
- Document reference panel

Available Actions:
- Prepare Meeting: Set up meeting context and resources
- Plan Tasks: Generate task breakdowns from discussions
- Summarize: Create meeting summaries
- Create Ticket: Generate Jira tickets
- Define Next Steps: Extract action items
- Identify Questions: List open questions and concerns

## 3. Data Organization

### 3.1 Project Structure
Projects are configured in YAML format with the following structure:
```yaml
projects:
  - name: "Project Name"
    key: "PROJECT-KEY"
    description: "Project Description"
    confluence:
      space: "SPACE-KEY"
      pages:
        - title: "Page Title"
          id: "page-id"
    jira:
      epics:
        - title: "Epic Title"
          id: "PROJ-123"
    web:
      - title: "Resource Title"
        url: "https://example.com"
```

### 3.2 Meeting Data Structure
```
data/
├── projects/
│   ├── metadata.json      # Project meta data with document index (documents and meeting transcripts and analysis)
│   ├── meeting_[id]/
│   │   ├── audio_01.wav   # Recording
│   │   ├── transcript.md  # Transcription
│   │   └── analysis.md    # AI analysis
│   └── documents/
│       ├── confluence/    # Markdown files extracted from Confluence
│       ├── jira/          # Jira tickets as markdown files
│       └── web/           # Scraped webpages stored as markdown files
├── audio_devices.yaml     # Device settings
└── projects.yaml          # Project configuration
```

### 3.3 Metadata Types

#### Meeting Metadata
- Meeting ID and title
- Date and duration
- Participants
- Project association
- Status information
- Related documents and resources

#### Content Metadata
- Content ID and type
- Source information (Confluence, Jira, Web)
- Last update timestamp
- Version tracking
- Processing status

## 4. Configuration

### 4.1 External Integrations
- Confluence: URL, credentials, cloud/server mode
- Jira: URL, credentials, cloud/server mode
- Web: User agent, timeout, retry settings

### 4.2 Audio Settings
- Format: WAV (16kHz, 16-bit, mono)
- Device Configuration:
  - Input devices selection
  - Output recording options
  - Gain control (0-30 dB)
  - Silence detection
  - Chunk handling

### 4.3 Speech Recognition
- Model: Whisper (tiny to large)
- Language support: 18+ languages
- Chunk length: 30 seconds
- Minimum speech duration: 0.3s

### 4.4 Voice Generation
- Engine: F5-TTS
- Sample rate: 24kHz
- Available voices:
  - Nature Voice
  - Morgan Freeman
- Voice customization options

### 4.5 Analysis Settings
- Minimum segment length: 10s
- Summary token limit: 500
- Context window: 4000 tokens
- Temperature: 0.7

### 4.6 System Settings
- Cache duration: 1 hour
- Maximum cache size: 100 items
- Log levels and formats
- File patterns and locations

## 5. Usage Guidelines

### 5.1 Project Setup
1. Configure project details in projects.yaml
2. Set up external integration credentials
3. Configure audio devices
4. Verify connection to Confluence/Jira

### 5.2 Recording Meetings
1. Select project and create meeting
2. Configure audio inputs
3. Start recording
4. Monitor real-time transcription
5. End recording and process

### 5.3 Analysis Workflow
1. Review auto-generated summary
2. Extract action items
3. Generate tasks and tickets
4. Export meeting minutes
5. Follow up on action items

### 5.4 Resource Management
1. Link relevant documents
2. Connect to Confluence pages
3. Associate Jira tickets
4. Track external resources

## 6. Best Practices
- Configure audio devices before meetings
- Use project templates for consistency
- Regular metadata backup
- Proper meeting categorization
- Document linking for context
- Regular cache cleanup

## 7. Future Enhancements
- Multi-speaker detection
- Custom Whisper model training
- Enhanced search capabilities
- Real-time collaboration
- Meeting templates
- Advanced document indexing
- Offline mode support
