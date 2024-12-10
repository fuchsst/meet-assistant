# Meeting Assistant

An intelligent meeting assistant that records, transcribes, and analyzes meetings in real-time using local Whisper transcription and F5-TTS for voice responses. Features integration with Confluence and Jira for comprehensive document and task management.

## Features

- **Real-time Audio Processing**
  - Multi-source audio recording (microphone and system audio)
  - Local Whisper-powered transcription
  - Audio level monitoring and gain control
  - Configurable input/output device selection

- **AI-Powered Analysis**
  - Meeting summaries and key points extraction
  - Action item identification
  - Task and timeline generation
  - Integration with project documentation

- **External Integrations**
  - Confluence integration for documentation
  - Jira integration for task management
  - Web resource fetching
  - Project context awareness

- **Voice Capabilities**
  - Local F5-TTS integration
  - Multiple voice options
  - Voice response for analysis results

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for Whisper)
- Working microphone and speakers
- Git (for cloning the repository)
- FFmpeg (for audio recording)
- Confluence and Jira access (optional)

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

4. Install FFmpeg:
   - Windows: Download from [FFmpeg official website](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

5. Download Whisper model:
```bash
# Download medium model (recommended)
python -c "import whisper; whisper.load_model('medium')"
```

## Configuration

### 1. Audio Device Setup

Run the audio configuration tool to select input/output devices:

```bash
python config_audio.py
```

Features:
- Real-time audio level monitoring
- Input device selection
- Output device configuration
- System audio capture setup

### 2. External Services

Configure external services in environment variables:

```bash
# Confluence settings
CONFLUENCE_URL=https://your-instance.atlassian.net
CONFLUENCE_USERNAME=your-email
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_CLOUD=true

# Jira settings
JIRA_URL=https://your-instance.atlassian.net
JIRA_USERNAME=your-email
JIRA_API_TOKEN=your-api-token
JIRA_CLOUD=true
```

### 3. Project Configuration

Projects are configured in `data/projects.yaml`. This file defines project metadata and integration settings for Confluence, Jira, and web resources.

#### Basic Structure

```yaml
projects:
  - name: "Project Name"          # Display name of the project
    key: "PROJECT-KEY"           # Unique identifier (uppercase with hyphens)
    description: "Description"    # Brief project description
    members:                     # Optional: Team member details
      - name: "John Doe"
        role: "Project Manager"
        user_names:             # Identifiers across different systems
          - "johndoe"           # Username
          - "john@example.com"  # Email
          - "JD123"            # Employee ID
```

#### Integration Configuration

##### Confluence Integration
```yaml
    confluence:
      space: "SPACE-KEY"         # Confluence space key
      pages:                     # Pages to track
        - title: "Requirements"  # Page title for reference
          id: "123456"          # Confluence page ID
        - title: "Architecture"
          id: "123457"
```

##### Jira Integration
```yaml
    jira:
      epics:                     # Epics to track
        - title: "Phase 1"       # Epic title for reference
          id: "PROJ-123"        # Jira epic key
        - title: "Phase 2"
          id: "PROJ-456"
```

##### Web Resources
```yaml
    web:                         # External web resources
      - title: "API Docs"        # Resource title
        url: "https://api.example.com/docs"
      - title: "Wiki"
        url: "https://wiki.example.com"
```

#### Complete Example

Here's a complete example of a project configuration:

```yaml
projects:
  - name: "Customer Portal"
    key: "PORTAL"
    description: "Customer self-service portal development"
    members:
      - name: "Jane Smith"
        role: "Tech Lead"
        user_names:
          - "jsmith"
          - "jane.smith@company.com"
          - "JS789"
      - name: "Mike Johnson"
        role: "Developer"
        user_names:
          - "mjohnson"
          - "mike.j@company.com"
          - "MJ456"
    
    confluence:
      space: "PORTAL"
      pages:
        - title: "Requirements Specification"
          id: "987654"
        - title: "Technical Design"
          id: "987655"
        - title: "API Documentation"
          id: "987656"
    
    jira:
      epics:
        - title: "User Authentication"
          id: "PORTAL-100"
        - title: "Dashboard Development"
          id: "PORTAL-101"
        - title: "Payment Integration"
          id: "PORTAL-102"
    
    web:
      - title: "Design System"
        url: "https://design.company.com"
      - title: "API Reference"
        url: "https://api.company.com/docs"
      - title: "Style Guide"
        url: "https://style.company.com"
```

#### Configuration Guidelines

1. **Project Keys**
   - Use uppercase letters, numbers, and hyphens
   - Start with a letter
   - Keep it short but meaningful
   - Examples: `PORTAL`, `CRM-API`, `HR-SYSTEM`

2. **Member Configuration**
   - Include all relevant usernames/emails
   - Use consistent role names
   - Add system-specific IDs if available

3. **Confluence Integration**
   - Use space keys from your Confluence instance
   - Get page IDs from page URLs or page info
   - Include key documentation pages only

4. **Jira Integration**
   - Use epic keys from your Jira instance
   - Include epics that are actively being worked on
   - Update as new epics are created

5. **Web Resources**
   - Include only stable, long-term resources
   - Use descriptive titles
   - Ensure URLs are accessible to team members

#### Schema Validation

A JSON schema for validating the configuration is available at `config/projects_schema.json`. You can use tools like `yamllint` or `jsonschema` to validate your configuration:

```bash
# Install validation tool
pip install yamllint jsonschema

# Validate configuration
python -c "import yaml, jsonschema; schema = json.load(open('config/projects_schema.json')); config = yaml.safe_load(open('data/projects.yaml')); jsonschema.validate(config, schema)"
```

## Usage

### 1. Recording Meetings

Use the recording CLI tool:

```bash
python recorder_cli.py --title "Meeting Title" [--project-id PROJECT_ID]

Arguments:
  --title TEXT        Recording session title (required)
  --project-id TEXT   Optional project ID (uses default project if not provided)
```

Features:
- Automatically detects available audio input and output devices
- Records both microphone input and system audio output
- Saves recordings as WAV files in the project's meeting directory
- Supports multiple recording sessions within the same run

Controls:
- Press Enter to start a new recording
- Press Ctrl+C to exit the program

Output:
- Recordings are saved in `data/PROJECT_ID/meetings/` if a project ID is provided
- If no project ID is provided, recordings are saved in `data/YYYYMMDD_default/meetings/`
- Filenames follow the format: `YYYYMMDD_meeting_title_001.wav`

### 2. Transcribing Recordings

Use the transcription CLI tool:

```bash
python transcribe_cli.py [options] INPUT_PATH [OUTPUT_PATH]

Options:
  --language TEXT    Language code (default: "en")
  INPUT_PATH        Path to WAV file or directory
  OUTPUT_PATH       Optional output path for VTT files
```

Features:
- Supports single file or batch processing
- Multiple language support
- Automatic timestamp generation
- Progress tracking

### 3. Processing Transcripts

Process transcripts with AI analysis:

```bash
python process_transcript.py [options] MEETING_DIR

Options:
  --project-config TEXT   Path to project config YAML
  --context-window INT    Context window size (default: 5)
```

Features:
- Combines multiple VTT files
- Generates meeting summaries
- Extracts action items
- Creates markdown reports
- Integrates project context

### 4. Main Application

Launch the Streamlit interface:

```bash
streamlit run src/streamlit/app.py
```

Features:
- Project and meeting management
- Real-time transcription display
- Meeting analysis and summaries
- Task and ticket creation
- Document integration

## Common Workflows

### 1. Quick Meeting Recording

1. Configure audio devices (automatic)

2. Start recording:
   ```bash
   python recorder_cli.py --title "Quick Meeting"
   ```

3. Press Enter to start/stop recordings as needed

4. Press Ctrl+C to exit when done

5. Process the recording:
   ```bash
   python transcribe_cli.py data/YYYYMMDD_default/meetings/
   python process_transcript.py data/YYYYMMDD_default/meetings/
   ```

### 2. Project Meeting

1. Set up project configuration in `data/projects.yaml`

2. Start recording with project ID:
   ```bash
   python recorder_cli.py --title "Project Meeting" --project-id YOUR_PROJECT_ID
   ```

3. Press Enter to start/stop recordings as needed

4. Press Ctrl+C to exit when done

5. Process the recording:
   ```bash
   python transcribe_cli.py data/YOUR_PROJECT_ID/meetings/
   python process_transcript.py data/YOUR_PROJECT_ID/meetings/
   ```

## Troubleshooting

### Audio Issues

1. No audio devices detected:
   - Ensure FFmpeg is installed and in the system PATH
   - Check system permissions for audio access
   - Verify PyAudio installation

2. Poor recording quality:
   - Check microphone and speaker settings in your operating system
   - Ensure both input and output devices are correctly selected
   - Try different audio input/output devices if available

### Transcription Issues

1. GPU memory errors:
   - Reduce Whisper model size
   - Free up GPU memory
   - Switch to CPU processing

2. Language detection issues:
   - Specify language explicitly
   - Check audio quality
   - Verify Whisper model size

## Support

- Create an issue for bugs
- Check documentation in docs/
- Review logs in logs/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
