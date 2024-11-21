# Meeting Assistant User Guide

## Getting Started

### First Launch
1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. The interface will open in your default browser
3. You'll see the main dashboard with:
   - Meeting controls in the center
   - Meeting list in the sidebar
   - Analysis options below

### Creating a New Meeting
1. Click "New Meeting" in the sidebar
2. Fill in meeting details:
   - Title (e.g., "Weekly Team Sync")
   - Participants (comma-separated names)
   - Description (optional)
3. Click "Update Details" to save

## Recording Meetings

### Starting a Recording
1. Ensure your microphone is connected and working
2. Click "Start Recording" button
3. You'll see:
   - 🔴 Recording indicator
   - Real-time transcript appearing
   - Audio level indicators

### During Recording
- Monitor the live transcript
- Watch for audio quality indicators
- Track meeting duration
- Add participants if needed

### Stopping a Recording
1. Click "Stop Recording" button
2. The recording will be saved automatically
3. Analysis options will become available

## Analysis Features

### Generating Meeting Minutes
1. After recording, click "Generate Minutes"
2. The system will process:
   - Meeting summary
   - Key points
   - Timeline
3. Export options:
   - Copy to clipboard
   - Download as markdown

Example minutes output:
```markdown
# Meeting Minutes: Weekly Team Sync
Date: 2024-03-19
Duration: 45 minutes

## Summary
Weekly team sync discussing Q1 objectives and project timeline.

## Key Points
- Reviewed project milestones
- Discussed resource allocation
- Set deadlines for deliverables

## Action Items
1. Update project timeline (Bob)
2. Schedule follow-up meetings (Alice)
3. Prepare budget report (Charlie)
```

### Extracting Action Items
1. Click "Extract Tasks" button
2. Review generated tasks
3. Each task includes:
   - Description
   - Assignee
   - Due date
   - Context

### Identifying Decisions
1. Click "Identify Decisions"
2. Review key decisions
3. Export for documentation

### Custom Analysis
1. Use the custom task input:
   - Text input: Type your question
   - Voice input: Click microphone icon
2. Example queries:
   - "Summarize the discussion about budget"
   - "What were the main concerns raised?"
   - "List all deadlines mentioned"

## Voice Features

### Voice Input
1. Click the microphone icon
2. Speak your query clearly
3. Wait for processing
4. Review both text and voice response

### Voice Configuration
1. Access voice settings
2. Options:
   - Speaker selection
   - Speed adjustment
   - Language selection

### Voice Playback
- Click play icon for voice responses
- Adjust volume as needed
- Use pause/resume controls

## Tips and Best Practices

### Audio Quality
1. Environment:
   - Quiet room
   - Minimal background noise
   - Good microphone placement

2. Speaking:
   - Clear enunciation
   - Normal pace
   - Avoid overlapping speech

### Effective Meetings
1. Before meeting:
   - Test audio
   - Clear meeting agenda
   - Invite participants

2. During meeting:
   - Introduce speakers
   - Regular pauses
   - Summarize key points

3. After meeting:
   - Review transcript
   - Verify action items
   - Distribute minutes

### File Management
1. Regular cleanup:
   - Archive old meetings
   - Remove unnecessary files
   - Maintain backups

2. Organization:
   - Use descriptive titles
   - Tag important meetings
   - Categorize by project

## Troubleshooting

### Audio Issues
1. No recording:
   - Check microphone connection
   - Verify permissions
   - Test system audio settings

2. Poor quality:
   - Check input levels
   - Reduce background noise
   - Update audio drivers

### Transcription Issues
1. Inaccurate transcription:
   - Speak clearly
   - Reduce background noise
   - Check audio quality

2. Missing text:
   - Check internet connection
   - Verify system resources
   - Monitor CPU usage

### Analysis Issues
1. Slow processing:
   - Check internet connection
   - Monitor system resources
   - Reduce meeting length

2. Incorrect analysis:
   - Verify transcript quality
   - Provide clear context
   - Use specific queries

## Keyboard Shortcuts

- `Ctrl + N`: New meeting
- `Ctrl + R`: Start/Stop recording
- `Ctrl + M`: Generate minutes
- `Ctrl + T`: Extract tasks
- `Ctrl + Space`: Voice input
- `Ctrl + S`: Save current view

## Data Management

### Exporting Data
1. Meeting data:
   - Transcripts (TXT)
   - Minutes (MD)
   - Action items (JSON)

2. Analysis results:
   - Summaries
   - Task lists
   - Decision logs

### Backup
1. Automatic backups:
   - After each meeting
   - Daily snapshots
   - Weekly archives

2. Manual backups:
   - Export important meetings
   - Save analysis results
   - Archive project data

## Support

### Getting Help
1. Documentation:
   - README.md
   - API documentation
   - This user guide

2. Issue reporting:
   - GitHub issues
   - Bug reports
   - Feature requests

3. Community:
   - Discord channel
   - Wiki pages
   - FAQ section
