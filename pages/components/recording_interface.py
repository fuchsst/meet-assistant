"""Recording interface component for Meeting Assistant."""
import logging
from datetime import datetime
from pathlib import Path
import streamlit as st
from typing import Optional, Dict, Any
import time

from src.core.audio.audio_monitor import AudioMonitor
from src.core.storage.file_manager import FileManager
from config.config import WHISPER_CONFIG, AUDIO_CONFIG

logger = logging.getLogger(__name__)

class RecordingInterface:
    """Handles the recording interface and controls."""

    def __init__(self, file_manager: FileManager):
        """Initialize recording interface."""
        self.file_manager = file_manager
        
        # Initialize session state
        if "recording_active" not in st.session_state:
            st.session_state.recording_active = False
        if "monitor" not in st.session_state:
            st.session_state.monitor = None
        if "transcript" not in st.session_state:
            st.session_state.transcript = ""
        if "current_meeting_id" not in st.session_state:
            st.session_state.current_meeting_id = None
        if "stop_requested" not in st.session_state:
            st.session_state.stop_requested = False
        
        # Handle any pending stop request from previous run
        if st.session_state.stop_requested and st.session_state.monitor:
            logger.info("Handling pending stop request from previous run")
            try:
                st.session_state.monitor.stop_monitoring()
                time.sleep(0.5)  # Give time for cleanup logs
                logger.info("Completed pending stop request")
            except Exception as e:
                logger.error(f"Error handling pending stop: {e}")
            finally:
                st.session_state.monitor = None
                st.session_state.recording_active = False
                st.session_state.stop_requested = False
        
        logger.info("RecordingInterface initialized")

    def _handle_recording_change(self) -> None:
        """Handle recording toggle state changes."""
        recording_requested = st.session_state.recording_toggle
        
        if recording_requested and not st.session_state.recording_active:
            # Start recording
            try:
                language_code = st.session_state.get('selected_language')
                gain_db = st.session_state.get('gain_slider', AUDIO_CONFIG["default_gain_db"])
                
                logger.info(f"Starting recording with language={language_code}, gain={gain_db}dB")
                
                if not st.session_state.monitor:
                    logger.debug("Creating new AudioMonitor instance")
                    new_monitor = AudioMonitor(language=language_code)
                    new_monitor.set_gain(gain_db)
                    st.session_state.monitor = new_monitor
                
                def transcription_callback(text: str) -> None:
                    """Wrapper to add logging around transcription callback"""
                    logger.debug(f"Transcription callback received text: {text}")
                    self._handle_transcription(st.session_state.current_meeting_id, text)
                
                logger.debug("Starting audio monitoring with transcription callback")
                st.session_state.monitor.start_monitoring(transcription_callback)
                
                st.session_state.recording_active = True
                logger.info("Recording started successfully")
                st.success("Recording started!")
                
            except Exception as e:
                logger.error(f"Failed to start recording: {str(e)}", exc_info=True)
                st.error(f"Failed to start recording: {str(e)}")
                st.session_state.recording_active = False
                st.session_state.recording_toggle = False
                if st.session_state.monitor:
                    try:
                        logger.debug("Attempting to stop monitoring after error")
                        st.session_state.monitor.stop_monitoring()
                        time.sleep(0.5)  # Give time for cleanup logs
                    except Exception as stop_error:
                        logger.error(f"Error stopping monitor: {stop_error}", exc_info=True)
                st.session_state.monitor = None
                
        elif not recording_requested and st.session_state.recording_active:
            # Set stop request flag and rerun to handle in next iteration
            logger.info("Stop requested, setting flag for next run")
            st.session_state.stop_requested = True
            st.session_state.recording_toggle = False
            st.rerun()

    def render_metadata_form(self, meeting_id: str) -> None:
        """Render and handle meeting metadata form."""
        # Store meeting ID in session state
        st.session_state.current_meeting_id = meeting_id
        logger.debug(f"Set current meeting ID to: {meeting_id}")
        
        # Get existing metadata
        metadata = self.file_manager.get_metadata(meeting_id) or {}

        # Create form
        with st.form(key="meeting_metadata"):
            # Title
            title = st.text_input(
                "Meeting Title",
                value=metadata.get("title", ""),
                placeholder="Enter meeting title"
            )

            # Participants
            participants = st.text_input(
                "Participants",
                value=", ".join(metadata.get("participants", [])),
                placeholder="Enter participant names (comma-separated)"
            )

            # Description
            description = st.text_area(
                "Description",
                value=metadata.get("description", ""),
                placeholder="Enter meeting description"
            )

            # Submit button
            if st.form_submit_button("Update Details"):
                try:
                    # Update metadata
                    updated_metadata = {
                        "title": title,
                        "participants": [p.strip() for p in participants.split(",") if p.strip()],
                        "description": description,
                        "last_modified": datetime.now().isoformat()
                    }
                    
                    # Preserve existing metadata fields
                    updated_metadata.update({
                        k: v for k, v in metadata.items()
                        if k not in updated_metadata
                    })
                    
                    logger.debug(f"Saving updated metadata for meeting {meeting_id}")
                    self.file_manager.save_metadata(meeting_id, updated_metadata)
                    st.success("Meeting details updated successfully!")
                
                except Exception as e:
                    logger.error(f"Failed to update metadata: {str(e)}", exc_info=True)
                    st.error(f"Failed to update details: {str(e)}")

    def render_recording_controls(
        self,
        meeting_id: str,
        monitor: Optional[AudioMonitor] = None,
        is_recording: bool = False
    ) -> Dict[str, Any]:
        """Render recording control buttons and status."""
        st.caption("Recording Controls")

        # Update session state with current monitor
        if monitor and monitor != st.session_state.monitor:
            logger.debug("Updating session state with new monitor instance")
            st.session_state.monitor = monitor
            
        # Update session state with current recording status
        if is_recording != st.session_state.recording_active:
            logger.debug(f"Updating recording status to: {is_recording}")
            st.session_state.recording_active = is_recording

        # Create columns for controls
        lang_col, gain_col, toggle_col = st.columns([2, 2, 2])

        # Language selection (only shown when not recording)
        with lang_col:
            if not st.session_state.recording_active:
                # Create a list of language options with display names
                language_options = list(WHISPER_CONFIG["available_languages"].items())
                selected_lang_idx = next(
                    (i for i, (code, _) in enumerate(language_options) 
                     if code == WHISPER_CONFIG["language"]),
                    0
                )
                
                selected_lang = st.selectbox(
                    "Audio Language",
                    range(len(language_options)),
                    format_func=lambda i: language_options[i][1],
                    index=selected_lang_idx,
                    help="Select the language of the audio that will be recorded",
                    label_visibility="collapsed",
                    key="language_select"
                )
                st.session_state.selected_language = language_options[selected_lang][0]
                logger.debug(f"Selected language: {st.session_state.selected_language}")
            else:
                st.empty()

        # Gain control
        with gain_col:
            gain_db = st.slider(
                "Input Gain",
                min_value=float(AUDIO_CONFIG["min_gain_db"]),
                max_value=float(AUDIO_CONFIG["max_gain_db"]),
                value=float(st.session_state.monitor.gain_db if st.session_state.monitor else AUDIO_CONFIG["default_gain_db"]),
                step=1.0,
                help="Adjust microphone sensitivity",
                key="gain_slider"
            )
            if st.session_state.monitor and gain_db != st.session_state.monitor.gain_db:
                logger.debug(f"Updating gain to {gain_db}dB")
                st.session_state.monitor.set_gain(gain_db)

        # Recording toggle
        with toggle_col:
            st.toggle(
                "Recording",
                value=st.session_state.recording_active,
                help="Toggle recording on/off",
                key="recording_toggle",
                on_change=self._handle_recording_change
            )

        # Display recording status
        st.caption(f"Status: {'🔴 Recording in progress...' if st.session_state.recording_active else '⚪ Ready to record'}")

        return {
            "monitor": st.session_state.monitor,
            "is_recording": st.session_state.recording_active
        }

    def render_transcript_display(self, transcript: str) -> None:
        """Render real-time transcript display."""
        # Display transcript
        st.text_area(
            label="Real-time transcript",
            value=transcript,
            height=400,
            disabled=True,
            key="transcript_display"
        )

        # Word count and duration
        if transcript:
            word_count = len(transcript.split())
            duration = self._estimate_duration(transcript)
            
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Words", word_count)
            with stats_col2:
                st.metric("Duration", f"{duration} seconds")

    def _handle_transcription(self, meeting_id: str, text: str) -> None:
        """Handle incoming transcription text."""
        if not text:
            logger.debug("Received empty transcription text, skipping")
            return
            
        try:
            # Save to file
            transcript_path = Path(f"data/meetings/{meeting_id}/transcript.txt")
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Writing transcription to {transcript_path}")
            with open(transcript_path, 'a', encoding='utf-8') as f:
                f.write(f"{text}\n")
            
            logger.info(f"Saved transcription to {transcript_path}")
            
            # Update session state
            if "transcript" not in st.session_state:
                st.session_state.transcript = ""
            st.session_state.transcript += f"{text}\n"
            logger.debug("Updated transcript in session state")
            
            # Force streamlit to update
            st.rerun()
        
        except Exception as e:
            logger.error(f"Failed to handle transcription: {str(e)}", exc_info=True)
            st.error(f"Failed to save transcript: {str(e)}")

    def _estimate_duration(self, transcript: str) -> int:
        """Estimate recording duration from transcript timestamps."""
        try:
            import re
            timestamps = re.findall(r'\[(\d{2}:\d{2}:\d{2})\]', transcript)
            if not timestamps:
                return 0
            
            # Convert last timestamp to seconds
            last_time = timestamps[-1]
            h, m, s = map(int, last_time.split(':'))
            duration = h * 3600 + m * 60 + s
            logger.debug(f"Estimated duration: {duration} seconds")
            return duration
            
        except Exception as e:
            logger.warning(f"Failed to estimate duration: {str(e)}")
            return 0
