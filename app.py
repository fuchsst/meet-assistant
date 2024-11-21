"""Main Streamlit application for Meeting Assistant."""
import logging
from pathlib import Path
import streamlit as st
from datetime import datetime

from config.config import UI_CONFIG, MEETINGS_DIR, VOICE_CONFIG
from src.core.audio.audio_monitor import AudioMonitor
from src.core.storage.file_manager import FileManager
from src.core.ai.analysis_pipeline import AnalysisPipeline
from pages.components.recording_interface import RecordingInterface
from pages.components.analysis_interface import AnalysisInterface
from src.core.utils.logging_config import setup_logging, log_audit
from src.core.utils.error_handling import handle_error, AppError

# Set up logging
logger = logging.getLogger(__name__)
audit_logger = setup_logging()

# Configure Streamlit page
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
)

def initialize_session_state():
    """Initialize session state variables."""
    try:
        if "monitor" not in st.session_state:
            st.session_state.monitor = None
        if "file_manager" not in st.session_state:
            st.session_state.file_manager = FileManager()
        if "analysis_pipeline" not in st.session_state:
            st.session_state.analysis_pipeline = AnalysisPipeline()
        if "recording_interface" not in st.session_state:
            st.session_state.recording_interface = RecordingInterface(
                st.session_state.file_manager
            )
        if "analysis_interface" not in st.session_state:
            st.session_state.analysis_interface = AnalysisInterface(
                st.session_state.analysis_pipeline,
                st.session_state.file_manager
            )
        if "current_meeting_id" not in st.session_state:
            st.session_state.current_meeting_id = None
        if "is_recording" not in st.session_state:
            st.session_state.is_recording = False
        if "transcript" not in st.session_state:
            st.session_state.transcript = ""
        if "error" not in st.session_state:
            st.session_state.error = None
        if "selected_voice" not in st.session_state:
            st.session_state.selected_voice = VOICE_CONFIG["default_voice"]
        
        logger.info("Session state initialized successfully")
        
    except Exception as e:
        error_info = handle_error(e)
        st.error(error_info["message"])
        logger.error("Failed to initialize session state", exc_info=True)

def display_header():
    """Display application header."""
    st.title("Meeting Assistant")
    st.markdown("Record, transcribe, and analyze meetings in real-time.")

    # Display error if any
    if st.session_state.error:
        st.error(st.session_state.error)
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()

def display_sidebar():
    """Display sidebar with meeting list and controls."""
    with st.sidebar:        
        # Voice selection
        voices = VOICE_CONFIG["voices"]
        voice_options = {voice_id: voice["name"] for voice_id, voice in voices.items()}
        selected_voice = st.selectbox(
            "Select Voice",
            options=list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=list(voice_options.keys()).index(st.session_state.selected_voice)
        )
        
        if selected_voice != st.session_state.selected_voice:
            st.session_state.selected_voice = selected_voice
            st.rerun()
            
        # Display voice description
        if selected_voice in voices:
            st.info(voices[selected_voice]["description"])
        
        st.header("Meetings")
        
        # New meeting button
        if st.button("New Meeting", type="primary"):
            try:
                st.session_state.error = None
                meeting_id = st.session_state.file_manager.generate_meeting_id()
                st.session_state.file_manager.create_meeting_directory(meeting_id)
                st.session_state.current_meeting_id = meeting_id
                st.session_state.transcript = ""
                
                log_audit("MEETING_CREATED", {"meeting_id": meeting_id})
                logger.info(f"Created new meeting: {meeting_id}")
                
                st.success("New meeting created!")
                st.rerun()
                
            except Exception as e:
                error_info = handle_error(e)
                st.error(error_info["message"])
                logger.error("Failed to create new meeting", exc_info=True)

        # Meeting details form when a meeting is selected
        if st.session_state.current_meeting_id:
            st.session_state.recording_interface.render_metadata_form(
                st.session_state.current_meeting_id
            )

        # Meeting list
        st.subheader("Recent Meetings")
        try:
            meetings = st.session_state.file_manager.list_meetings()
            if not meetings:
                st.info("No meetings found")
            
            for meeting_id, metadata in meetings.items():
                if st.button(
                    f"{metadata.get('title', 'Untitled')} - {metadata.get('date', 'No date')}",
                    key=f"meeting_{meeting_id}"
                ):
                    try:
                        st.session_state.error = None
                        st.session_state.current_meeting_id = meeting_id
                        
                        log_audit("MEETING_SELECTED", {"meeting_id": meeting_id})
                        logger.info(f"Selected meeting: {meeting_id}")
                        
                        st.rerun()
                        
                    except Exception as e:
                        error_info = handle_error(e)
                        st.error(error_info["message"])
                        logger.error("Failed to select meeting", exc_info=True)
                        
        except Exception as e:
            error_info = handle_error(e)
            st.error(error_info["message"])
            logger.error("Failed to list meetings", exc_info=True)

def main():
    """Main application entry point."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Display components
        display_header()
        display_sidebar()
        
        # Main content
        if st.session_state.current_meeting_id:
            # Create two columns for main layout
            chat_col, record_col = st.columns([1, 1])
            
            # Left column: Chat and analysis interface
            with chat_col:
                # Display analysis interface
                st.session_state.analysis_interface.render_custom_task(
                    st.session_state.current_meeting_id
                )
                if not st.session_state.is_recording:
                    st.session_state.analysis_interface.render_analysis_controls(
                        st.session_state.current_meeting_id
                    )
            
            # Right column: Recording controls and transcript
            with record_col:
                # Display recording controls
                status_updates = st.session_state.recording_interface.render_recording_controls(
                    st.session_state.current_meeting_id,
                    st.session_state.monitor,
                    st.session_state.is_recording
                )
                
                # Update session state with status updates
                st.session_state.monitor = status_updates["monitor"]
                st.session_state.is_recording = status_updates["is_recording"]
                
                # Display transcript
                st.session_state.recording_interface.render_transcript_display(
                    st.session_state.transcript
                )
        else:
            st.info("Select or create a meeting to begin.")

    except Exception as e:
        error_info = handle_error(e)
        st.error(error_info["message"])
        logger.error("Application error", exc_info=True)

if __name__ == "__main__":
    main()
