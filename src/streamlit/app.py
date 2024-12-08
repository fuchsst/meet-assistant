# src/streamlit/app.py
import streamlit as st
from pathlib import Path
import json
from typing import Dict, List, Optional
from datetime import datetime

from src.agents.crews import CrewFactory

class MeetingAssistantApp:
    """Streamlit interface for the Meeting Assistant."""
    
    def __init__(self):
        """Initialize the Meeting Assistant application."""
        self.setup_page_config()
        
    def setup_page_config(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Meeting Assistant",
            page_icon="👥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def main(self):
        """Main application entry point."""
        st.title("Meeting Assistant")
        
        # Sidebar - Project and Meeting Selection
        self.setup_sidebar()
        
        # Main content area
        if 'selected_meeting' in st.session_state:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                self.setup_main_column()
            
            with col2:
                self.setup_secondary_column()
    
    def setup_sidebar(self):
        """Setup the sidebar for meeting selection."""
        st.sidebar.title("Meeting Navigator")
        
        # Load project metadata
        project_metadata_path = Path("data/project_metadata.json")
        if project_metadata_path.exists():
            with open(project_metadata_path, 'r') as f:
                st.session_state.project_metadata = json.load(f)
        
        # List available meeting directories
        meetings_dir = Path("data/meetings")
        if meetings_dir.exists():
            meeting_dirs = [d for d in meetings_dir.iterdir() if d.is_dir()]
            
            if meeting_dirs:
                meeting_options = {
                    str(d): self._get_meeting_title(d) for d in meeting_dirs
                }
                selected_meeting = st.sidebar.selectbox(
                    "Select Meeting",
                    options=list(meeting_options.keys()),
                    format_func=lambda x: meeting_options[x]
                )
                
                if selected_meeting:
                    st.session_state.selected_meeting = selected_meeting
                    self.load_meeting_data(Path(selected_meeting))
    
    def _get_meeting_title(self, meeting_path: Path) -> str:
        """Get the meeting title from metadata or fallback to directory name."""
        metadata_path = meeting_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get("title", meeting_path.name)
            except:
                pass
        return meeting_path.name
    
    def load_meeting_data(self, meeting_path: Path):
        """Load meeting data and metadata."""
        try:
            # Load metadata
            metadata_path = meeting_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    st.session_state.meeting_metadata = json.load(f)
            
            # Load transcript
            transcript_path = meeting_path / "transcript.md"
            if transcript_path.exists():
                with open(transcript_path, 'r') as f:
                    st.session_state.transcript = f.read()
            
            # Load analysis
            analysis_path = meeting_path / "analysis.md"
            if analysis_path.exists():
                with open(analysis_path, 'r') as f:
                    st.session_state.analysis = f.read()
                    
        except Exception as e:
            st.error(f"Error loading meeting data: {str(e)}")
    
    def setup_main_column(self):
        """Setup the main content column."""
        st.header("Task Interface")
        
        # Task input
        task_description = st.text_area(
            "Task Description",
            placeholder="Describe what you'd like to accomplish...",
            key="task_description"
        )
        
        # Task buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Prepare Meeting"):
                self.execute_task("prepare_meeting", task_description)
            if st.button("📝 Plan Tasks"):
                self.execute_task("plan_tasks", task_description)
        
        with col2:
            if st.button("📊 Summarize"):
                self.execute_task("summarize", task_description)
            if st.button("🎫 Create Ticket"):
                self.execute_task("create_ticket", task_description)
        
        with col3:
            if st.button("➡️ Define Next Steps"):
                self.execute_task("define_next_steps", task_description)
            if st.button("❓ Identify Questions"):
                self.execute_task("identify_questions", task_description)
        
        # Display transcript
        if "transcript" in st.session_state:
            with st.expander("Meeting Transcript", expanded=False):
                st.markdown(st.session_state.transcript)
        
        # Display results
        if "current_result" in st.session_state:
            st.markdown("### Results")
            st.markdown(st.session_state.current_result)
    
    def setup_secondary_column(self):
        """Setup the secondary column with details."""
        st.header("Details")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Summary", "Context", "Analysis"])
        
        with tab1:
            if "current_summary" in st.session_state:
                st.markdown(st.session_state.current_summary)
        
        with tab2:
            if "meeting_metadata" in st.session_state:
                self.render_context(st.session_state.meeting_metadata.get("context", {}))
        
        with tab3:
            if "analysis" in st.session_state:
                st.markdown(st.session_state.analysis)
    
    def render_context(self, context: Dict):
        """Render meeting context information."""
        # Web resources
        if context.get("web"):
            st.markdown("### Web Resources")
            for resource in context["web"]:
                with st.expander(resource["title"]):
                    st.markdown(f"**URL:** {resource.get('url', 'N/A')}")
                    st.markdown(f"**Description:** {resource.get('description', 'N/A')}")
        
        # Confluence pages
        if context.get("confluence"):
            st.markdown("### Confluence Pages")
            for page in context["confluence"]:
                with st.expander(page["title"]):
                    st.markdown(f"**Page ID:** {page.get('page_id', 'N/A')}")
                    st.markdown(f"**Status:** {page.get('status', 'N/A')}")
                    st.markdown(f"**Description:** {page.get('description', 'N/A')}")
        
        # Jira tickets
        if context.get("jira"):
            st.markdown("### Related Tickets")
            for ticket in context["jira"]:
                with st.expander(f"{ticket.get('ticket_id', 'N/A')} - {ticket['title']}"):
                    st.markdown(f"**Type:** {ticket.get('type', 'N/A')}")
                    st.markdown(f"**Status:** {ticket.get('status', 'N/A')}")
                    st.markdown(f"**Description:** {ticket.get('description', 'N/A')}")
    
    def execute_task(self, task_type: str, task_description: str):
        """Execute a task using the appropriate crew."""
        try:
            with st.spinner("Processing..."):
                # Prepare context with metadata
                context = {
                    "project_metadata": st.session_state.get("project_metadata", {}),
                    "meeting_metadata": st.session_state.get("meeting_metadata", {}),
                    "task_type": task_type,
                    "task_description": task_description
                }
                
                # Create appropriate crew
                if task_type == "prepare_meeting":
                    crew = CrewFactory.create_preparation_crew(context)
                elif task_type == "summarize":
                    crew = CrewFactory.create_summary_crew(context)
                elif task_type == "plan_tasks":
                    crew = CrewFactory.create_task_breakdown_crew(context)
                elif task_type == "identify_questions":
                    crew = CrewFactory.create_question_crew(context)
                else:
                    st.error(f"Unknown task type: {task_type}")
                    return
                
                # Execute crew tasks
                result = crew.kickoff()
                
                # Update session state
                st.session_state.current_result = result
                if task_type == "summarize":
                    st.session_state.current_summary = result
                
                # Force streamlit to update
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error executing task: {str(e)}")

if __name__ == "__main__":
    app = MeetingAssistantApp()
    app.main()
