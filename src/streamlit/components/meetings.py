import streamlit as st
from datetime import datetime

def render_meeting_list(metadata_manager, project_id: str):
    """Render the list of meetings for a project.
    
    Args:
        metadata_manager: UnifiedMetadataManager instance
        project_id: Current project ID
    """
    meetings = metadata_manager.get_meeting_metadata(project_id)
    
    # Sort meetings by date (newest first)
    sorted_meetings = sorted(
        meetings.items(),
        key=lambda x: x[1].get('created_at', ''),
        reverse=True
    )
    
    # Add new meeting button for admin users
    if metadata_manager.is_admin:
        if st.button("New Meeting"):
            st.session_state.selected_meeting = None
            st.session_state.new_meeting = True
            return
    
    # Render meeting list
    for meeting_id, meeting in sorted_meetings:
        # Skip deleted meetings unless showing all
        if meeting.get('status') == 'deleted' and not st.session_state.get('show_deleted', False):
            continue
            
        # Create meeting card
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                title = meeting.get('title', 'Untitled Meeting')
                date = datetime.fromisoformat(str(meeting['created_at'])).strftime('%Y-%m-%d %H:%M')
                if st.button(f"{title} - {date}", key=f"meeting_{meeting_id}"):
                    st.session_state.selected_meeting = meeting_id
                    st.session_state.new_meeting = False
            
            # Show status indicator
            with col2:
                status = meeting.get('status', 'in_progress')
                if status == 'completed':
                    st.success('Completed')
                elif status == 'deleted':
                    st.error('Deleted')
                else:
                    st.info('In Progress')

def render_meeting_details(metadata_manager, project_id: str, meeting_id: str = None):
    """Render meeting details view.
    
    Args:
        metadata_manager: UnifiedMetadataManager instance
        project_id: Current project ID
        meeting_id: Optional meeting ID to display
    """
    if st.session_state.get('new_meeting', False):
        # New meeting form
        with st.form("new_meeting"):
            title = st.text_input("Meeting Title")
            description = st.text_area("Description")
            participants = st.multiselect(
                "Participants",
                options=["User 1", "User 2", "User 3"]  # TODO: Get from user service
            )
            
            if st.form_submit_button("Create Meeting"):
                if not title:
                    st.error("Title is required")
                    return
                    
                try:
                    # Generate meeting ID from title
                    meeting_id = metadata_manager.generate_meeting_id(title)
                    
                    # Create meeting metadata
                    metadata = {
                        "title": title,
                        "description": description,
                        "participants": participants,
                        "status": "in_progress"
                    }
                    
                    # Save meeting
                    metadata_manager.update_meeting_metadata(
                        project_id,
                        meeting_id,
                        metadata
                    )
                    
                    # Update state
                    st.session_state.selected_meeting = meeting_id
                    st.session_state.new_meeting = False
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Failed to create meeting: {str(e)}")
                    
    elif meeting_id:
        # Get meeting data
        meeting = metadata_manager.get_meeting_metadata(project_id, meeting_id)
        if not meeting:
            st.error("Meeting not found")
            return
            
        # Display meeting details
        st.header(meeting.get('title', 'Untitled Meeting'))
        
        # Status and actions
        col1, col2 = st.columns([3, 1])
        with col1:
            status = meeting.get('status', 'in_progress')
            if status == 'completed':
                st.success('Completed')
            elif status == 'deleted':
                st.error('Deleted')
            else:
                st.info('In Progress')
                
        # Admin actions
        if metadata_manager.is_admin:
            with col2:
                if status != 'deleted':
                    if st.button("Delete Meeting"):
                        try:
                            metadata_manager.delete_meeting(project_id, meeting_id)
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to delete meeting: {str(e)}")
                            
        # Meeting metadata
        st.subheader("Details")
        st.write(f"**Description:** {meeting.get('description', 'No description')}")
        st.write(f"**Participants:** {', '.join(meeting.get('participants', []))}")
        
        # Related documents
        st.subheader("Related Documents")
        docs = meeting.get('related_documents', [])
        if docs:
            for doc in docs:
                st.write(f"- {doc}")
        else:
            st.info("No related documents")
            
        # Meeting content
        if 'metadata' in meeting:
            st.subheader("Meeting Content")
            metadata = meeting['metadata']
            if 'transcript' in metadata:
                st.text_area("Transcript", value=metadata['transcript'], height=200)
            if 'summary' in metadata:
                st.text_area("Summary", value=metadata['summary'], height=100)

def render_meetings_page():
    """Render the meetings page with list and details view."""
    # Get metadata manager from session state
    if 'metadata_manager' not in st.session_state:
        st.error("Session not initialized")
        return
        
    if 'selected_project' not in st.session_state:
        st.error("No project selected")
        return
        
    metadata_manager = st.session_state.metadata_manager
    project = st.session_state.selected_project
    
    # Create two columns - left for list, right for details
    col1, col2 = st.columns([1, 2])
    
    # Meeting list
    with col1:
        st.subheader("Meetings")
        
        # Toggle for showing deleted meetings
        if metadata_manager.is_admin:
            show_deleted = st.checkbox(
                "Show Deleted Meetings",
                value=st.session_state.get('show_deleted', False),
                key='show_deleted'
            )
        
        render_meeting_list(metadata_manager, project['project_id'])
    
    # Meeting details
    with col2:
        st.subheader("Meeting Details")
        if st.session_state.get('new_meeting', False) or st.session_state.get('selected_meeting'):
            render_meeting_details(
                metadata_manager,
                project['project_id'],
                st.session_state.get('selected_meeting')
            )
        else:
            st.info("Select a meeting or create a new one")
