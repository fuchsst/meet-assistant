import streamlit as st

def render_chats_page():
    """Render the chats page with channels, threads and details view."""
    # Create two columns - left for channels/threads, right for details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Slack Channels")
        # Placeholder for channels list
        st.info("Slack channels list will be displayed here")
        
        st.subheader("Threads")
        # Placeholder for threads list
        st.info("Channel threads will be displayed here when a channel is selected")
        
    with col2:
        st.subheader("Chat Details")
        # Placeholder for chat details
        st.info("Selected thread content will be displayed here")
