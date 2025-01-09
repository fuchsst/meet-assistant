import streamlit as st

def render_project_config_page():
    """Render the project configuration page."""
    st.header("Project Configuration")
    
    if not hasattr(st.session_state, 'selected_project'):
        st.warning("No project selected")
        return
    
    project = st.session_state.selected_project
    
    # Project Info Section
    st.subheader("Project Information")
    st.info(f"""
    Name: {project['name']}
    Key: {project['key']}
    Description: {project['description']}
    """)
    
    # Placeholder for configuration sections
    st.subheader("Integration Settings")
    
    # Confluence Settings
    with st.expander("Confluence Settings"):
        st.info("Confluence configuration options will be added here")
    
    # Jira Settings
    with st.expander("Jira Settings"):
        st.info("Jira configuration options will be added here")
    
    # Slack Settings
    with st.expander("Slack Settings"):
        st.info("Slack configuration options will be added here")
    
    # Web Resources
    with st.expander("Web Resources"):
        st.info("Web resources configuration will be added here")
    
    # Team Members
    with st.expander("Team Members"):
        st.info("Team member management will be added here")
