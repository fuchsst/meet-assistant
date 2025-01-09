import streamlit as st

def render_tickets_page():
    """Render the tickets page with list and details view."""
    # Create two columns - left for list, right for details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Jira Tickets")
        # Placeholder for tickets list
        st.info("Jira tickets list will be displayed here")
        
    with col2:
        st.subheader("Ticket Details")
        # Placeholder for ticket details and actions
        st.info("Selected ticket details and actions will be displayed here")
