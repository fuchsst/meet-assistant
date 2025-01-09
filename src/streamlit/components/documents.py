import streamlit as st

def render_documents_page():
    """Render the documents page with list and details view."""
    # Create two columns - left for list, right for details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Confluence Pages")
        # Placeholder for documents list
        st.info("Confluence pages list will be displayed here")
        
    with col2:
        st.subheader("Document Details")
        # Placeholder for document details
        st.info("Selected document content will be displayed here")
