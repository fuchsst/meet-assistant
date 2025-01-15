import streamlit as st
import logging

from src.core.storage.metadata_manager import UnifiedMetadataManager
from src.core.utils.logging_config import setup_logging
from src.streamlit.components.meetings import render_meetings_page
from src.streamlit.components.tickets import render_tickets_page
from src.streamlit.components.documents import render_documents_page
from src.streamlit.components.chats import render_chats_page
from src.streamlit.components.project_config import render_project_config_page

logger = logging.getLogger(__name__)

class MeetingAssistantApp:
    """Streamlit interface for the Meeting Assistant."""
    
    def __init__(self):
        """Initialize the Meeting Assistant application."""
        self.setup_page_config()
        
        # Initialize Snowflake session and metadata manager
        try:
            from config.config import get_snowflake_session
            
            logger.info("Initializing Snowflake session...")
            # Get or create Snowflake session
            session = get_snowflake_session()
            
            # Initialize metadata manager with session
            self.metadata_manager = UnifiedMetadataManager(session)
            
            # Store in session state for page access
            if 'metadata_manager' not in st.session_state:
                st.session_state.metadata_manager = self.metadata_manager
                logger.info("Metadata manager initialized and stored in session state")
                
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake connection: {str(e)}")
            st.error(f"Failed to initialize Snowflake connection: {str(e)}")
            st.stop()
        
    def setup_page_config(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Meeting Assistant",
            page_icon="👥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_projects(self):
        """Load projects from metadata manager.
        
        Returns:
            list: List of project dictionaries with project_id and name
        """
        try:
            logger.info("Checking database connection...")
            # Get first project to check access
            project = self.metadata_manager.get_project()
            if not project:
                logger.info("No projects found in initial check")
                return []
                
            # Get all projects by listing them
            logger.info("Fetching all projects...")
            projects = []
            df = self.metadata_manager.projects_table.select(
                ["project_id", "name", "description"]
            ).collect()
            
            logger.info(f"Found {len(df)} projects")
            
            for row in df:
                projects.append({
                    "key": row["project_id"],
                    "name": row["name"],
                    "description": row.get("description", "")
                })
            
            return projects
            
        except ValueError as e:
            if "No projects exist" in str(e):
                logger.info("No projects exist in database")
                return []
            logger.error(f"Error loading projects: {str(e)}")
            st.error(f"Error loading projects: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error loading projects: {str(e)}")
            st.error(f"Error loading projects: {str(e)}")
            return []
    
    def setup_sidebar(self):
        """Setup the sidebar with project selection and navigation.
        
        Returns:
            function: Selected page renderer function or None if no project selected
        """
        st.sidebar.title("Navigation")
        
        try:
            # Project selection
            projects = self.load_projects()
            project_names = {p["key"]: p["name"] for p in projects}
            
            if not projects:
                st.sidebar.warning("No projects available. Please create a project first.")
                
            
            # Add Create Project button for admins when projects exist
            if self.metadata_manager.is_admin:
                if st.sidebar.button("Create New Project"):
                    logger.info("Create New Project button clicked")
                    st.session_state.selected_page = "Project Settings"
                    # Clear selected project to show creation form
                    if 'selected_project' in st.session_state:
                        del st.session_state.selected_project
                    return render_project_config_page
            
            selected_project_key = st.sidebar.selectbox(
                "Select Project",
                options=list(project_names.keys()),
                format_func=lambda x: project_names[x]
            )
            
            if selected_project_key:
                logger.info(f"Selected project: {selected_project_key}")
                # Get full project details
                project = self.metadata_manager.get_project(selected_project_key)
                
                # Store in session state
                st.session_state.selected_project = project
            
                # Page navigation
                st.sidebar.subheader("Pages")
                pages = {
                    "Meetings": render_meetings_page,
                    "Documents": render_documents_page,
                    "Tickets": render_tickets_page,
                    "Chats": render_chats_page
                }
                
                # Add Project Settings for admin users
                if self.metadata_manager.is_admin:
                    pages["Project Settings"] = render_project_config_page
                
                # Get selected page from state or radio
                if 'selected_page' not in st.session_state:
                    st.session_state.selected_page = "Meetings"
                    
                selected_page = st.sidebar.radio(
                    "",
                    list(pages.keys()),
                    index=list(pages.keys()).index(st.session_state.selected_page)
                )
                st.session_state.selected_page = selected_page
                logger.info(f"Selected page: {selected_page}")
                
                return pages[selected_page]
            
            return None
            
        except Exception as e:
            logger.error(f"Error setting up navigation: {str(e)}")
            st.sidebar.error(f"Error setting up navigation: {str(e)}")
            return None
    
    def main(self):
        """Main application entry point."""
        try:
            st.title("Meeting Assistant")
            
            # Setup sidebar and get selected page renderer
            page_renderer = self.setup_sidebar()
            
            # Render selected page if project is selected
            if page_renderer:
                # Pass metadata manager to page renderer
                page_renderer()
            elif not self.metadata_manager.is_admin:
                st.info("Please select a project to continue")
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error(f"Application error: {str(e)}")
            st.error("Please try refreshing the page or contact support if the error persists.")

if __name__ == "__main__":
    # Initialize logging
    setup_logging()
    logger.info("Starting Meeting Assistant application")
    
    # Start application
    app = MeetingAssistantApp()
    app.main()
