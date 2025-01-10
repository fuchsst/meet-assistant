"""Unified metadata management system for the Meeting Assistant using Snowflake."""
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Union
import uuid
from slugify import slugify
from snowflake.snowpark import Session
from snowflake.snowpark.functions import current_timestamp, col

logger = logging.getLogger(__name__)

class UnifiedMetadataManager:
    """Centralized metadata management system combining project, meeting, and content metadata."""
    
    def __init__(self, session: Session):
        """Initialize the unified metadata manager.
        
        Args:
            session: Snowflake session object
            
        Note:
            Requires either project_assistant_admin or project_assistant_service role
            for accessing pa_core schema tables.
            
        Raises:
            ValueError: If current role does not have required access
        """
        self.session = session
        
        # Verify role access
        current_role = self.session.sql("SELECT CURRENT_ROLE()").collect()[0][0]
        # if current_role not in ['PROJECT_ASSISTANT_ADMIN', 'PROJECT_ASSISTANT_SERVICE']:
        #     raise ValueError(
        #         f"Current role {current_role} does not have required access. "
        #         "Must be project_assistant_admin or project_assistant_service."
        #     )
        
        # Get table references from pa_core schema
        self.projects_table = self.session.table("pa_core.projects")
        self.meetings_table = self.session.table("pa_core.meetings") 
        self.documents_table = self.session.table("pa_core.documents")
        
        # Store role for access control
        self.is_admin = True #current_role == 'PROJECT_ASSISTANT_ADMIN'

    def get_project(self, project_id: Optional[str] = None) -> Dict:
        """Get project configuration.
        
        Args:
            project_id: Optional project ID. If not provided, returns first project.
            
        Returns:
            Dict containing project configuration
            
        Raises:
            ValueError: If no projects exist or project not found
            
        Note:
            Accessible by both project_assistant_admin and project_assistant_service roles
        """
        if not project_id:
            # Get default project from first project found
            df = self.projects_table.limit(1)
            if df.count() == 0:
                raise ValueError("No projects exist and no project ID provided")
            project = df.collect()[0].as_dict()
        else:
            df = self.projects_table.filter(col("project_id") == project_id)
            if df.count() == 0:
                raise ValueError(f"Project not found: {project_id}")
            project = df.collect()[0].as_dict()
            
        return project

    def create_project(self, project_id: str, name: str, description: str = "", config: Dict = None):
        """Create a new project.
        
        Args:
            project_id: Unique project identifier
            name: Project name
            description: Optional project description
            config: Optional project configuration
            
        Raises:
            PermissionError: If current role is not project_assistant_admin
        """
        if not self.is_admin:
            raise PermissionError(
                "Only project_assistant_admin role can create projects"
            )
        project_data = {
            "project_id": project_id,
            "name": name,
            "description": description,
            "pinned_documents": [],
            "config": config or {},
            "created_at": current_timestamp(),
            "updated_at": current_timestamp()
        }
        self.projects_table.insert([project_data])

    def update_project(self, project_id: str, updates: Dict):
        """Update project metadata.
        
        Args:
            project_id: Project to update
            updates: Dictionary of fields to update
            
        Raises:
            PermissionError: If current role is not project_assistant_admin
            ValueError: If project_id does not exist
        """
        if not self.is_admin:
            raise PermissionError(
                "Only project_assistant_admin role can update projects"
            )
            
        # Verify project exists
        if self.projects_table.filter(col("project_id") == project_id).count() == 0:
            raise ValueError(f"Project not found: {project_id}")
            
        updates["updated_at"] = current_timestamp()
        self.projects_table.update(
            updates,
            col("project_id") == project_id
        )

    def get_content_metadata(
        self,
        project_id: str,
        source_type: str,
        content_id: str
    ) -> Optional[Dict]:
        """Get metadata for specific content.
        
        Args:
            project_id: Project containing the content
            source_type: Type of content (e.g., 'confluence', 'jira', 'slack')
            content_id: Unique content identifier
            
        Returns:
            Dict containing content metadata if found, None otherwise
            
        Note:
            Accessible by both project_assistant_admin and project_assistant_service roles
        """
        df = self.documents_table.filter(
            (col("project_id") == project_id) &
            (col("source_type") == source_type) &
            (col("content_id") == content_id)
        )
        
        if df.count() == 0:
            return None
            
        return df.collect()[0].as_dict()

    def update_content_metadata(
        self,
        project_id: str,
        source_type: str,
        content_id: str,
        metadata: Dict
    ):
        """Update metadata for specific content.
        
        Args:
            project_id: Project containing the content
            source_type: Type of content (e.g., 'confluence', 'jira', 'slack')
            content_id: Unique content identifier
            metadata: Content metadata to update
            
        Raises:
            PermissionError: If current role is not project_assistant_admin
            ValueError: If project_id does not exist
        """
        if not self.is_admin:
            raise PermissionError(
                "Only project_assistant_admin role can update content metadata"
            )
            
        # Verify project exists
        if self.projects_table.filter(col("project_id") == project_id).count() == 0:
            raise ValueError(f"Project not found: {project_id}")
        # Check if record exists
        df = self.documents_table.filter(
            (col("project_id") == project_id) &
            (col("source_type") == source_type) &
            (col("content_id") == content_id)
        )
        
        update_data = {
            "metadata": metadata,
            "version": metadata.get("version", 0) + 1,
            "token_count": metadata.get("token_count"),
            "updated_at": current_timestamp()
        }
        
        if df.count() == 0:
            # Insert new record
            insert_data = {
                "content_id": content_id,
                "project_id": project_id,
                "source_type": source_type,
                "title": metadata.get("title"),
                "content_hash": metadata.get("content_hash"),
                **update_data
            }
            self.documents_table.insert([insert_data])
        else:
            # Update existing record
            self.documents_table.update(
                update_data,
                (col("project_id") == project_id) &
                (col("source_type") == source_type) &
                (col("content_id") == content_id)
            )

    def should_process_content(
        self,
        project_id: str,
        source_type: str,
        content_id: str,
        current_metadata: Dict
    ) -> bool:
        """Check if content should be processed based on changes.
        
        Args:
            project_id: Project containing the content
            source_type: Type of content (e.g., 'confluence', 'jira', 'slack')
            content_id: Unique content identifier
            current_metadata: Current content metadata to compare against stored
            
        Returns:
            bool: True if content should be processed, False otherwise
            
        Note:
            Accessible by both project_assistant_admin and project_assistant_service roles
            Returns True if content does not exist or if any tracked fields have changed
        """
        stored_metadata = self.get_content_metadata(project_id, source_type, content_id)
        
        if not stored_metadata:
            return True
        
        for key in ['content_hash', 'version', 'last_updated']:
            if key in current_metadata:
                if key not in stored_metadata or stored_metadata[key] != current_metadata[key]:
                    return True
        
        return False

    def generate_meeting_id(self, title: str) -> str:
        """Generate unique meeting ID based on current date and slugified title.
        
        Args:
            title: Meeting title to use for ID generation
            
        Returns:
            str: Generated meeting ID in format 'YYYYMMDD_slugified_title'
            
        Note:
            Accessible by both project_assistant_admin and project_assistant_service roles
            Uses current date and slugified title to ensure uniqueness and readability
        """
        date_str = datetime.now().strftime("%Y%m%d")
        title_slug = slugify(title)
        return f"{date_str}_{title_slug}"

    def update_meeting_metadata(
        self,
        project_id: str,
        meeting_id: str,
        metadata: Dict
    ):
        """Update meeting metadata.
        
        Args:
            project_id: Project containing the meeting
            meeting_id: Meeting to update
            metadata: Meeting metadata to update
            
        Raises:
            PermissionError: If current role is not project_assistant_admin
            ValueError: If project_id does not exist
        """
        if not self.is_admin:
            raise PermissionError(
                "Only project_assistant_admin role can update meeting metadata"
            )
            
        # Verify project exists
        if self.projects_table.filter(col("project_id") == project_id).count() == 0:
            raise ValueError(f"Project not found: {project_id}")
        # Check if record exists
        df = self.meetings_table.filter(
            (col("project_id") == project_id) &
            (col("meeting_id") == meeting_id)
        )
        
        update_data = {
            "metadata": metadata,
            "status": metadata.get("status", "in_progress"),
            "participants": metadata.get("participants", []),
            "related_documents": metadata.get("related_documents", []),
            "updated_at": current_timestamp()
        }
        
        if df.count() == 0:
            # Insert new record
            insert_data = {
                "meeting_id": meeting_id,
                "project_id": project_id,
                "title": metadata.get("title"),
                **update_data
            }
            self.meetings_table.insert([insert_data])
        else:
            # Update existing record
            self.meetings_table.update(
                update_data,
                (col("project_id") == project_id) &
                (col("meeting_id") == meeting_id)
            )

    def get_meeting_metadata(
        self,
        project_id: str,
        meeting_id: Optional[str] = None
    ) -> Dict:
        """Get meeting metadata.
        
        Args:
            project_id: Project containing the meeting(s)
            meeting_id: Optional specific meeting ID. If not provided, returns all meetings.
            
        Returns:
            Dict: If meeting_id provided, returns meeting metadata dict.
                 If no meeting_id, returns dict of {meeting_id: metadata} for all meetings.
                 Returns empty dict if meeting not found.
            
        Note:
            Accessible by both project_assistant_admin and project_assistant_service roles
            
        Example:
            >>> get_meeting_metadata("project1", "20240315_daily_standup")
            {'meeting_id': '20240315_daily_standup', 'title': 'Daily Standup', ...}
            
            >>> get_meeting_metadata("project1")
            {'20240315_daily_standup': {...}, '20240316_sprint_planning': {...}}
        """
        if meeting_id:
            df = self.meetings_table.filter(
                (col("project_id") == project_id) &
                (col("meeting_id") == meeting_id)
            )
            if df.count() == 0:
                return {}
            return df.collect()[0].as_dict()
        else:
            # Return all meetings for project
            df = self.meetings_table.filter(col("project_id") == project_id)
            return {row["meeting_id"]: row.as_dict() for row in df.collect()}

    def delete_meeting(self, project_id: str, meeting_id: str, backup: bool = True):
        """Delete meeting data.
        
        Args:
            project_id: Project ID containing the meeting
            meeting_id: Meeting to delete
            backup: If True, soft delete by updating status. If False, hard delete.
            
        Raises:
            PermissionError: If current role is not project_assistant_admin
            ValueError: If meeting not found
        """
        if not self.is_admin:
            raise PermissionError(
                "Only project_assistant_admin role can delete meetings"
            )
            
        # Verify meeting exists
        if self.meetings_table.filter(
            (col("project_id") == project_id) & 
            (col("meeting_id") == meeting_id)
        ).count() == 0:
            raise ValueError(f"Meeting not found: {meeting_id}")
            
        if backup:
            # Soft delete
            self.meetings_table.update(
                {
                    "status": "deleted",
                    "updated_at": current_timestamp()
                },
                (col("project_id") == project_id) &
                (col("meeting_id") == meeting_id)
            )
        else:
            # Hard delete if explicitly requested
            self.meetings_table.delete(
                (col("project_id") == project_id) &
                (col("meeting_id") == meeting_id)
            )

    def pin_document(self, project_id: str, content_id: str, source_type: str):
        """Pin a document to a project.
        
        Args:
            project_id: Project to pin document to
            content_id: Document content ID
            source_type: Type of content (e.g., 'confluence', 'jira', 'slack')
            
        Raises:
            PermissionError: If current role is not project_assistant_admin
            ValueError: If project or document not found
        """
        if not self.is_admin:
            raise PermissionError(
                "Only project_assistant_admin role can pin documents"
            )
            
        # Verify document exists
        if self.documents_table.filter(
            (col("project_id") == project_id) &
            (col("source_type") == source_type) &
            (col("content_id") == content_id)
        ).count() == 0:
            raise ValueError(f"Document not found: {content_id}")
        project = self.get_project(project_id)
        pinned_docs = project.get("pinned_documents", [])
        doc_ref = f"{source_type}:{content_id}"
        
        if doc_ref not in pinned_docs:
            pinned_docs.append(doc_ref)
            self.update_project(project_id, {"pinned_documents": pinned_docs})

    def unpin_document(self, project_id: str, content_id: str, source_type: str):
        """Unpin a document from a project.
        
        Args:
            project_id: Project to unpin document from
            content_id: Document content ID
            source_type: Type of content (e.g., 'confluence', 'jira', 'slack')
            
        Raises:
            PermissionError: If current role is not project_assistant_admin
            ValueError: If project or document not found
        """
        if not self.is_admin:
            raise PermissionError(
                "Only project_assistant_admin role can unpin documents"
            )
            
        # Verify document exists
        if self.documents_table.filter(
            (col("project_id") == project_id) &
            (col("source_type") == source_type) &
            (col("content_id") == content_id)
        ).count() == 0:
            raise ValueError(f"Document not found: {content_id}")
        project = self.get_project(project_id)
        pinned_docs = project.get("pinned_documents", [])
        doc_ref = f"{source_type}:{content_id}"
        
        if doc_ref in pinned_docs:
            pinned_docs.remove(doc_ref)
            self.update_project(project_id, {"pinned_documents": pinned_docs})
