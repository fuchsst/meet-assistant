"""Unified metadata management system for the Meeting Assistant."""
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from datetime import datetime
import uuid
import yaml
import jsonschema

from config.config import DATA_DIR

logger = logging.getLogger(__name__)

class UnifiedMetadataManager:
    """Centralized metadata management system combining project, meeting, and content metadata."""
    
    def __init__(self):
        """Initialize the unified metadata manager."""
        self.data_dir = DATA_DIR / "projects"  # Base all project data under projects/
        self.config_path = DATA_DIR / "projects.yaml"
        self.schema_path = Path("config/projects_schema.json")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load schema for validation
        with open(self.schema_path, 'r') as f:
            self.project_schema = json.load(f)
    
    def _get_meetings_dir(self, project_id: str) -> Path:
        """Get project-specific meetings directory."""
        return self.data_dir / project_id / "meetings"
    
    def _get_documents_dir(self, project_id: str) -> Path:
        """Get project-specific documents directory."""
        return self.data_dir / project_id / "documents"
    
    def _load_project_config(self) -> Dict:
        """Load and validate project configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    jsonschema.validate(instance=config, schema=self.project_schema)
                    return config
            return {"projects": [], "default_project": None}
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Project config validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading project config: {e}")
            return {"projects": [], "default_project": None}
    
    def _load_project_metadata(self, project_id: str) -> Dict:
        """Load project-specific metadata."""
        metadata_path = self.data_dir / project_id / "project_metadata.json"
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return {
                "meetings": {},
                "documents": {
                    "confluence": {},
                    "jira": {},
                    "web": {}
                },
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error loading project metadata: {e}")
            return {}
    
    def _save_project_metadata(self, project_id: str, metadata: Dict):
        """Save project-specific metadata."""
        try:
            metadata_path = self.data_dir / project_id / "project_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update last modified timestamp
            metadata["last_updated"] = datetime.utcnow().isoformat()
            
            # Atomic write using temporary file
            temp_path = metadata_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            temp_path.replace(metadata_path)
            
        except Exception as e:
            logger.error(f"Error saving project metadata: {e}")
            raise
    
    # Project Management
    
    def get_project(self, project_id: Optional[str] = None) -> Dict:
        """Get project configuration."""
        config = self._load_project_config()
        
        if not project_id:
            project_id = config.get("default_project")
            if not project_id:
                raise ValueError("No project ID provided and no default project set")
        
        for project in config.get("projects", []):
            if project.get("key") == project_id:
                return project
        raise ValueError(f"Project not found: {project_id}")
    
    def set_default_project(self, project_id: str):
        """Set the default project."""
        config = self._load_project_config()
        
        # Verify project exists
        if not any(p.get("key") == project_id for p in config.get("projects", [])):
            raise ValueError(f"Project not found: {project_id}")
        
        config["default_project"] = project_id
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    # Content Management
    
    def get_content_metadata(
        self,
        project_id: str,
        source_type: str,
        content_id: str
    ) -> Optional[Dict]:
        """Get metadata for specific content."""
        metadata = self._load_project_metadata(project_id)
        return metadata.get("documents", {}).get(source_type, {}).get(content_id)
    
    def update_content_metadata(
        self,
        project_id: str,
        source_type: str,
        content_id: str,
        metadata: Dict
    ):
        """Update metadata for specific content."""
        project_metadata = self._load_project_metadata(project_id)
        
        if "documents" not in project_metadata:
            project_metadata["documents"] = {}
        if source_type not in project_metadata["documents"]:
            project_metadata["documents"][source_type] = {}
        
        project_metadata["documents"][source_type][content_id] = {
            **metadata,
            "last_processed": datetime.utcnow().isoformat(),
            "version": metadata.get("version", 0) + 1
        }
        
        self._save_project_metadata(project_id, project_metadata)
    
    def should_process_content(
        self,
        project_id: str,
        source_type: str,
        content_id: str,
        current_metadata: Dict
    ) -> bool:
        """Check if content should be processed based on changes."""
        stored_metadata = self.get_content_metadata(project_id, source_type, content_id)
        
        if not stored_metadata:
            return True
        
        for key in ['content_hash', 'version', 'last_updated']:
            if key in current_metadata:
                if key not in stored_metadata or stored_metadata[key] != current_metadata[key]:
                    return True
        
        return False
    
    def get_content_path(
        self,
        project_id: str,
        source_type: str,
        content_id: str
    ) -> Path:
        """Get content storage path."""
        content_dir = self._get_documents_dir(project_id) / source_type
        content_dir.mkdir(parents=True, exist_ok=True)
        return content_dir / f"{content_id}.md"
    
    # Meeting Management
    
    def generate_meeting_id(self) -> str:
        """Generate unique meeting ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"meeting_{timestamp}_{unique_id}"
    
    def get_meeting_dir(self, project_id: str, meeting_id: str) -> Path:
        """Get meeting directory path."""
        meetings_dir = self._get_meetings_dir(project_id)
        meetings_dir.mkdir(parents=True, exist_ok=True)
        return meetings_dir / meeting_id
    
    def update_meeting_metadata(
        self,
        project_id: str,
        meeting_id: str,
        metadata: Dict
    ):
        """Update meeting metadata."""
        project_metadata = self._load_project_metadata(project_id)
        
        if "meetings" not in project_metadata:
            project_metadata["meetings"] = {}
        
        # Ensure required fields are present
        meeting_metadata = {
            **project_metadata["meetings"].get(meeting_id, {}),
            **metadata,
            "project_id": project_id,
            "last_modified": datetime.utcnow().isoformat(),
            "status": metadata.get("status", "in_progress"),
            "participants": metadata.get("participants", []),
            "related_documents": metadata.get("related_documents", [])
        }
        
        project_metadata["meetings"][meeting_id] = meeting_metadata
        self._save_project_metadata(project_id, project_metadata)
    
    def get_meeting_metadata(
        self,
        project_id: str,
        meeting_id: Optional[str] = None
    ) -> Dict:
        """Get meeting metadata."""
        project_metadata = self._load_project_metadata(project_id)
        if meeting_id:
            return project_metadata.get("meetings", {}).get(meeting_id, {})
        return project_metadata.get("meetings", {})
    
    def get_meeting_files(self, project_id: str, meeting_id: str) -> Dict[str, Path]:
        """Get meeting file paths.
        - Final transcript is saved as transcript.md
        - Analysis is saved as analysis.md
        """
        meeting_dir = self.get_meeting_dir(project_id, meeting_id)
        return {
            "transcript": meeting_dir / "transcript.md", 
            "analysis": meeting_dir / "analysis.md"
        }
    
    def create_meeting_backup(self, project_id: str, meeting_id: str) -> Path:
        """Create meeting backup."""
        meeting_dir = self.get_meeting_dir(project_id, meeting_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self._get_meetings_dir(project_id) / "backups" / f"{meeting_id}_{timestamp}"
        
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(meeting_dir, backup_dir)
        
        return backup_dir
    
    def delete_meeting(self, project_id: str, meeting_id: str, backup: bool = True):
        """Delete meeting data."""
        if backup:
            self.create_meeting_backup(project_id, meeting_id)
        
        meeting_dir = self.get_meeting_dir(project_id, meeting_id)
        shutil.rmtree(meeting_dir)
        
        project_metadata = self._load_project_metadata(project_id)
        if meeting_id in project_metadata.get("meetings", {}):
            del project_metadata["meetings"][meeting_id]
            self._save_project_metadata(project_id, project_metadata)
