"""Fetch and process Confluence pages."""
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional
import fire
from atlassian import Confluence
from bs4 import BeautifulSoup
from config.config import CONFLUENCE_CONFIG, DATA_DIR
from src.core.ai.llm_client import LLMClient
from src.core.storage.metadata_manager import UnifiedMetadataManager

logger = logging.getLogger(__name__)

class ConfluenceFetcher:
    """Fetch and process Confluence pages."""
    
    def __init__(self, project_id: str):
        """Initialize the Confluence fetcher using config settings.
        
        Args:
            project_id: Project identifier for organizing content
        """
        if not all([CONFLUENCE_CONFIG["url"], 
                   CONFLUENCE_CONFIG["username"], 
                   CONFLUENCE_CONFIG["password"]]):
            raise ValueError(
                "Confluence configuration incomplete. Ensure CONFLUENCE_URL, "
                "CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN environment "
                "variables are set."
            )
        
        self.project_id = project_id
        self.client = Confluence(
            url=CONFLUENCE_CONFIG["url"],
            username=CONFLUENCE_CONFIG["username"],
            password=CONFLUENCE_CONFIG["password"],
            cloud=CONFLUENCE_CONFIG["cloud"]
        )
        self.llm_client = LLMClient()
        self.metadata_manager = UnifiedMetadataManager()
    
    def fetch_page(self, page_id: str) -> Dict:
        """Fetch a Confluence page by ID.
        
        Args:
            page_id: Confluence page ID
            
        Returns:
            Dict containing page content and metadata
        """
        try:
            page = self.client.get_page_by_id(
                page_id=page_id,
                expand='body.storage,version,space,history,metadata'
            )
            
            metadata = {
                "title": page['title'],
                "page_id": page_id,
                "space_key": page['space']['key'],
                "space_name": page['space']['name'],
                "version": page['version']['number'],
                "created": page['history']['createdDate'],
                "creator": page['history']['createdBy']['displayName'],
                "last_updated": page['version']['when'],
                "last_updater": page['version']['by']['displayName'],
                "content_type": "confluence_page",
                "type": "confluence_page",
                "status": page.get('status', 'current'),
                "labels": [label['name'] for label in page.get('metadata', {}).get('labels', {}).get('results', [])]
            }
            
            # Check if content needs processing
            if not self.metadata_manager.should_process_content(
                self.project_id, "confluence", page_id, metadata
            ):
                logger.info(f"Content unchanged for page {page_id}, skipping processing")
                stored_metadata = self.metadata_manager.get_content_metadata(
                    self.project_id, "confluence", page_id
                )
                return {
                    "content": stored_metadata["content"],
                    "metadata": metadata
                }
            
            # Clean HTML content
            html_content = page['body']['storage']['value']
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
                element.decompose()
            
            # Convert to markdown using LLM
            task_description = """
            Convert this Confluence page content to well-formatted markdown:
            1. Preserve the document structure (headings, lists, tables, etc.)
            2. Keep all important content and formatting
            3. Remove any unnecessary HTML artifacts
            4. Ensure code blocks are properly formatted
            5. Maintain the original meaning and context
            
            Return only the markdown content without any explanations or metadata.
            """
            
            markdown_content = self.llm_client.execute_custom_task(
                str(soup),
                task_description
            )
            
            # Update metadata store with processed content
            metadata["content"] = markdown_content
            self.metadata_manager.update_content_metadata(
                self.project_id, "confluence", page_id, metadata
            )
            
            return {
                "content": markdown_content,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error fetching Confluence page {page_id}: {e}")
            raise
    
    def save_page(self, page_data: Dict) -> Dict:
        """Save Confluence page content and metadata.
        
        Args:
            page_data: Dict containing page content and metadata
            
        Returns:
            Dict with file paths and metadata
        """
        try:
            # Get content path from metadata manager
            content_path = self.metadata_manager.get_content_path(
                self.project_id,
                "confluence",
                f"{page_data['metadata']['page_id']}.md"
            )
            
            # Save content
            content_path.write_text(page_data["content"])
            
            # Return metadata with file path
            return {
                "title": page_data["metadata"]["title"],
                "path": str(content_path.relative_to(DATA_DIR)),
                "page_id": page_data["metadata"]["page_id"],
                "space_key": page_data["metadata"]["space_key"],
                "space_name": page_data["metadata"]["space_name"],
                "version": page_data["metadata"]["version"],
                "created": page_data["metadata"]["created"],
                "creator": page_data["metadata"]["creator"],
                "last_changed": page_data["metadata"]["last_updated"],
                "last_updater": page_data["metadata"]["last_updater"],
                "status": page_data["metadata"]["status"],
                "labels": page_data["metadata"]["labels"],
                "type": "confluence_page"
            }
            
        except Exception as e:
            logger.error(f"Error saving Confluence page: {e}")
            raise

def fetch_confluence_pages(
    project_config: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """CLI tool to fetch Confluence pages.
    
    This script fetches pages defined in the project config and saves them as markdown files.
    It uses environment variables for authentication:
    - CONFLUENCE_URL: Base URL of your Confluence instance
    - CONFLUENCE_USERNAME: Your username
    - CONFLUENCE_API_TOKEN: Your API token (for cloud) or password (for server)
    - CONFLUENCE_CLOUD: "true" for cloud, "false" for server (default: "true")
    
    Args:
        project_config: Path to project config YAML (default: data/projects.yaml)
        output_dir: Optional directory to save content (default: data/meetings/current_date)
    """
    try:
        # Load project config
        if project_config:
            config_path = Path(project_config)
        else:
            config_path = DATA_DIR / "projects.yaml"
        
        if not config_path.exists():
            raise ValueError(f"Project config not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Process each project
        for project in config.get("projects", []):
            if "confluence" not in project:
                continue
            
            project_metadata = []
            
            # Initialize fetcher with project ID
            fetcher = ConfluenceFetcher(project["key"])
            
            # Fetch each page
            for page in project["confluence"]["pages"]:
                try:
                    page_data = fetcher.fetch_page(page["id"])
                    metadata = fetcher.save_page(page_data)
                    project_metadata.append(metadata)
                    print(f"Fetched: {page['title']} ({page['id']})")
                except Exception as e:
                    logger.error(f"Error processing page {page['id']}: {e}")
                    continue
            
            # Save project metadata
            if project_metadata:
                print(f"\nSuccessfully processed {len(project_metadata)} pages for project {project['key']}")
                print(f"Content saved to: {DATA_DIR / project['key'] / 'confluence'}")
                print(f"Metadata saved to: {DATA_DIR / project['key'] / 'project_metadata.json'}")
        
    except Exception as e:
        logger.error(f"Error fetching Confluence pages: {e}")
        raise

if __name__ == "__main__":
    fire.Fire(fetch_confluence_pages)
