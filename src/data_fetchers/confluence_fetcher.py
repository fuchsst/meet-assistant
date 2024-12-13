"""Fetch and process Confluence pages."""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from datetime import datetime, timedelta
import fire
from atlassian import Confluence
from bs4 import BeautifulSoup
from config.config import CONFLUENCE_CONFIG, DATA_DIR, LLM_CONFIG
from src.core.ai.llm_client import LLMClient
from src.core.storage.metadata_manager import UnifiedMetadataManager
from src.core.utils.logging_config import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logging()

# Constants for optimization based on LLM config
MAX_TOKENS = LLM_CONFIG["max_tokens"]
CHARS_PER_TOKEN = 4  # Approximate characters per token
MAX_CHUNK_CHARS = (MAX_TOKENS // 4) * 3  # Reserve 25% for response
MIN_SEGMENT_CHARS = 50  # Minimum size for independent processing

class ConfluenceFetcher:
    """Fetch and process Confluence pages."""
    
    def __init__(self, project_id: str):
        """Initialize the Confluence fetcher using config settings.
        
        Args:
            project_id: Project identifier for organizing content
        """
        logger.debug(f"Initializing ConfluenceFetcher for project {project_id}")
        
        if not all([CONFLUENCE_CONFIG["url"], 
                   CONFLUENCE_CONFIG["username"], 
                   CONFLUENCE_CONFIG["password"]]):
            logger.error("Confluence configuration incomplete - missing required credentials")
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
        logger.debug("Successfully initialized Confluence client")
        
        self.llm_client = LLMClient()
        self.metadata_manager = UnifiedMetadataManager()
        
        # Get project config using metadata manager
        self.project_config = self.metadata_manager.get_project(project_id)
        logger.info(f"ConfluenceFetcher initialized for project {project_id}")
    
    def _get_last_update_time(self, page_ids: List[str]) -> Optional[str]:
        """Get the last update time for a set of pages.
        
        Args:
            page_ids: List of Confluence page IDs
            
        Returns:
            ISO formatted datetime string of last update or None if no data
        """
        try:
            logger.debug(f"Getting last update time for {len(page_ids)} pages")
            last_update = None
            
            for page_id in page_ids:
                page_metadata = self.metadata_manager.get_content_metadata(
                    self.project_id, "confluence", page_id
                )
                if page_metadata:
                    page_updated = page_metadata.get("last_updated")
                    if page_updated and (not last_update or page_updated > last_update):
                        last_update = page_updated
            
            logger.debug(f"Last update time across all pages: {last_update}")
            return last_update
            
        except Exception as e:
            logger.error(f"Error getting last update time: {e}")
            return None
    
    def _get_updated_pages(self, page_ids: List[str], last_update: Optional[str] = None) -> List[str]:
        """Get list of pages that have been updated since last fetch.
        
        Args:
            page_ids: List of Confluence page IDs to check
            last_update: Optional timestamp to check updates since
            
        Returns:
            List of page IDs that need updating
        """
        try:
            logger.debug(f"Checking for updates in {len(page_ids)} pages")
            if not last_update:
                logger.debug("No last update time, will fetch all pages")
                return page_ids
            
            # Convert to datetime and subtract buffer
            update_time = (
                datetime.fromisoformat(last_update.replace('Z', '+00:00')) - 
                timedelta(minutes=1)
            )
            
            updated_pages = []
            for page_id in page_ids:
                try:
                    page = self.client.get_page_by_id(
                        page_id=page_id,
                        expand='version'
                    )
                    if datetime.fromisoformat(page['version']['when'].replace('Z', '+00:00')) > update_time:
                        updated_pages.append(page_id)
                except Exception as e:
                    logger.error(f"Error checking page {page_id}: {e}")
                    # Include page if we can't check its status
                    updated_pages.append(page_id)
            
            logger.info(f"Found {len(updated_pages)} pages needing update")
            return updated_pages
            
        except Exception as e:
            logger.error(f"Error checking for updated pages: {e}")
            return page_ids

    def _convert_html_to_markdown(self, html_content: str, project_context: Dict) -> str:
        """Convert HTML content to Markdown using incremental processing.
        
        Args:
            html_content: HTML content to convert
            project_context: Project context for conversion
            
        Returns:
            Converted markdown content
        """
        logger.debug("Converting HTML to Markdown using incremental processing")
        
        # Split content into manageable chunks
        chunks = [html_content[i:i + MAX_CHUNK_CHARS] 
                 for i in range(0, len(html_content), MAX_CHUNK_CHARS)]
        
        converted_chunks = []
        previous_chunks = []  # Store previous chunks for context
        
        for i, chunk in enumerate(chunks):
            try:
                # Build conversion context
                context = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "previous_chunks": previous_chunks[-2:],  # Last 2 chunks for context
                    "project_context": project_context,
                    "conversion_parameters": {
                        "preserve_formatting": True,
                        "maintain_links": True,
                        "handle_tables": True,
                        "process_code_blocks": True
                    }
                }
                
                task_description = f"""You are a document conversion expert. Your task is to convert a Confluence page from HTML to Markdown format.

Project Context:
Project: {project_context['name']} ({project_context['key']})
Description: {project_context['description']}

Team Members:
{yaml.dump([{"name": m["name"], "role": m["role"]} for m in project_context["members"]], default_flow_style=False)}

Primary Objectives:
1. Convert HTML content to clean, well-structured Markdown
2. Preserve all important information and formatting
3. Replace user mentions with full names

Specific Requirements:
1. Document Structure:
   - Maintain heading hierarchy (h1 -> #, h2 -> ##, etc.)
   - Preserve list formatting (bullet points and numbered lists)
   - Keep table structures intact
   - Retain code block formatting with language hints

2. Content Handling:
   - Keep all meaningful content
   - Remove HTML artifacts and styling
   - Preserve links and images
   - Maintain emphasis (bold, italic) where important

3. User Mentions:
   Replace any usernames or email addresses with full names using this mapping:
{yaml.dump({username.lower(): member["name"]
           for member in project_context["members"]
           for username in member.get("usernames", [])}, default_flow_style=False)}

4. Metadata Integration:
   - Include relevant status information
   - Preserve comment context
   - Maintain author and timestamp information where relevant

Format the output as clean Markdown without any explanations or additional context.
Always return the full chunk content without any truncation or summarization.

This is chunk {i + 1} of {len(chunks)}. Process it while maintaining consistency with previous chunks.
"""
                
                converted_chunk = self.llm_client.execute_custom_task(
                    chunk,
                    task_description,
                    additional_context=context
                )
                
                # Validate converted chunk
                if not converted_chunk or len(converted_chunk.strip()) < MIN_SEGMENT_CHARS:
                    raise ValueError("Converted chunk is too short or empty")
                
                converted_chunks.append(converted_chunk)
                previous_chunks.append(converted_chunk)
                
            except Exception as e:
                logger.error(f"Error converting chunk {i + 1}: {e}")
                # On error, include original chunk to maintain content
                converted_chunks.append(chunk)
                previous_chunks.append(chunk)
        
        # Combine chunks and clean up
        markdown_content = "\n".join(converted_chunks)
        return markdown_content
    
    def fetch_page(self, page_id: str) -> Dict:
        """Fetch a Confluence page by ID.
        
        Args:
            page_id: Confluence page ID
            
        Returns:
            Dict containing page content and metadata
        """
        try:
            logger.info(f"Fetching Confluence page {page_id}")
            page = self.client.get_page_by_id(
                page_id=page_id,
                expand='body.storage,version,space,history,metadata'
            )
            
            # Extract metadata
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
                "labels": [label['name'] for label in page.get('metadata', {}).get('labels', {}).get('results', [])],
                "content_hash": hash(str(page['body']['storage']['value']))
            }
            
            # Check if content needs processing
            if not self.metadata_manager.should_process_content(
                self.project_id, "confluence", page_id, metadata
            ):
                logger.info(f"Page {page_id} unchanged, skipping processing")
                stored_metadata = self.metadata_manager.get_content_metadata(
                    self.project_id, "confluence", page_id
                )
                return {
                    "content": stored_metadata["content"],
                    "metadata": metadata
                }
            
            logger.info(f"Processing page {page_id} content")
            
            # Prepare project context for LLM
            project_context = {
                "name": self.project_config["name"],
                "key": self.project_config["key"],
                "description": self.project_config["description"],
                "members": [
                    {
                        "name": m["name"],
                        "role": m.get("role", ""),
                        "usernames": m.get("user_names", [])
                    }
                    for m in self.project_config.get("members", [])
                ]
            }
            
            # Convert HTML to Markdown using incremental processing
            markdown_content = self._convert_html_to_markdown(
                page['body']['storage']['value'],
                project_context
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
            logger.error(f"Error fetching Confluence page {page_id}: {e}", exc_info=True)
            raise
    
    def save_page(self, page_data: Dict) -> Dict:
        """Save Confluence page content and metadata.
        
        Args:
            page_data: Dict containing page content and metadata
            
        Returns:
            Dict with file paths and metadata
        """
        try:
            logger.info(f"Saving page {page_data['metadata']['page_id']}")
            # Get content path from metadata manager
            content_path = self.metadata_manager.get_content_path(
                self.project_id,
                "confluence",
                f"{page_data['metadata']['page_id']}"
            )
            
            # Save content
            content_path.write_text(page_data["content"])
            logger.debug(f"Saved page content to {content_path}")
            
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
                "type": "confluence_page",
                "content_hash": page_data["metadata"]["content_hash"]
            }
            
        except Exception as e:
            logger.error(f"Error saving Confluence page: {e}", exc_info=True)
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
        logger.info("Starting Confluence page fetch process")
        metadata_manager = UnifiedMetadataManager()
        
        # Get all projects from metadata manager
        config = metadata_manager._load_project_config()
        
        # Process each project
        for project in config.get("projects", []):
            if "confluence" not in project:
                logger.debug(f"Skipping project {project.get('key', 'unknown')} - no Confluence configuration")
                continue
            
            project_metadata = []
            logger.info(f"Processing project: {project['key']}")
            
            # Initialize fetcher with project ID
            fetcher = ConfluenceFetcher(project["key"])
            
            # Get all page IDs from config
            page_ids = [page["id"] for page in project["confluence"]["pages"]]
            
            # Get last update time
            last_update = fetcher._get_last_update_time(page_ids)
            
            # Get list of pages needing update
            pages_to_update = fetcher._get_updated_pages(page_ids, last_update)
            
            if not pages_to_update:
                logger.info("No pages need updating")
                continue
            
            # Fetch each page that needs updating
            for page_id in pages_to_update:
                try:
                    page_info = next(
                        (p for p in project["confluence"]["pages"] if p["id"] == page_id),
                        None
                    )
                    if not page_info:
                        continue
                        
                    logger.info(f"Processing page: {page_id} - {page_info['title']}")
                    page_data = fetcher.fetch_page(page_id)
                    metadata = fetcher.save_page(page_data)
                    project_metadata.append(metadata)
                    logger.info(f"Successfully processed page: {page_id} - {page_info['title']}")
                except Exception as e:
                    logger.error(f"Error processing page {page_id}: {e}")
                    continue
            
            # Save project metadata
            if project_metadata:
                logger.info(f"Successfully processed {len(project_metadata)} pages for project {project['key']}")
                logger.info(f"Content saved to: {DATA_DIR / project['key'] / 'confluence'}")
                logger.info(f"Metadata saved to: {DATA_DIR / project['key'] / 'project_metadata.json'}")
        
    except Exception as e:
        logger.error(f"Failed to fetch Confluence pages: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    fire.Fire(fetch_confluence_pages)
