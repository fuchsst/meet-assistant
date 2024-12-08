"""Fetch and process web pages."""
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional
import fire
import requests
from bs4 import BeautifulSoup
import hashlib
from config.config import WEB_CONFIG, DATA_DIR
from src.core.ai.llm_client import LLMClient
from src.core.storage.metadata_manager import UnifiedMetadataManager

logger = logging.getLogger(__name__)

class WebFetcher:
    """Fetch and process web pages."""
    
    def __init__(self, project_id: str):
        """Initialize the web fetcher with config settings.
        
        Args:
            project_id: Project identifier for organizing content
        """
        self.project_id = project_id
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': WEB_CONFIG["user_agent"]
        })
        self.timeout = WEB_CONFIG["timeout"]
        self.max_retries = WEB_CONFIG["max_retries"]
        self.llm_client = LLMClient()
        self.metadata_manager = UnifiedMetadataManager()
    
    def fetch_page(self, url: str) -> Dict:
        """Fetch a web page.
        
        Args:
            url: Web page URL
            
        Returns:
            Dict containing page content and metadata
        """
        try:
            retries = 0
            while retries < self.max_retries:
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    retries += 1
                    if retries == self.max_retries:
                        raise e
                    logger.warning(f"Retry {retries} for URL {url}: {str(e)}")
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title and description
            title = self._extract_title(soup)
            description = self._extract_description(soup)
            
            # Generate content hash
            content_hash = hashlib.md5(response.content).hexdigest()
            
            # Create metadata
            metadata = {
                "title": title,
                "url": url,
                "description": description,
                "content_hash": content_hash,
                "last_updated": datetime.utcnow().isoformat(),
                "content_type": response.headers.get('content-type', ''),
                "size": len(response.content),
                "type": "web_page"
            }
            
            # Check if content needs processing
            if not self.metadata_manager.should_process_content(
                self.project_id, "web", url, metadata
            ):
                logger.info(f"Content unchanged for {url}, skipping processing")
                stored_metadata = self.metadata_manager.get_content_metadata(
                    self.project_id, "web", url
                )
                return {
                    "content": stored_metadata["content"],
                    "metadata": metadata
                }
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
                element.decompose()
            
            # Convert to markdown using LLM
            task_description = """
            Convert this web page content to well-formatted markdown:
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
                self.project_id, "web", url, metadata
            )
            
            return {
                "content": markdown_content,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error fetching web page {url}: {e}")
            raise
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        # Try meta title first
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title['content']
        
        # Fall back to title tag
        if soup.title:
            return soup.title.string.strip()
        
        return "Untitled Page"
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description from HTML."""
        # Try meta description first
        meta_desc = soup.find('meta', {'name': 'description'}) or \
                   soup.find('meta', property='og:description')
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content']
        
        # Fall back to first paragraph
        first_p = soup.find('p')
        if first_p:
            return first_p.get_text().strip()
        
        return "No description available"
    
    def save_page(self, page_data: Dict) -> Dict:
        """Save web page content and metadata.
        
        Args:
            page_data: Dict containing page content and metadata
            
        Returns:
            Dict with file paths and metadata
        """
        try:
            # Get content path from metadata manager
            content_path = self.metadata_manager.get_content_path(
                self.project_id,
                "web",
                f"{page_data['metadata']['content_hash']}.md"
            )
            
            # Save content
            content_path.write_text(page_data["content"])
            
            # Return metadata with file path
            return {
                "title": page_data["metadata"]["title"],
                "path": str(content_path.relative_to(DATA_DIR)),
                "url": page_data["metadata"]["url"],
                "description": page_data["metadata"]["description"],
                "last_changed": page_data["metadata"]["last_updated"],
                "content_type": page_data["metadata"]["content_type"],
                "size": page_data["metadata"]["size"],
                "type": "web_page"
            }
            
        except Exception as e:
            logger.error(f"Error saving web page: {e}")
            raise

def fetch_web_pages(
    project_config: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """CLI tool to fetch web pages.
    
    This script fetches web pages defined in the project config and saves them as markdown files.
    
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
            if "web" not in project:
                continue
            
            project_metadata = []
            
            # Initialize fetcher with project ID
            fetcher = WebFetcher(project["key"])
            
            # Fetch each web page
            for page in project["web"]:
                try:
                    page_data = fetcher.fetch_page(page["url"])
                    metadata = fetcher.save_page(page_data)
                    project_metadata.append(metadata)
                    print(f"Fetched: {page['url']}")
                except Exception as e:
                    logger.error(f"Error processing URL {page['url']}: {e}")
                    continue
            
            # Save project metadata
            if project_metadata:
                print(f"\nSuccessfully processed {len(project_metadata)} pages for project {project['key']}")
                print(f"Content saved to: {DATA_DIR / project['key'] / 'web'}")
                print(f"Metadata saved to: {DATA_DIR / project['key'] / 'project_metadata.json'}")
        
    except Exception as e:
        logger.error(f"Error fetching web pages: {e}")
        raise

if __name__ == "__main__":
    fire.Fire(fetch_web_pages)
