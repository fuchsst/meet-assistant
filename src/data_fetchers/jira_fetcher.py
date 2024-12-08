"""Fetch and process Jira tickets."""
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional
import fire
from atlassian import Jira
from config.config import JIRA_CONFIG, DATA_DIR
from src.core.ai.llm_client import LLMClient
from src.core.storage.metadata_manager import UnifiedMetadataManager

logger = logging.getLogger(__name__)

class JiraFetcher:
    """Fetch and process Jira tickets."""
    
    def __init__(self, project_id: str):
        """Initialize the Jira fetcher using config settings.
        
        Args:
            project_id: Project identifier for organizing content
        """
        if not all([JIRA_CONFIG["url"], 
                   JIRA_CONFIG["username"], 
                   JIRA_CONFIG["password"]]):
            raise ValueError(
                "Jira configuration incomplete. Ensure JIRA_URL, "
                "JIRA_USERNAME, and JIRA_API_TOKEN environment "
                "variables are set."
            )
        
        self.project_id = project_id
        self.client = Jira(
            url=JIRA_CONFIG["url"],
            username=JIRA_CONFIG["username"],
            password=JIRA_CONFIG["password"],
            cloud=JIRA_CONFIG["cloud"]
        )
        self.llm_client = LLMClient()
        self.metadata_manager = UnifiedMetadataManager()
    
    def fetch_epic_with_issues(self, epic_key: str) -> Dict:
        """Fetch an epic and all its issues.
        
        Args:
            epic_key: Jira epic key
            
        Returns:
            Dict containing epic and issues data
        """
        try:
            # First fetch the epic itself with all fields
            epic = self.client.issue(epic_key, fields='*all')
            
            # Extract epic metadata
            epic_metadata = {
                "title": epic['fields']['summary'],
                "ticket_id": epic_key,
                "type": "epic",
                "status": epic['fields']['status']['name'],
                "created": epic['fields']['created'],
                "last_updated": epic['fields']['updated'],
                "priority": epic['fields'].get('priority', {}).get('name', 'None'),
                "assignee": epic['fields'].get('assignee', {}).get('displayName', 'Unassigned'),
                "reporter": epic['fields'].get('reporter', {}).get('displayName', 'Unknown'),
                "version": epic['fields'].get('customfield_10019', 0)  # Sprint/version field
            }
            
            # Check if epic content needs processing
            epic_needs_processing = self.metadata_manager.should_process_content(
                self.project_id, "jira", epic_key, epic_metadata
            )
            
            if not epic_needs_processing:
                logger.info(f"Epic {epic_key} unchanged, skipping processing")
                stored_metadata = self.metadata_manager.get_content_metadata(
                    self.project_id, "jira", epic_key
                )
                epic_content = stored_metadata["content"]
            else:
                # Process epic content
                epic_content = self._format_ticket_content(epic)
                epic_metadata["content"] = epic_content
                self.metadata_manager.update_content_metadata(
                    self.project_id, "jira", epic_key, epic_metadata
                )
            
            # Then fetch all issues that have this epic as parent
            jql = f'parent = {epic_key} ORDER BY created DESC'
            issues = self.client.jql(
                jql,
                fields=[
                    'summary',
                    'description',
                    'status',
                    'created',
                    'updated',
                    'issuetype',
                    'priority',
                    'assignee',
                    'reporter',
                    'comment',
                    'parent'
                ],
                expand=['changelog', 'renderedFields']
            )
            
            # Process issues
            issue_list = []
            for issue in issues['issues']:
                issue_metadata = {
                    "title": issue['fields']['summary'],
                    "ticket_id": issue['key'],
                    "type": issue['fields']['issuetype']['name'],
                    "status": issue['fields']['status']['name'],
                    "created": issue['fields']['created'],
                    "last_updated": issue['fields']['updated'],
                    "priority": issue['fields'].get('priority', {}).get('name', 'None'),
                    "assignee": issue['fields'].get('assignee', {}).get('displayName', 'Unassigned'),
                    "reporter": issue['fields'].get('reporter', {}).get('displayName', 'Unknown'),
                    "version": len(issue.get('changelog', {}).get('histories', []))  # Use changelog length as version
                }
                
                # Check if issue content needs processing
                if not self.metadata_manager.should_process_content(
                    self.project_id, "jira", issue['key'], issue_metadata
                ):
                    logger.info(f"Issue {issue['key']} unchanged, skipping processing")
                    stored_metadata = self.metadata_manager.get_content_metadata(
                        self.project_id, "jira", issue['key']
                    )
                    issue_content = stored_metadata["content"]
                else:
                    # Process issue content
                    issue_content = self._format_ticket_content(issue)
                    issue_metadata["content"] = issue_content
                    self.metadata_manager.update_content_metadata(
                        self.project_id, "jira", issue['key'], issue_metadata
                    )
                
                issue_list.append({
                    "content": issue_content,
                    "metadata": issue_metadata
                })
            
            return {
                "epic": {
                    "content": epic_content,
                    "metadata": epic_metadata
                },
                "issues": issue_list
            }
            
        except Exception as e:
            logger.error(f"Error fetching Jira epic {epic_key}: {e}")
            raise
    
    def _format_ticket_content(self, ticket_data: Dict) -> str:
        """Format ticket data as markdown content using LLM."""
        try:
            # Prepare ticket data for LLM
            content = []
            fields = ticket_data['fields']
            
            # Basic info
            content.append(f"# [{ticket_data['key']}] {fields['summary']}\n")
            content.append("## Metadata\n")
            content.append(f"- Type: {fields['issuetype']['name']}")
            content.append(f"- Status: {fields['status']['name']}")
            content.append(f"- Created: {fields['created']}")
            content.append(f"- Updated: {fields['updated']}")
            if 'priority' in fields and fields['priority']:
                content.append(f"- Priority: {fields['priority']['name']}")
            if 'assignee' in fields and fields['assignee']:
                content.append(f"- Assignee: {fields['assignee']['displayName']}")
            if 'reporter' in fields and fields['reporter']:
                content.append(f"- Reporter: {fields['reporter']['displayName']}")
            content.append("")
            
            # Description
            if fields.get('description'):
                content.append("## Description\n")
                content.append(fields['description'])
                content.append("\n")
            
            # Comments
            if fields.get('comment', {}).get('comments'):
                content.append("## Comments\n")
                for comment in fields['comment']['comments']:
                    content.append(f"### {comment['author']['displayName']} - {comment['created']}")
                    content.append(comment['body'])
                    content.append("\n")
            
            raw_content = "\n".join(content)
            
            # Use LLM to format content
            task_description = """
            Format this Jira ticket content as clean markdown:
            1. Preserve the ticket structure (metadata, description, comments)
            2. Format any code blocks or technical content properly
            3. Clean up any Jira-specific markup
            4. Ensure lists and tables are properly formatted
            5. Maintain all important information
            
            Return only the markdown content without any explanations or metadata.
            """
            
            return self.llm_client.execute_custom_task(raw_content, task_description)
            
        except Exception as e:
            logger.error(f"Error formatting ticket content: {e}")
            raise
    
    def save_epic_with_issues(self, epic_data: Dict) -> Dict:
        """Save epic and issues content and metadata.
        
        Args:
            epic_data: Dict containing epic and issues data
            
        Returns:
            Dict with file paths and metadata
        """
        try:
            # Save epic
            epic = epic_data["epic"]
            epic_path = self.metadata_manager.get_content_path(
                self.project_id,
                "jira",
                f"{epic['metadata']['ticket_id']}.md"
            )
            epic_path.write_text(epic["content"])
            
            # Save issues
            saved_issues = []
            for issue in epic_data["issues"]:
                issue_path = self.metadata_manager.get_content_path(
                    self.project_id,
                    "jira",
                    f"{issue['metadata']['ticket_id']}.md"
                )
                issue_path.write_text(issue["content"])
                saved_issues.append({
                    "title": issue["metadata"]["title"],
                    "path": str(issue_path.relative_to(DATA_DIR)),
                    "ticket_id": issue["metadata"]["ticket_id"],
                    "type": issue["metadata"]["type"],
                    "status": issue["metadata"]["status"],
                    "created": issue["metadata"]["created"],
                    "last_changed": issue["metadata"]["last_updated"],
                    "priority": issue["metadata"]["priority"],
                    "assignee": issue["metadata"]["assignee"],
                    "reporter": issue["metadata"]["reporter"]
                })
            
            # Return combined metadata
            return {
                "epic": {
                    "title": epic["metadata"]["title"],
                    "path": str(epic_path.relative_to(DATA_DIR)),
                    "ticket_id": epic["metadata"]["ticket_id"],
                    "type": epic["metadata"]["type"],
                    "status": epic["metadata"]["status"],
                    "created": epic["metadata"]["created"],
                    "last_changed": epic["metadata"]["last_updated"],
                    "priority": epic["metadata"]["priority"],
                    "assignee": epic["metadata"]["assignee"],
                    "reporter": epic["metadata"]["reporter"]
                },
                "issues": saved_issues
            }
            
        except Exception as e:
            logger.error(f"Error saving Jira data: {e}")
            raise

def fetch_jira_epics(
    project_config: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """CLI tool to fetch Jira epics and their issues.
    
    This script fetches epics and their issues from Jira and saves them as markdown files.
    It uses environment variables for authentication:
    - JIRA_URL: Base URL of your Jira instance
    - JIRA_USERNAME: Your username
    - JIRA_API_TOKEN: Your API token (for cloud) or password (for server)
    - JIRA_CLOUD: "true" for cloud, "false" for server (default: "true")
    
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
            if "jira" not in project:
                continue
            
            project_metadata = []
            
            # Initialize fetcher with project ID
            fetcher = JiraFetcher(project["key"])
            
            # Fetch each epic
            for epic in project["jira"]["epics"]:
                try:
                    epic_data = fetcher.fetch_epic_with_issues(epic["id"])
                    metadata = fetcher.save_epic_with_issues(epic_data)
                    project_metadata.append(metadata)
                    print(f"Fetched epic: {epic['id']} - {epic['title']}")
                except Exception as e:
                    logger.error(f"Error processing epic {epic['id']}: {e}")
                    continue
            
            # Save project metadata
            if project_metadata:
                print(f"\nSuccessfully processed {len(project_metadata)} epics for project {project['key']}")
                print(f"Content saved to: {DATA_DIR / project['key'] / 'jira'}")
                print(f"Metadata saved to: {DATA_DIR / project['key'] / 'project_metadata.json'}")
        
    except Exception as e:
        logger.error(f"Error fetching Jira epics: {e}")
        raise

if __name__ == "__main__":
    fire.Fire(fetch_jira_epics)
