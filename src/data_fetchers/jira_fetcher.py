"""Fetch and process Jira tickets."""
import logging
from pathlib import Path
import json
import yaml
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import fire
from atlassian import Jira
from config.config import JIRA_CONFIG, DATA_DIR
from src.core.ai.llm_client import LLMClient
from src.core.storage.metadata_manager import UnifiedMetadataManager
from src.core.utils.logging_config import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logging()

class JiraFetcher:
    """Fetch and process Jira tickets."""
    
    def __init__(self, project_id: str):
        """Initialize the Jira fetcher using config settings.
        
        Args:
            project_id: Project identifier for organizing content
        """
        logger.debug(f"Initializing JiraFetcher for project {project_id}")
        
        if not all([JIRA_CONFIG["url"], 
                   JIRA_CONFIG["username"], 
                   JIRA_CONFIG["password"]]):
            logger.error("Jira configuration incomplete - missing required credentials")
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
        logger.debug("Successfully initialized Jira client")
        
        self.llm_client = LLMClient()
        self.metadata_manager = UnifiedMetadataManager()
        
        # Load project config and build member mapping
        self.project_config = self.metadata_manager.get_project(project_id)
        self.member_mapping = self._build_member_mapping()
        logger.info(f"JiraFetcher initialized for project {project_id} with {len(self.member_mapping)} member mappings")
    
    def _build_member_mapping(self) -> Dict[str, str]:
        """Build mapping of usernames/accountIds to full names from project config.
        
        Returns:
            Dict mapping various user identifiers to their full names
        """
        logger.debug("Building member mapping from project config")
        mapping = {}
        for member in self.project_config.get("members", []):
            name = member.get("name")
            if name:
                # Map all usernames to the member's full name
                for username in member.get("user_names", []):
                    mapping[username.lower()] = name
        logger.debug(f"Built member mapping with {len(mapping)} entries")
        return mapping
    
    def _get_last_update_time(self, epic_key: str) -> Optional[str]:
        """Get the last update time for an epic and its issues.
        
        Args:
            epic_key: Jira epic key
            
        Returns:
            ISO formatted datetime string of last update or None if no data
        """
        try:
            logger.debug(f"Getting last update time for epic {epic_key}")
            # Get epic metadata
            epic_metadata = self.metadata_manager.get_content_metadata(
                self.project_id, "jira", epic_key
            )
            if not epic_metadata:
                logger.debug(f"No existing metadata found for epic {epic_key}")
                return None
                
            last_update = epic_metadata.get("last_updated")
            
            # Get all issue metadata for this epic
            project_metadata = self.metadata_manager._load_project_metadata(self.project_id)
            jira_docs = project_metadata.get("documents", {}).get("jira", {})
            
            # Find all issues belonging to this epic
            for doc_id, metadata in jira_docs.items():
                if metadata.get("parent_epic") == epic_key:
                    issue_updated = metadata.get("last_updated")
                    if issue_updated and (not last_update or issue_updated > last_update):
                        last_update = issue_updated
            
            logger.debug(f"Last update time for epic {epic_key}: {last_update}")
            return last_update
            
        except Exception as e:
            logger.error(f"Error getting last update time for epic {epic_key}: {e}")
            return None
    
    def _replace_user_mentions(self, text: str) -> str:
        """Replace Jira account ID mentions with full names.
        
        Handles Jira's wiki syntax [~ACCOUNTID] format, replacing account IDs
        with the corresponding member's full name from the project config.
        
        Args:
            text: Text containing Jira user mentions
            
        Returns:
            Text with user mentions replaced by full names
        """
        if not text:
            return text
            
        def replace_match(match):
            account_id = match.group(1)
            return self.member_mapping.get(account_id.lower(), f"@{account_id}")
        
        # Replace [~ACCOUNTID] pattern
        logger.debug("Replacing user mentions in text")
        return re.sub(r'\[~([^\]]+)\]', replace_match, text)
    
    def _clean_jira_formatting(self, text: str) -> str:
        """Clean up Jira-specific wiki formatting."""
        if not text:
            return ""
            
        logger.debug("Cleaning Jira wiki formatting")
        # Replace user mentions first
        text = self._replace_user_mentions(text)
            
        # Clean up code blocks
        text = re.sub(r'\{code(?::\w+)?\}', '```', text)
        
        # Clean up links
        # Replace [Text|http://url] with [Text](http://url)
        text = re.sub(r'\[([^\|]+)\|([^\]]+)\]', r'[\1](\2)', text)
        
        # Handle Jira issue links [KEY-123]
        text = re.sub(r'\[([A-Z]+-\d+)\]', r'[\1](issues/\1)', text)
        
        # Clean up lists
        text = text.replace("{noformat}", "```")
        text = re.sub(r'^\s*\*\s', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*#\s', '1. ', text, flags=re.MULTILINE)
        
        # Clean up tables
        text = re.sub(r'\|\|', '|', text)  # Convert Jira table headers
        
        # Clean up text formatting
        text = re.sub(r'\{color:[^}]+\}(.*?)\{color\}', r'**\1**', text)  # Convert colored text to bold
        text = re.sub(r'\{quote\}(.*?)\{quote\}', r'> \1', text, flags=re.DOTALL)  # Convert quotes
        
        logger.debug("Completed Jira formatting cleanup")
        return text
    
    def _format_ticket_content(self, ticket_data: Dict) -> str:
        """Format ticket data as markdown content."""
        try:
            logger.debug(f"Formatting ticket content for {ticket_data['key']}")
            content = []
            fields = ticket_data['fields']
            
            # Title and Key
            content.append(f"# [{ticket_data['key']}] {fields['summary']}\n")
            
            # Metadata Section
            content.append("## Metadata\n")
            
            # Get user names from account IDs
            assignee_id = fields.get('assignee', {}).get('accountId', '')
            reporter_id = fields.get('reporter', {}).get('accountId', '')
            
            metadata_items = [
                ("Type", fields['issuetype']['name']),
                ("Status", fields['status']['name']),
                ("Created", fields['created']),
                ("Updated", fields['updated']),
                ("Priority", fields.get('priority', {}).get('name', 'None')),
                ("Assignee", self.member_mapping.get(assignee_id.lower(), 
                                                   fields.get('assignee', {}).get('displayName', 'Unassigned'))),
                ("Reporter", self.member_mapping.get(reporter_id.lower(),
                                                   fields.get('reporter', {}).get('displayName', 'Unknown')))
            ]
            
            # Add optional metadata
            if fields.get('labels'):
                metadata_items.append(("Labels", ", ".join(fields['labels'])))
            if fields.get('components'):
                metadata_items.append(("Components", ", ".join([c['name'] for c in fields['components']])))
            if fields.get('fixVersions'):
                metadata_items.append(("Fix Versions", ", ".join([v['name'] for v in fields['fixVersions']])))
            if fields.get('customfield_10011'):  # Epic Name
                metadata_items.append(("Epic Name", fields['customfield_10011']))
            
            # Format metadata as table
            content.append("| Field | Value |")
            content.append("|-------|-------|")
            for field, value in metadata_items:
                if value:  # Only add non-empty values
                    content.append(f"| {field} | {value} |")
            content.append("")
            
            # Description Section
            if fields.get('description'):
                content.append("## Description\n")
                description = self._clean_jira_formatting(fields['description'])
                content.append(description)
                content.append("")
            
            # Comments Section
            if fields.get('comment', {}).get('comments'):
                content.append("## Comments\n")
                for comment in fields['comment']['comments']:
                    author_id = comment['author'].get('accountId', '')
                    author_name = self.member_mapping.get(
                        author_id.lower(),
                        comment['author'].get('displayName', author_id)
                    )
                    content.append(f"### {author_name} - {comment['created']}")
                    comment_body = self._clean_jira_formatting(comment['body'])
                    content.append(comment_body)
                    content.append("")
            
            # Changelog/History (if requested)
            if 'changelog' in ticket_data and ticket_data['changelog'].get('histories'):
                content.append("## Change History\n")
                for history in ticket_data['changelog']['histories']:
                    author_id = history['author'].get('accountId', '')
                    author_name = self.member_mapping.get(
                        author_id.lower(),
                        history['author'].get('displayName', author_id)
                    )
                    content.append(f"### {author_name} - {history['created']}")
                    for item in history['items']:
                        # Handle user field changes
                        from_value = item.get('fromString', 'empty')
                        to_value = item.get('toString', 'empty')
                        if item['field'].lower() in ['assignee', 'reporter']:
                            from_value = self.member_mapping.get(from_value.lower(), from_value)
                            to_value = self.member_mapping.get(to_value.lower(), to_value)
                        content.append(f"- Changed {item['field']} from '{from_value}' to '{to_value}'")
                    content.append("")
            
            logger.debug(f"Successfully formatted ticket content for {ticket_data['key']}")
            return "\n".join(content)
            
        except Exception as e:
            logger.error(f"Error formatting ticket content: {e}")
            raise
    
    def fetch_epic_with_issues(self, epic_key: str) -> Dict:
        """Fetch an epic and all its issues.
        
        Args:
            epic_key: Jira epic key
            
        Returns:
            Dict containing epic and issues data
        """
        try:
            logger.info(f"Fetching epic {epic_key} and its issues")
            # Get last update time
            last_update = self._get_last_update_time(epic_key)
            
            # First fetch the epic itself with all fields
            logger.debug(f"Fetching epic {epic_key} details")
            epic = self.client.issue(epic_key, fields='*all')
            
            # Extract epic metadata
            assignee_id = epic['fields'].get('assignee', {}).get('accountId', '')
            reporter_id = epic['fields'].get('reporter', {}).get('accountId', '')
            
            epic_metadata = {
                "title": epic['fields']['summary'],
                "ticket_id": epic_key,
                "type": "epic",
                "status": epic['fields']['status']['name'],
                "created": epic['fields']['created'],
                "last_updated": epic['fields']['updated'],
                "priority": epic['fields'].get('priority', {}).get('name', 'None'),
                "assignee": self.member_mapping.get(assignee_id.lower(), 
                                                  epic['fields'].get('assignee', {}).get('displayName', 'Unassigned')),
                "reporter": self.member_mapping.get(reporter_id.lower(),
                                                  epic['fields'].get('reporter', {}).get('displayName', 'Unknown')),
                "labels": epic['fields'].get('labels', []),
                "components": [c['name'] for c in epic['fields'].get('components', [])],
                "epic_name": epic['fields'].get('customfield_10011', ''),  # Epic Name field
                "content_hash": hash(str(epic['fields']))
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
                logger.info(f"Processing epic {epic_key} content")
                # Process epic content
                epic_content = self._format_ticket_content(epic)
                epic_metadata["content"] = epic_content
                self.metadata_manager.update_content_metadata(
                    self.project_id, "jira", epic_key, epic_metadata
                )
            
            # Build JQL for fetching updated issues
            jql_parts = [f'parent = {epic_key}']
            if last_update:
                # Add 1 minute buffer to avoid missing updates
                update_time = (
                    datetime.fromisoformat(last_update.replace('Z', '+00:00')) - 
                    timedelta(minutes=1)
                ).strftime('%Y-%m-%d %H:%M')
                jql_parts.append(f'updated >= "{update_time}"')
            
            jql = ' AND '.join(jql_parts) + ' ORDER BY created DESC'
            logger.debug(f"Fetching issues with JQL: {jql}")
            
            # Fetch issues
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
                    'parent',
                    'labels',
                    'components'
                ]
            )
            
            # Process issues
            issue_list = []
            logger.info(f"Processing {len(issues['issues'])} issues for epic {epic_key}")
            for issue in issues['issues']:
                logger.debug(f"Processing issue {issue['key']}")
                assignee_id = issue['fields'].get('assignee', {}).get('accountId', '')
                reporter_id = issue['fields'].get('reporter', {}).get('accountId', '')
                
                issue_metadata = {
                    "title": issue['fields']['summary'],
                    "ticket_id": issue['key'],
                    "type": issue['fields']['issuetype']['name'],
                    "status": issue['fields']['status']['name'],
                    "created": issue['fields']['created'],
                    "last_updated": issue['fields']['updated'],
                    "priority": issue['fields'].get('priority', {}).get('name', 'None'),
                    "assignee": self.member_mapping.get(assignee_id.lower(),
                                                      issue['fields'].get('assignee', {}).get('displayName', 'Unassigned')),
                    "reporter": self.member_mapping.get(reporter_id.lower(),
                                                      issue['fields'].get('reporter', {}).get('displayName', 'Unknown')),
                    "version": len(issue.get('changelog', {}).get('histories', [])),
                    "labels": issue['fields'].get('labels', []),
                    "components": [c['name'] for c in issue['fields'].get('components', [])],
                    "parent_epic": epic_key,
                    "content_hash": hash(str(issue['fields']))
                }
                
                # Check if issue content needs processing
                if not self.metadata_manager.should_process_content(
                    self.project_id, "jira", issue['key'], issue_metadata
                ):
                    logger.debug(f"Issue {issue['key']} unchanged, skipping processing")
                    stored_metadata = self.metadata_manager.get_content_metadata(
                        self.project_id, "jira", issue['key']
                    )
                    issue_content = stored_metadata["content"]
                else:
                    logger.debug(f"Processing content for issue {issue['key']}")
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
            
            logger.info(f"Successfully fetched and processed epic {epic_key} with {len(issue_list)} issues")
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
    
    def save_epic_with_issues(self, epic_data: Dict) -> Dict:
        """Save epic and issues content and metadata.
        
        Args:
            epic_data: Dict containing epic and issues data
            
        Returns:
            Dict with file paths and metadata
        """
        try:
            logger.info("Saving epic and issues content")
            # Save epic
            epic = epic_data["epic"]
            epic_path = self.metadata_manager.get_content_path(
                self.project_id,
                "jira",
                f"{epic['metadata']['ticket_id']}"
            )
            epic_path.write_text(epic["content"])
            logger.debug(f"Saved epic content to {epic_path}")
            
            # Save issues
            saved_issues = []
            logger.info(f"Saving {len(epic_data['issues'])} issues")
            for issue in epic_data["issues"]:
                issue_path = self.metadata_manager.get_content_path(
                    self.project_id,
                    "jira",
                    f"{issue['metadata']['ticket_id']}"
                )
                issue_path.write_text(issue["content"])
                logger.debug(f"Saved issue content to {issue_path}")
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
                    "reporter": issue["metadata"]["reporter"],
                    "labels": issue["metadata"].get("labels", []),
                    "components": issue["metadata"].get("components", []),
                    "parent_epic": issue["metadata"]["parent_epic"],
                    "content_hash": issue["metadata"]["content_hash"]
                })
            
            # Return combined metadata
            result = {
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
                    "reporter": epic["metadata"]["reporter"],
                    "labels": epic["metadata"].get("labels", []),
                    "components": epic["metadata"].get("components", []),
                    "epic_name": epic["metadata"].get("epic_name"),
                    "content_hash": epic["metadata"]["content_hash"]
                },
                "issues": saved_issues
            }
            
            logger.info("Successfully saved all epic and issue content")
            return result
            
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
        logger.info("Starting Jira epic fetch process")
        # Load project config
        if project_config:
            config_path = Path(project_config)
        else:
            config_path = DATA_DIR / "projects.yaml"
        
        if not config_path.exists():
            logger.error(f"Project config not found at {config_path}")
            raise ValueError(f"Project config not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Process each project
        for project in config.get("projects", []):
            if "jira" not in project:
                logger.debug(f"Skipping project {project.get('key', 'unknown')} - no Jira configuration")
                continue
            
            project_metadata = []
            logger.info(f"Processing project: {project['key']}")
            
            # Initialize fetcher with project ID
            fetcher = JiraFetcher(project["key"])
            
            # Fetch each epic
            for epic in project["jira"]["epics"]:
                try:
                    logger.info(f"Processing epic: {epic['id']} - {epic['title']}")
                    epic_data = fetcher.fetch_epic_with_issues(epic["id"])
                    metadata = fetcher.save_epic_with_issues(epic_data)
                    project_metadata.append(metadata)
                    logger.info(f"Successfully processed epic: {epic['id']} - {epic['title']}")
                except Exception as e:
                    logger.error(f"Error processing epic {epic['id']}: {e}")
                    continue
            
            # Save project metadata
            if project_metadata:
                logger.info(f"Successfully processed {len(project_metadata)} epics for project {project['key']}")
                logger.info(f"Content saved to: {DATA_DIR / project['key'] / 'jira'}")
                logger.info(f"Metadata saved to: {DATA_DIR / project['key'] / 'project_metadata.json'}")
        
    except Exception as e:
        logger.error(f"Failed to fetch Jira epics: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    fire.Fire(fetch_jira_epics)
