"""Fetch and process Slack messages."""
import logging
from datetime import datetime
import time
import re
from typing import Dict, List, Optional
import fire
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pathlib import Path

from config.config import DATA_DIR, LLM_CONFIG, SLACK_CONFIG
from src.core.storage.metadata_manager import UnifiedMetadataManager
from src.core.utils.logging_config import setup_logging
from src.core.ai.llm_client import LLMClient

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logging()

class SlackFetcher:
    """Fetch and process Slack messages."""
    
    def __init__(self, project_id: str):
        """Initialize the Slack fetcher using config settings.
        
        Args:
            project_id: Project identifier for organizing content
        """
        logger.debug(f"Initializing SlackFetcher for project {project_id}")
        
        self.project_id = project_id
        self.metadata_manager = UnifiedMetadataManager()
        self.llm_client = LLMClient()
        
        # Load project config
        self.project_config = self.metadata_manager.get_project(project_id)
        
        # Build member mapping
        self.member_mapping = self._build_member_mapping()
        logger.info(f"SlackFetcher initialized for project {project_id}")
    
    def _build_member_mapping(self) -> Dict[str, str]:
        """Build mapping of usernames to full names from project config."""
        logger.debug("Building member mapping from project config")
        mapping = {}
        for member in self.project_config.get("members", []):
            name = member.get("name")
            if name:
                for username in member.get("user_names", []):
                    mapping[username.lower()] = name
        logger.debug(f"Built member mapping with {len(mapping)} entries")
        return mapping
    
    def _get_last_update_time(self, channel_id: str) -> datetime:
        """Get the last update time for a channel.
        
        Args:
            channel_id: Slack channel ID
            
        Returns:
            Datetime of last update or configured history days ago if no data
        """
        try:
            logger.debug(f"Getting last update time for channel {channel_id}")
            
            # Get all thread metadata for this channel
            channel_metadata = self.metadata_manager.get_content_metadata(
                self.project_id, "slack", channel_id
            )
            
            if not channel_metadata:
                # Default to configured history days ago if no data
                return datetime.now().timestamp() - (SLACK_CONFIG['history_days'] * 24 * 60 * 60)
            
            last_update = channel_metadata.get("last_updated")
            if not last_update:
                return datetime.now().timestamp() - (SLACK_CONFIG['history_days'] * 24 * 60 * 60)
                
            return float(last_update)
            
        except Exception as e:
            logger.error(f"Error getting last update time for channel {channel_id}: {e}")
            # Default to configured history days ago on error
            return datetime.now().timestamp() - (SLACK_CONFIG['history_days'] * 24 * 60 * 60)

    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic based on config settings.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result from the function
            
        Raises:
            SlackApiError: If all retries are exhausted
        """
        retries = 0
        while retries <= SLACK_CONFIG['max_retries']:
            try:
                return func(*args, **kwargs)
            except SlackApiError as e:
                retries += 1
                if retries > SLACK_CONFIG['max_retries']:
                    raise
                logger.warning(f"Slack API error (attempt {retries}/{SLACK_CONFIG['max_retries']}): {e}")
                time.sleep(SLACK_CONFIG['retry_delay'])
    
    def _generate_thread_title(self, messages: List[Dict]) -> str:
        """Generate a title for a thread using LLM.
        
        Args:
            messages: List of thread messages
            
        Returns:
            Generated title string
        """
        try:
            logger.debug("Generating thread title using LLM")
            
            # Prepare thread content for LLM
            thread_content = "\n".join([
                f"{self.member_mapping.get(msg.get('user', '').lower(), 'Unknown')}: {self._process_message_text(msg.get('text', ''))}"
                for msg in messages
            ])
            
            task_description = f"""You are a document title generator. Your task is to generate a concise, descriptive title for a Slack thread.

Project Context:
Project: {self.project_config['name']} ({self.project_config['key']})
Description: {self.project_config['description']}

Team Members Context:
{', '.join([f"{m['name']} ({m['role']})" for m in self.project_config.get('members', [])])}

Primary Objectives:
1. Generate a clear, informative title that captures the main topic and intent
2. Keep the title concise (3-8 words)
3. Make the title descriptive enough to understand the thread content
4. Use proper capitalization
5. Focus on the key decision, question, or outcome

Example Input:
2024-11-12 13:14:15 John Doe (Engineer):
Hey team, we need to decide on which cloud provider to use for our new data pipeline
2024-11-12 13:15:16 Mary Poppins (Architect):
I think AWS would be best given our existing infrastructure
2024-11-12 13:15:46 John Doe (Engineer): 
Agreed, their managed services would help us move faster
> :+1: 2 (Mary Poppins, Sarah Cooper)
2024-11-12 13:16:10 Sarah Cooper (PM): 
Let's go with AWS then, I'll update the project docs

Example Output:
AWS Selected as Cloud Provider for Data Pipeline

Content Format:
The content is a Slack thread with multiple messages, including member names and roles.

Requirements:
1. Focus on the main topic, decision, or outcome of the thread
2. Use professional, clear language
3. Be specific but concise
4. Capture the essence of the discussion

Return only the title without any additional text or formatting.
"""
            
            title = self.llm_client.execute_custom_task(
                thread_content,
                task_description
            )
            
            logger.debug(f"Generated title: {title}")
            return title.strip()
            
        except Exception as e:
            logger.error(f"Error generating thread title: {e}")
            # Fallback to timestamp-based title
            return f"Slack Thread {datetime.fromtimestamp(float(messages[0]['ts'])).strftime('%Y-%m-%d %H:%M')}"
    
    def _format_reactions(self, reactions: List[Dict]) -> str:
        """Format message reactions as markdown.
        
        Args:
            reactions: List of reaction dictionaries
            
        Returns:
            Formatted markdown string
        """
        if not reactions:
            return ""
            
        reaction_strs = []
        for reaction in reactions:
            emoji = reaction.get('name', '')
            count = reaction.get('count', 0)
            users = reaction.get('users', [])
            # Map user IDs to names
            user_names = [self.member_mapping.get(user.lower(), user) for user in users]
            reaction_strs.append(f":{emoji}: {count} ({', '.join(user_names)})")
            
        return "\n> " + "\n> ".join(reaction_strs)

    def _process_message_text(self, text: str) -> str:
        """Process message text to replace user mentions with real names.
        
        Args:
            text: Raw message text
            
        Returns:
            Processed text with user mentions replaced with real names
        """
        if not text:
            return ""
            
        def replace_mention(match):
            user_id = match.group(1)
            # Get the member's full name from mapping, or keep the user ID if not found
            return self.member_mapping.get(user_id.lower(), f"<@{user_id}>")
            
        # Replace <@U...> style mentions with the member's full name
        return re.sub(r'<@(U[A-Z0-9]+)>', replace_mention, text)
    
    def _format_thread_content(self, messages: List[Dict]) -> str:
        """Format thread messages as markdown content.
        
        Args:
            messages: List of thread messages
            
        Returns:
            Formatted markdown string
        """
        try:
            logger.debug("Formatting thread content as markdown")
            
            # Generate title
            title = self._generate_thread_title(messages)
            content = [f"# {title}\n"]
            
            # Format messages
            for msg in messages:
                # Convert timestamp to datetime
                timestamp = datetime.fromtimestamp(float(msg['ts'])).strftime('%Y-%m-%d %H:%M')
                
                # Get user's real name from mapping
                user_id = msg.get('user', 'UNKNOWN')
                username = self.member_mapping.get(user_id.lower(), user_id)
                
                # Add formatted message with processed text
                content.append(f"**{timestamp} {username}**")
                content.append(self._process_message_text(msg.get('text', '')))
                
                # Add reactions if present
                if msg.get('reactions'):
                    content.append(self._format_reactions(msg['reactions']))
                
                content.append("")  # Empty line between messages
            
            return "\n".join(content)
            
        except Exception as e:
            logger.error(f"Error formatting thread content: {e}")
            raise
    
    def _get_thread_data(self, client: WebClient, channel_id: str, thread_ts: str) -> Optional[Dict]:
        """Fetch thread data including messages and metadata.
        
        Args:
            client: WebClient instance
            channel_id: Channel ID
            thread_ts: Thread timestamp
            
        Returns:
            Dict containing thread data or None if error
        """
        try:
            logger.debug(f"Fetching thread data for {channel_id}:{thread_ts}")
            
            # Get thread replies with retry
            result = self._execute_with_retry(
                client.conversations_replies,
                channel=channel_id,
                ts=thread_ts
            )
            messages = result["messages"]

            # Get thread metadata with retry
            thread_info = self._execute_with_retry(
                client.conversations_info,
                channel=channel_id
            )

            # Get permalink with retry
            permalink_result = self._execute_with_retry(
                client.chat_getPermalink,
                channel=channel_id,
                message_ts=thread_ts
            )
            permalink = permalink_result["permalink"]

            thread_data = {
                "thread_ts": thread_ts,
                "messages": messages,
                "channel_name": thread_info["channel"]["name"],
                "updated_at": messages[-1]["ts"],
                "permalink": permalink
            }
            
            logger.debug(f"Successfully fetched thread data for {channel_id}:{thread_ts}")
            return thread_data
            
        except SlackApiError as e:
            logger.error(f"Error fetching thread data for {channel_id}:{thread_ts}: {e}")
            return None
    
    def save_thread(self, thread_data: Dict) -> Dict:
        """Save thread content and metadata.
        
        Args:
            thread_data: Dict containing thread data
            
        Returns:
            Dict with file paths and metadata
        """
        try:
            thread_ts = thread_data["thread_ts"]
            logger.info(f"Saving thread {thread_ts}")
            
            # Format content
            content = self._format_thread_content(thread_data["messages"])
            
            # Generate metadata
            metadata = {
                "thread_ts": thread_ts,
                "channel_id": thread_data["channel_id"],
                "channel_name": thread_data["channel_name"],
                "created_at": thread_data["messages"][0]["ts"],
                "updated_at": thread_data["messages"][-1]["ts"],
                "url": thread_data["permalink"],
                "message_count": len(thread_data["messages"]),
                "type": "slack_thread",
                "content_hash": hash(str(thread_data["messages"]))
            }
            
            # Get content path from metadata manager
            content_path = self.metadata_manager.get_content_path(
                self.project_id,
                "slack",
                thread_ts
            )
            
            # Save content
            content_path.write_text(content)
            logger.debug(f"Saved thread content to {content_path}")
            
            # Update metadata
            self.metadata_manager.update_content_metadata(
                self.project_id, "slack", thread_ts, metadata
            )
            
            return {
                "thread_ts": thread_ts,
                "path": str(content_path.relative_to(DATA_DIR)),
                "channel_id": metadata["channel_id"],
                "channel_name": metadata["channel_name"],
                "created_at": metadata["created_at"],
                "updated_at": metadata["updated_at"],
                "url": metadata["url"],
                "message_count": metadata["message_count"],
                "type": "slack_thread",
                "content_hash": metadata["content_hash"]
            }
            
        except Exception as e:
            logger.error(f"Error saving thread: {e}")
            raise

    def _is_public_channel(self, client: WebClient, channel_id: str) -> bool:
        """Check if a channel is public.
        
        Args:
            client: WebClient instance
            channel_id: Channel ID to check
            
        Returns:
            True if channel is public, False otherwise
        """
        try:
            # Get channel info
            channel_info = self._execute_with_retry(
                client.conversations_info,
                channel=channel_id
            )
            
            # Check if channel is private
            return not channel_info["channel"].get("is_private", True)
            
        except SlackApiError as e:
            logger.error(f"Error checking channel type for {channel_id}: {e}")
            return False

    def fetch_messages(self, channel_ids: List[str]):
        """Fetch messages from specified Slack channels.
        
        Args:
            channel_ids: List of channel IDs to fetch from
        """
        try:
            logger.info(f"Starting Slack message fetch for workspace: {SLACK_CONFIG['workspace']}")
            
            # Initialize client
            client = WebClient(token=SLACK_CONFIG['token'])
            
            for channel_id in channel_ids:
                try:
                    # Skip private channels since we don't have groups:history scope
                    if not self._is_public_channel(client, channel_id):
                        logger.info(f"Skipping private channel {channel_id} - missing groups:history scope")
                        continue
                        
                    logger.info(f"Processing channel: {channel_id}")
                    last_export = self._get_last_update_time(channel_id)
                    
                    threads_data = []
                    cursor = None
                    
                    while True:
                        # Get channel history with retry logic
                        result = self._execute_with_retry(
                            client.conversations_history,
                            channel=channel_id,
                            oldest=last_export,
                            cursor=cursor,
                            limit=SLACK_CONFIG['batch_size']
                        )
                        messages = result["messages"]
                        logger.info(f"Retrieved {len(messages)} messages")

                        # Process threaded messages
                        for message in messages:
                            if "thread_ts" in message:
                                thread_data = self._execute_with_retry(
                                    self._get_thread_data,
                                    client, channel_id, message["thread_ts"]
                                )
                                if thread_data:
                                    thread_data["channel_id"] = channel_id
                                    self.save_thread(thread_data)
                                else:
                                    logger.warning(
                                        f"Failed to fetch thread data for ts: {message['thread_ts']}"
                                    )

                        # Check for more messages
                        if "response_metadata" in result and "next_cursor" in result["response_metadata"]:
                            cursor = result["response_metadata"]["next_cursor"]
                        else:
                            break
                        
                    logger.info(f"Completed processing channel: {channel_id}")
                    
                except SlackApiError as e:
                    logger.error(f"Error processing channel {channel_id}: {e}")
                    continue
                    
            logger.info("Slack message fetch completed successfully")
            
        except Exception as e:
            logger.error(f"Error in Slack message fetch: {e}")
            raise

def fetch_slack_messages(
    project_config: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """CLI tool to fetch Slack messages.
    
    This script fetches messages from configured Slack channels and stores them as markdown files.
    Configuration is read from config.py SLACK_CONFIG:
    - token: Bot User OAuth Token
    - workspace: Workspace name
    - max_retries: Maximum number of API retries
    - retry_delay: Delay between retries in seconds
    - batch_size: Number of messages to fetch per batch
    - history_days: Number of days of history to fetch
    
    Args:
        project_config: Path to project config YAML
        output_dir: Optional directory to save content
    """
    try:
        logger.info("Starting Slack message fetch process")
        metadata_manager = UnifiedMetadataManager()
        
        # Get all projects from metadata manager
        config = metadata_manager._load_project_config()
        
        # Process each project
        for project in config.get("projects", []):
            if "slack" not in project:
                logger.debug(f"Skipping project {project.get('key', 'unknown')} - no Slack configuration")
                continue
                
            logger.info(f"Processing project: {project['key']}")
            
            # Initialize fetcher
            fetcher = SlackFetcher(project["key"])
            
            # Get Slack channels from config
            slack_channels = project["slack"]["channels"]  # Access the channels array correctly
            channel_ids = [channel["channel_id"] for channel in slack_channels]  # Use channel_id instead of channel
            
            try:
                fetcher.fetch_messages(channel_ids=channel_ids)
                logger.info(f"Successfully processed Slack messages for project {project['key']}")
                
            except Exception as e:
                logger.error(f"Error processing Slack messages for project {project['key']}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Failed to fetch Slack messages: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    fire.Fire(fetch_slack_messages)
