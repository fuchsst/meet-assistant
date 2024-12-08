# src/tools/document_tools.py
from pathlib import Path
from typing import Dict, List, Optional
import json
from crewai.tools import BaseTool
from crewai.llm import LLM

from config.config import LLM_CONFIG

class DocumentSelector(BaseTool):
    """Tool for selecting relevant documents based on metadata and task context."""
    
    name = "document_selector"
    description = """
    Select relevant documents based on metadata and task context.
    Input should be a JSON string with:
    - task_type: Type of task being performed (e.g., "summarize", "plan_tasks")
    - task_description: Description of the task
    Returns a list of relevant document paths.
    """
    
    def __init__(self, project_metadata: Dict, meeting_metadata: Dict):
        """Initialize with project and meeting metadata."""
        super().__init__()
        self.project_metadata = project_metadata
        self.meeting_metadata = meeting_metadata
        self.llm = LLM(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
            top_p=LLM_CONFIG["top_p"],
            frequency_penalty=LLM_CONFIG["frequency_penalty"],
            presence_penalty=LLM_CONFIG["presence_penalty"]
        )
        
    def _create_selection_prompt(self, task_type: str, task_description: str) -> str:
        """Create a prompt for the LLM to select relevant documents."""
        return f"""Given a task and available documents, select the most relevant files for the task.

Available Documents:
1. Meeting Transcript (transcript.md):
   - Contains the full meeting discussion
   - Title: {self.meeting_metadata.get('title', 'Unknown')}
   - Description: {self.meeting_metadata.get('description', 'No description')}

2. Meeting Analysis (analysis.md):
   - Contains AI-generated analysis of the meeting
   - Includes summary, key points, and action items

3. Project Context Documents:
{self._format_context_docs()}

Task Information:
- Type: {task_type}
- Description: {task_description}

Example Selections:
1. For "summarize" tasks:
   {
     "files": [
       {"path": "transcript.md", "reason": "Primary source for meeting content"},
       {"path": "analysis.md", "reason": "Contains existing analysis and key points"}
     ]
   }

2. For "plan_tasks" tasks:
   {
     "files": [
       {"path": "transcript.md", "reason": "Source of discussed tasks"},
       {"path": "analysis.md", "reason": "Contains identified action items"},
       {"path": "project_key/jira/PROJ-123.md", "reason": "Related epic with existing tasks"}
     ]
   }

Select the most relevant files for this specific task and explain why each is needed.
Return your selection in the same JSON format as the examples.
"""

    def _format_context_docs(self) -> str:
        """Format context documents for the prompt."""
        context_str = []
        
        # Get project key from metadata
        project_key = self.project_metadata.get("project_key", "")
        
        # Format each document type
        for source_type, items in self.meeting_metadata.get("context", {}).items():
            for item in items:
                path = f"{project_key}/{source_type}/{Path(item['path']).name}"
                context_str.append(f"   - {path}:")
                context_str.append(f"     Title: {item['title']}")
                context_str.append(f"     Type: {source_type}")
                if "description" in item:
                    context_str.append(f"     Description: {item['description']}")
                if "status" in item:
                    context_str.append(f"     Status: {item['status']}")
                if "last_changed" in item:
                    context_str.append(f"     Last Updated: {item['last_changed']}")
        return "\n".join(context_str)

    def _run(self, tool_input: str) -> str:
        try:
            # Parse input
            params = json.loads(tool_input)
            task_type = params.get("task_type", "")
            task_description = params.get("task_description", "")
            
            # Create selection prompt
            prompt = self._create_selection_prompt(task_type, task_description)
            
            # Get LLM selection
            messages = [
                {"role": "system", "content": "You are a document selection assistant."},
                {"role": "user", "content": prompt}
            ]
            
            result = self.llm.call(messages)
            
            # Parse and validate result
            try:
                selection = json.loads(result)
                if not isinstance(selection, dict) or "files" not in selection:
                    raise ValueError("Invalid selection format")
                return json.dumps(selection, indent=2)
            except json.JSONDecodeError:
                return json.dumps({
                    "error": "Failed to parse LLM response",
                    "files": []
                })
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "files": []
            })

class DocumentCombiner(BaseTool):
    """Tool for combining content from selected documents."""
    
    name = "document_combiner"
    description = """
    Combine content from selected documents into a single context.
    Input should be a JSON string with:
    - files: List of document paths to combine
    Returns the combined content with document metadata.
    """
    
    def __init__(self, project_metadata: Dict, meeting_metadata: Dict):
        """Initialize with project and meeting metadata."""
        super().__init__()
        self.project_metadata = project_metadata
        self.meeting_metadata = meeting_metadata
        self.base_path = Path(self.meeting_metadata.get("path", "")).parent
    
    def _run(self, tool_input: str) -> str:
        try:
            # Parse input
            params = json.loads(tool_input)
            files = params.get("files", [])
            
            if not files:
                return json.dumps({"error": "No files provided"})
            
            # Get project key from metadata
            project_key = self.project_metadata.get("project_key", "")
            
            # Combine document contents
            combined_content = []
            for file_info in files:
                path = self.base_path / file_info["path"]
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        combined_content.append(f"{'-'*40}\n{path.name}\n\n{content}")
            
            return "\n\n".join(combined_content)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
