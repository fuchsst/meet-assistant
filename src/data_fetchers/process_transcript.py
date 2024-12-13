"""Process and combine VTT transcripts with improvements using LLM."""
import json
import sys
import logging
from pathlib import Path
import time
from typing import List, Dict, Optional, Tuple, Set
import re
from datetime import datetime
from tqdm import tqdm
import fire
from config.config import DATA_DIR, LLM_CONFIG, ANALYSIS_CONFIG
from src.core.ai.llm_client import LLMClient
from src.core.storage.metadata_manager import UnifiedMetadataManager
from src.core.utils.logging_config import setup_logging
from src.data_fetchers.prompts import (
    CLEAN_TRANSCRIPT_PROMPT,
    SELECT_DOCUMENTS_PROMPT,
    ANALYSIS_ROLES,
    ANALYSIS_PROMPTS
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logging()

# Constants for optimization based on LLM config
MAX_TOKENS = LLM_CONFIG["max_tokens"]
CHARS_PER_TOKEN = 4  # Approximate characters per token
MAX_CHUNK_CHARS = (MAX_TOKENS // 4) * 3  # Reserve 25% for response
MIN_SEGMENT_CHARS = 50  # Minimum size for independent processing
MAX_CONTEXT_CHARS = ANALYSIS_CONFIG["max_context_window"] // 4  # Characters for previous context

class TranscriptProcessor:
    """Process and combine VTT transcripts with LLM-based improvements."""
    
    def __init__(self):
        """Initialize transcript processor."""
        self.llm_client = LLMClient()
        self.metadata_manager = UnifiedMetadataManager()
        logger.debug("TranscriptProcessor initialized with LLM client and metadata manager")

    def _get_previous_meeting_data(self, project_id: str, meeting_id: str) -> Dict[str, str]:
        """Get summary and followup from the previous meeting."""
        try:
            # Get all meetings for the project
            project_meetings = self.metadata_manager.list_meetings(project_id)
            
            # Sort meetings by date
            sorted_meetings = sorted(
                [(m, self.metadata_manager.get_meeting_metadata(project_id, m)) 
                 for m in project_meetings],
                key=lambda x: x[1].get('start_time', ''),
                reverse=True
            )
            
            # Find the previous meeting
            current_found = False
            for meeting_tuple in sorted_meetings:
                if current_found:
                    prev_meeting_id = meeting_tuple[0]
                    prev_meeting_dir = self.metadata_manager.get_meeting_dir(project_id, prev_meeting_id)
                    
                    prev_data = {}
                    # Try to read summary and followup
                    for doc_type in ['summary', 'followup']:
                        try:
                            with open(prev_meeting_dir / f"{doc_type}.md", 'r', encoding='utf-8') as f:
                                prev_data[doc_type] = f.read()
                        except FileNotFoundError:
                            prev_data[doc_type] = f"No {doc_type} available from previous meeting"
                    
                    return prev_data
                
                if meeting_tuple[0] == meeting_id:
                    current_found = True
            
            return {
                "summary": "No previous meeting found",
                "followup": "No previous meeting found"
            }
            
        except Exception as e:
            logger.error(f"Error getting previous meeting data: {e}")
            return {
                "summary": f"Error retrieving previous meeting data: {str(e)}",
                "followup": f"Error retrieving previous meeting data: {str(e)}"
            }

    def _clean_transcript(self, transcript: str) -> str:
        """Clean and improve the transcript text."""
        logger.debug("Cleaning transcript text")
        
        # Split transcript into manageable chunks
        chunk_size = MAX_CHUNK_CHARS
        chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
        
        cleaned_chunks = []
        retry_count = 0
        max_retries = 3
        
        for chunk in chunks:
            while retry_count < max_retries:
                try:
                    # Create inputs for the task
                    inputs = {
                        "transcript_chunk": chunk,
                        "previous_chunks": cleaned_chunks[-2:] if cleaned_chunks else [],
                        "cleaning_parameters": {
                            "maintain_speaker_consistency": True,
                            "preserve_technical_terms": True,
                            "standardize_formatting": True
                        }
                    }
                    
                    cleaned_chunk = self.llm_client.execute_custom_task(
                        prompt=CLEAN_TRANSCRIPT_PROMPT.format(transcript_chunk=chunk),
                        task_description="Clean and improve the transcript text while maintaining context and formatting.",
                        additional_context=inputs
                    )
                    
                    # Validate the cleaned chunk
                    if not cleaned_chunk or len(cleaned_chunk.strip()) < MIN_SEGMENT_CHARS:
                        raise ValueError("Cleaned chunk is too short or empty")
                    
                    # Check for common transcription artifacts
                    if any(artifact in cleaned_chunk.lower() for artifact in ['[inaudible]', '[unclear]', '[silence]']):
                        logger.warning("Cleaned chunk contains transcription artifacts")
                        retry_count += 1
                        continue
                    
                    cleaned_chunks.append(cleaned_chunk)
                    break
                    
                except Exception as e:
                    logger.error(f"Chunk cleaning failed (attempt {retry_count + 1}): {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.warning(f"Using original chunk after {max_retries} failed cleaning attempts")
                        cleaned_chunks.append(chunk)
                    time.sleep(2 ** retry_count)  # Exponential backoff
        
        return "\n".join(cleaned_chunks)

    def _collect_vtt_files(self, project_id: str, meeting_id: str) -> List[Path]:
        """Collect all VTT files for a given meeting."""
        meeting_dir = self.metadata_manager.get_meeting_dir(project_id, meeting_id)
        logger.debug(f"Searching for VTT files in {meeting_dir}")
        vtt_files = list(meeting_dir.glob("*.vtt"))
        if not vtt_files:
            logger.error(f"No VTT files found in meeting directory: {meeting_dir}")
            raise ValueError(f"No VTT files found in meeting directory: {meeting_dir}")
        logger.info(f"Found {len(vtt_files)} VTT files")
        return vtt_files

    def _clean_vtt_content(self, vtt_path: Path) -> str:
        """Remove header, empty lines, and timestamps from VTT file."""
        logger.debug(f"Cleaning VTT content from {vtt_path}")
        clean_lines = []
        try:
            with open(vtt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    # Skip WEBVTT header, empty lines, and timestamp lines
                    if (not line or 
                        line == "WEBVTT" or 
                        '-->' in line or 
                        re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3}$', line)):
                        continue
                    clean_lines.append(line)
            logger.debug(f"Successfully cleaned {len(lines)} lines to {len(clean_lines)} content lines")
            return ' '.join(clean_lines)
        except Exception as e:
            logger.error(f"Failed to clean VTT content from {vtt_path}: {e}")
            raise

    def _get_recent_documents(self, project_id: str) -> List[Dict]:
        """Get most recent documents sorted by last_updated."""
        logger.debug(f"Retrieving recent documents for project {project_id}")
        project_metadata = self.metadata_manager._load_project_metadata(project_id)
        documents = []
        
        # Collect documents from all sources
        for source_type, docs in project_metadata.get("documents", {}).items():
            logger.debug(f"Processing {len(docs)} documents from {source_type}")
            for doc_id, metadata in docs.items():
                doc_info = {
                    "id": doc_id,
                    "title": metadata.get("title", "Untitled"),
                    "last_updated": metadata.get("last_updated", ""),
                    "source_type": source_type,
                    "status": metadata.get("status", ""),
                    "path": metadata.get("path", ""),
                    "url": metadata.get("url", "")
                }
                documents.append(doc_info)
        
        # Sort by last_updated and limit to 100
        documents.sort(key=lambda x: x["last_updated"], reverse=True)
        documents = documents[:100]
        logger.info(f"Retrieved {len(documents)} recent documents")
        return documents

    def _format_document_metadata(self, documents: List[Dict]) -> str:
        """Format document metadata as markdown."""
        logger.debug("Formatting document metadata as markdown")
        markdown = "# Available Documents\n\n"
        for doc in documents:
            markdown += f"## {doc['title']}\n"
            markdown += f"- File path: {doc['path']}\n"
            markdown += f"- Last Updated: {doc['last_updated']}\n"
            markdown += f"- Type: {doc['source_type']}\n"
            if doc['status']:
                markdown += f"- Status: {doc['status']}\n"
            if doc['url']:
                markdown += f"- URL: {doc['url']}\n"
            markdown += "\n"
        return markdown

    def _select_relevant_documents(
        self,
        transcript: str,
        doc_metadata: str,
        max_tokens: int,
        meeting_metadata: Dict
    ) -> List[str]:
        """Ask LLM to select relevant documents based on transcript."""
        logger.debug("Selecting relevant documents using LLM")
        
        # Calculate available tokens for transcript excerpt
        prompt_tokens = len(SELECT_DOCUMENTS_PROMPT) // CHARS_PER_TOKEN
        metadata_tokens = len(doc_metadata) // CHARS_PER_TOKEN
        available_tokens = max_tokens - prompt_tokens - metadata_tokens
        transcript_excerpt = transcript[:available_tokens * CHARS_PER_TOKEN]
        
        # Create inputs for document selection
        inputs = {
            "transcript_excerpt": transcript_excerpt,
            "doc_metadata": doc_metadata,
            "meeting_metadata": meeting_metadata,
            "selection_criteria": {
                "relevance_threshold": 0.7,
                "max_historical_days": 90,
                "prioritize_technical_docs": True
            }
        }
        
        response = self.llm_client.execute_custom_task(
            prompt=SELECT_DOCUMENTS_PROMPT.format(
                transcript_excerpt=transcript_excerpt,
                doc_metadata=doc_metadata
            ),
            task_description="Select relevant documents based on the meeting transcript and context.",
            additional_context=inputs
        )
        
        try:
            # Validate and clean the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("Response does not contain a JSON array")
            
            json_str = response[json_start:json_end]
            selected_docs = json.loads(json_str)
            
            # Validate the structure
            if not isinstance(selected_docs, list) or not all(isinstance(x, str) for x in selected_docs):
                raise ValueError("Invalid response format - expected list of strings")
            
            logger.info(f"Selected {len(selected_docs)} relevant documents")
            return selected_docs
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse document selection response: {e}")
            logger.debug(f"Raw response: {response}")
            return []

    def _read_document_content(
        self,
        project_id: str,
        doc_files: List[str],
        documents: List[Dict]
    ) -> Dict[str, str]:
        """Read content of selected documents."""
        logger.debug(f"Reading content of {len(doc_files)} selected documents")
        content_map = {}
        total_chars = 0
        max_chars = (MAX_TOKENS - 1000) * CHARS_PER_TOKEN  # Reserve 1000 tokens for system prompt
        
        for doc in documents:
            if doc['path'] in doc_files:
                try:
                    doc_path = Path(doc['path'])
                    if not doc_path.is_absolute():
                        doc_path = self.metadata_manager.data_dir / doc_path
                    
                    logger.debug(f"Reading document: {doc_path}")
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content_chars = len(content)
                        
                        # Check if adding this document would exceed token limit
                        if total_chars + content_chars > max_chars:
                            logger.warning(f"Skipping {doc['title']}: would exceed token limit")
                            continue
                        
                        content_map[doc['title']] = content
                        total_chars += content_chars
                        logger.debug(f"Added document {doc['title']} ({content_chars} chars)")
                        
                except Exception as e:
                    logger.error(f"Error reading document {doc['title']}: {e}")
                    continue
        
        logger.info(f"Successfully read {len(content_map)} documents")
        return content_map

    def _generate_analysis(
        self,
        transcript: str,
        doc_contents: Dict[str, str],
        analysis_type: str,
        meeting_metadata: Dict,
        previous_analyses: Optional[Dict[str, str]] = None,
        previous_meeting_data: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate specific type of meeting analysis."""
        logger.debug(f"Generating {analysis_type} analysis")
        
        # Prepare context from relevant documents
        doc_context = "\n\n".join([
            f"# {title}\n{content}"
            for title, content in doc_contents.items()
        ])
        
        # Add meeting metadata context
        metadata_context = f"""
# Meeting Information
- Title: {meeting_metadata.get('title', 'Untitled Meeting')}
- Date: {meeting_metadata.get('date', 'Unknown')}
- Participants: {', '.join(meeting_metadata.get('participants', []))}
- Project: {meeting_metadata.get('project_description', 'No description available')}
"""
        
        # Add previous meeting context if available
        previous_meeting_context = ""
        if previous_meeting_data:
            previous_meeting_context = f"""
# Previous Meeting Context
## Summary
{previous_meeting_data.get('summary', 'No previous summary available')}

## Follow-up Items
{previous_meeting_data.get('followup', 'No previous follow-up items available')}
"""
        
        # Create inputs for analysis
        inputs = {
            "transcript": transcript,
            "doc_context": doc_context,
            "metadata_context": metadata_context,
            "previous_meeting_context": previous_meeting_context,
            "meeting_metadata": meeting_metadata,
            "document_contents": doc_contents,
            "previous_analyses": previous_analyses or {},
            "previous_meeting_data": previous_meeting_data or {},
            "analysis_parameters": {
                "type": analysis_type,
                "maintain_context": True,
                "consider_related_docs": True,
                "ensure_consistency": True,
                "reference_documents": True
            }
        }
        
        # Generate analysis with retries
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                return self.llm_client.analyze_transcript(
                    transcript=transcript,
                    analysis_type=analysis_type,
                    role=ANALYSIS_ROLES[analysis_type],
                    prompt=ANALYSIS_PROMPTS[analysis_type],
                    additional_context=inputs
                )
                
            except Exception as e:
                logger.error(f"Analysis generation attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    raise RuntimeError(f"Analysis generation failed after {max_retries} attempts") from e

    def process_meeting(
        self,
        project_id: Optional[str],
        meeting_id: str,
        analyses: Optional[Set[str]] = None,
        force: bool = False
    ) -> None:
        """Process meeting transcript and generate analyses."""
        try:
            # Set default analyses if none specified
            if analyses is None:
                analyses = {"summary", "minutes", "questions", "tasks", "followup"}
            
            # Get project info
            project = self.metadata_manager.get_project(project_id)
            project_id = project["key"]
            
            logger.info(f"Starting processing for meeting {meeting_id} in project {project_id}")
            
            # Get meeting metadata
            meeting_metadata = self.metadata_manager.get_meeting_metadata(project_id, meeting_id)
            
            # Get previous meeting data
            previous_meeting_data = self._get_previous_meeting_data(project_id, meeting_id)
            
            # Get meeting directory and transcript path
            meeting_dir = self.metadata_manager.get_meeting_dir(project_id, meeting_id)
            transcript_path = meeting_dir / "transcript.md"
            
            # Process transcript
            if transcript_path.exists() and not force:
                logger.info("Using existing transcript.md")
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    cleaned_transcript = f.read()
            else:
                # Process VTT files
                logger.info("Processing VTT files")
                vtt_files = self._collect_vtt_files(project_id, meeting_id)
                transcript = ""
                for vtt_file in vtt_files:
                    transcript += self._clean_vtt_content(vtt_file) + "\n"
                
                # Clean and improve transcript
                cleaned_transcript = self._clean_transcript(transcript)
                
                # Save cleaned transcript
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_transcript)
                
                # Update meeting metadata with transcript info
                self.metadata_manager.update_meeting_metadata(
                    project_id,
                    meeting_id,
                    {
                        "transcript": {
                            "path": str(transcript_path.relative_to(meeting_dir)),
                            "last_updated": datetime.now().isoformat(),
                            "content_hash": hash(cleaned_transcript)
                        }
                    }
                )
            
            # Get recent documents and their metadata
            recent_docs = self._get_recent_documents(project_id)
            doc_metadata = self._format_document_metadata(recent_docs)
            
            # Select relevant documents with meeting context
            relevant_doc_paths = self._select_relevant_documents(
                cleaned_transcript,
                doc_metadata,
                MAX_TOKENS,
                meeting_metadata  # Pass meeting metadata for context
            )
            
            # Read content of relevant documents
            doc_contents = self._read_document_content(
                project_id,
                relevant_doc_paths,
                recent_docs
            )
            
            # Initialize analysis results in metadata
            analysis_results = {}
            previous_analyses = {}  # Store completed analyses for context
            
            # Generate requested analyses
            for analysis_type in analyses:
                try:
                    output_path = meeting_dir / f"{analysis_type}.md"
                    
                    # Skip if file exists and force is False
                    if output_path.exists() and not force:
                        logger.info(f"Skipping {analysis_type} analysis - file exists (use --force to overwrite)")
                        # Load existing analysis for context
                        with open(output_path, 'r', encoding='utf-8') as f:
                            previous_analyses[analysis_type] = f.read()
                        continue
                    
                    logger.info(f"Generating {analysis_type} analysis")
                    analysis_content = self._generate_analysis(
                        cleaned_transcript,
                        doc_contents,
                        analysis_type,
                        meeting_metadata,
                        previous_analyses,  # Pass previous analyses for context
                        previous_meeting_data  # Pass previous meeting data for context
                    )
                    
                    # Save analysis file
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(analysis_content)
                    
                    # Store analysis metadata and content for context
                    previous_analyses[analysis_type] = analysis_content
                    analysis_results[analysis_type] = {
                        "title": f"Meeting {analysis_type} - {meeting_id}",
                        "path": str(output_path.relative_to(meeting_dir)),
                        "type": analysis_type,
                        "date": datetime.now().isoformat(),
                        "content_hash": hash(analysis_content),
                        "status": "completed"
                    }
                    
                    logger.debug(f"Saved {analysis_type} analysis to {output_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {analysis_type} analysis: {e}")
                    analysis_results[analysis_type] = {
                        "type": analysis_type,
                        "date": datetime.now().isoformat(),
                        "status": "failed",
                        "error": str(e)
                    }
                    # Don't continue with more analyses if one fails
                    raise RuntimeError(f"Analysis generation failed for {analysis_type}") from e
            
            # Update meeting metadata with all results
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "analysis_status": "completed",
                    "analysis_date": datetime.now().isoformat(),
                    "analyses": analysis_results,
                    "relevant_documents": relevant_doc_paths
                }
            )
            
            logger.info(f"Successfully completed all analyses for meeting {meeting_id}")
            
        except Exception as e:
            logger.error(f"Failed to process meeting: {e}", exc_info=True)
            # Update metadata to reflect failure
            self.metadata_manager.update_meeting_metadata(
                project_id,
                meeting_id,
                {
                    "analysis_status": "failed",
                    "analysis_date": datetime.now().isoformat(),
                    "analysis_error": str(e)
                }
            )
            raise

def main(
    meeting_id: str,
    project_id: Optional[str] = None,
    analyses: str = "all",
    force: bool = False
):
    """Process meeting transcripts with AI analysis.
    
    Args:
        meeting_id: ID of the meeting to process
        project_id: Optional project ID (uses default if not provided)
        analyses: Comma-separated list of analyses to generate
                 (summary,minutes,questions,tasks,followup) or 'all'
        force: If True, overwrite existing analysis files instead of skipping
    """
    try:
        # Parse requested analyses
        available_analyses = {"summary", "minutes", "questions", "tasks", "followup"}
        if analyses.lower() == "all":
            requested_analyses = available_analyses
        else:
            requested_analyses = {
                a.strip() for a in analyses.split(",")
                if a.strip() in available_analyses
            }
        
        if not requested_analyses:
            logger.error("No valid analyses requested")
            print("No valid analyses requested. Available options:")
            print(", ".join(available_analyses))
            sys.exit(1)
        
        logger.info(f"Starting transcript processing for meeting {meeting_id}")
        logger.info(f"Project: {project_id or 'default'}")
        logger.info(f"Requested analyses: {', '.join(requested_analyses)}")
        
        processor = TranscriptProcessor()
        processor.process_meeting(project_id, meeting_id, requested_analyses, force)
        
    except Exception as e:
        logger.error(f"Failed to process meeting: {e}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    fire.Fire(main)
