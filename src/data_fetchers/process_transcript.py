"""Process and combine VTT transcripts with improvements using LLM."""
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime
from crewai import Crew, Process, Task
from tqdm import tqdm
import fire
from config.config import DATA_DIR, LLM_CONFIG, ANALYSIS_CONFIG
from src.core.ai.llm_client import LLMClient
from src.core.storage.metadata_manager import UnifiedMetadataManager
from src.core.utils.logging_config import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logging()

# Constants for optimization based on LLM config
MAX_CHUNK_CHARS = (LLM_CONFIG["max_tokens"] // 4) * 3  # Reserve 25% for response
MIN_SEGMENT_CHARS = 50  # Minimum size for independent processing
MAX_CONTEXT_CHARS = ANALYSIS_CONFIG["max_context_window"] // 4  # Characters for previous context

class VTTSegment:
    """Represents a segment from a VTT file."""
    
    def __init__(self, start: str, end: str, text: str, source: str = "", recording_num: int = 0):
        """Initialize VTT segment.
        
        Args:
            start: Start timestamp (HH:MM:SS.mmm)
            end: End timestamp (HH:MM:SS.mmm)
            text: Segment text content
            source: Source identifier (default: "")
            recording_num: Recording number (default: 0)
        """
        self.start = start
        self.end = end
        self.text = text
        self.source = source
        self.recording_num = recording_num
    
    def __lt__(self, other):
        """Enable sorting segments by start time."""
        return self._time_to_seconds(self.start) < self._time_to_seconds(other.start)
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert VTT timestamp to seconds.
        
        Args:
            time_str: Timestamp string (HH:MM:SS.mmm)
            
        Returns:
            Float representing seconds
        """
        h, m, s = time_str.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)

class TranscriptProcessor:
    """Process and combine VTT transcripts with LLM-based improvements."""
    
    def __init__(self, context_window: int = 5):
        """Initialize transcript processor.
        
        Args:
            context_window: Number of segments to use for context (default: 5)
        """
        self.context_window = context_window
        self.llm_client = LLMClient()
        self.metadata_manager = UnifiedMetadataManager()
        logger.info("TranscriptProcessor initialized successfully")
    
    def _read_vtt_file(self, file_path: Path) -> List[VTTSegment]:
        """Read and parse a VTT file.
        
        Args:
            file_path: Path to VTT file
            
        Returns:
            List of VTTSegment objects
        """
        segments = []
        current_times = None
        current_text = []
        recording_num = 0
        
        # Extract recording number from filename
        match = re.search(r"_(\d+)\.vtt$", file_path.name)
        if match:
            recording_num = int(match.group(1))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line == "WEBVTT":
                    continue
                
                if '-->' in line:
                    if current_times and current_text:
                        segments.append(VTTSegment(
                            current_times[0],
                            current_times[1],
                            ' '.join(current_text),
                            file_path.stem,
                            recording_num
                        ))
                    start, end = line.split(' --> ')
                    current_times = (start, end)
                    current_text = []
                else:
                    current_text.append(line)
            
            # Add final segment
            if current_times and current_text:
                segments.append(VTTSegment(
                    current_times[0],
                    current_times[1],
                    ' '.join(current_text),
                    file_path.stem,
                    recording_num
                ))
            
            return segments
            
        except Exception as e:
            logger.error(f"Error reading VTT file {file_path}: {e}")
            return []
    
    def _combine_segments(self, segments: List[VTTSegment]) -> List[List[VTTSegment]]:
        """Combine segments into optimal chunks for processing.
        
        This method combines segments while ensuring:
        1. Total characters stay under MAX_CHUNK_CHARS (based on LLM token limit)
        2. Large segments (>MIN_SEGMENT_CHARS) are processed independently
        3. Natural breaks in conversation are preserved
        4. Related segments stay together when possible
        5. Merge segments from different vtt files into time order
        
        Args:
            segments: List of VTT segments to combine
            
        Returns:
            List of segment chunks, where each chunk is a list of segments
        """
        chunks = []
        current_chunk = []
        current_chars = 0
        
        for i, segment in enumerate(segments):
            segment_chars = len(segment.text)
            
            # Handle large segments independently
            if segment_chars >= MIN_SEGMENT_CHARS and current_chars > 0:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [segment]
                current_chars = segment_chars
                continue
            
            # Check for natural breaks (different recordings or long pauses)
            natural_break = False
            if i > 0:
                prev_segment = segments[i-1]
                if (prev_segment.recording_num != segment.recording_num or
                    self._time_to_seconds(segment.start) - 
                    self._time_to_seconds(prev_segment.end) > 5):  # 5 second pause
                    natural_break = True
            
            # Start new chunk if:
            # 1. Adding segment would exceed limit, or
            # 2. Natural break and current chunk has content
            if ((current_chars + segment_chars > MAX_CHUNK_CHARS) or
                (natural_break and current_chunk)):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [segment]
                current_chars = segment_chars
            else:
                current_chunk.append(segment)
                current_chars += segment_chars
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _improve_segment_chunk(
        self,
        chunk: List[VTTSegment],
        project_metadata: Dict,
        meeting_metadata: Dict,
        improved_segments: List[str]
    ) -> List[str]:
        """Improve a chunk of segments using LLM.
        
        Args:
            chunk: List of segments to improve
            project_metadata: Project metadata for context
            meeting_metadata: Meeting metadata for context
            improved_segments: Previously improved segments for context
            
        Returns:
            List of improved segment texts
        """
        try:
            # Get relevant previous context while staying under token limit
            prev_context = []
            context_chars = 0
            for segment in reversed(improved_segments[-self.context_window:]):
                if context_chars + len(segment) > MAX_CONTEXT_CHARS:
                    break
                prev_context.insert(0, segment)
                context_chars += len(segment)
            
            # Prepare rich context from actual metadata structure
            context = f"""
Meeting Context:
- Title: {meeting_metadata.get('title', 'Unknown')}
- Date: {meeting_metadata.get('date', 'Unknown')}
- Language: {meeting_metadata.get('language', 'en')}
- Input Alias: {meeting_metadata.get('input_alias', 'Speaker')}
- Output Alias: {meeting_metadata.get('output_alias', 'Assistant')}

Previous Discussion:
{' '.join(prev_context) if prev_context else 'No previous context'}

Current Segment(s):
{' '.join(s.text for s in chunk)}
"""
            
            task_description = """
            Improve these transcript segments by:
            1. Making the text concise and clear while preserving:
               - Essential information and key points
               - Technical terms and proper nouns
               - Names of participants
            
            2. Improving clarity:
               - Remove filler words, repetitions, and unnecessary content
               - Keep only the most relevant information
               - Maintain natural but efficient speech patterns
            
            3. Ensuring consistency:
               - Match previous context terminology
               - Maintain speaker's technical level
               - Use consistent formatting
            
            Return only the improved text without any explanations or metadata.
            Focus on being concise while maintaining accuracy.
            """
            
            improved_text = self.llm_client.execute_custom_task(
                context,
                task_description
            )
            
            # Split improved text back into segments roughly matching original lengths
            total_original_chars = sum(len(s.text) for s in chunk)
            if total_original_chars == 0:
                return [""] * len(chunk)
            
            improved_segments = []
            start_idx = 0
            
            for segment in chunk:
                # Calculate proportional length for this segment
                segment_ratio = len(segment.text) / total_original_chars
                end_idx = min(
                    len(improved_text),
                    start_idx + int(len(improved_text) * segment_ratio)
                )
                
                # Find natural break point (sentence end or space)
                if end_idx < len(improved_text):
                    sentence_break = improved_text.rfind('. ', start_idx, end_idx)
                    space_break = improved_text.rfind(' ', start_idx, end_idx)
                    break_point = sentence_break if sentence_break != -1 else space_break
                    if break_point != -1:
                        end_idx = break_point + 1
                
                improved_segments.append(improved_text[start_idx:end_idx].strip())
                start_idx = end_idx
            
            # Handle any remaining text
            if start_idx < len(improved_text):
                improved_segments[-1] += " " + improved_text[start_idx:].strip()
            
            return improved_segments
            
        except Exception as e:
            logger.error(f"Error improving segment chunk: {e}")
            return [s.text for s in chunk]  # Return original text on error
    
    def process_meeting(self, project_id: Optional[str], meeting_id: str) -> None:
        """Process all VTT files in a meeting directory.
        
        Args:
            project_id: Optional project ID (uses default if not provided)
            meeting_id: Meeting ID to process
        """
        try:
            # Get project and meeting info
            project = self.metadata_manager.get_project(project_id)
            meeting_dir = self.metadata_manager.get_meeting_dir(project["key"], meeting_id)
            meeting_metadata = self.metadata_manager.get_meeting_metadata(project["key"], meeting_id)
            
            if not meeting_dir.exists():
                raise ValueError(f"Meeting directory not found: {meeting_dir}")
            
            logger.info(f"Processing meeting: {meeting_id}")
            
            # Read all VTT files
            segments = []
            for vtt_file in meeting_dir.glob("*.vtt"):
                segments.extend(self._read_vtt_file(vtt_file))
            
            if not segments:
                raise ValueError("No VTT files found in meeting directory")
            
            # Sort segments by time
            segments.sort()
            
            # Process segments in optimized chunks
            improved_segments = []
            segment_chunks = self._combine_segments(segments)
            
            for chunk in tqdm(segment_chunks, desc="Processing segments"):
                improved_texts = self._improve_segment_chunk(
                    chunk,
                    project,
                    meeting_metadata,
                    improved_segments
                )
                improved_segments.extend(improved_texts)
            
            # Create improved VTTSegments
            improved_vtt_segments = []
            for original, improved_text in zip(segments, improved_segments):
                improved_vtt_segments.append(VTTSegment(
                    original.start,
                    original.end,
                    improved_text,
                    original.source,
                    original.recording_num
                ))
            
            # Generate and save transcript in the specified format
            transcript_path = meeting_dir / "transcript.md"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write("# Meeting Transcript\n\n")
                f.write(f"Title: {meeting_metadata.get('title', 'Untitled Meeting')}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n\n")
                
                # Write segments in chronological order
                for segment in improved_vtt_segments:
                    f.write(f"[{segment.start}] {segment.text}\n")
            
            # Generate meeting analysis
            full_transcript = "\n".join(segment.text for segment in improved_vtt_segments)
            
            # Create analysis task with proper format
            analysis = self.llm_client.analyze_text(full_transcript)
            
            # Parse analysis results
            try:
                analysis_data = json.loads(analysis)
            except json.JSONDecodeError:
                logger.error("Failed to parse analysis results")
                analysis_data = {
                    "summary": "Analysis failed",
                    "action_items": [],
                    "decisions": []
                }
            
            # Generate and save analysis
            analysis_path = meeting_dir / "analysis.md"
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write("# Meeting Analysis\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if "summary" in analysis_data:
                    f.write("## Summary\n\n")
                    f.write(f"{analysis_data['summary']}\n\n")
                
                if "action_items" in analysis_data:
                    f.write("## Action Items\n\n")
                    for item in analysis_data["action_items"]:
                        f.write(f"- {item}\n")
                    f.write("\n")
                
                if "decisions" in analysis_data:
                    f.write("## Key Decisions\n\n")
                    for decision in analysis_data["decisions"]:
                        f.write(f"- {decision}\n")
                    f.write("\n")
            
            # Update metadata
            transcript_metadata = {
                "title": f"Meeting Transcript - {datetime.now().strftime('%Y-%m-%d')}",
                "path": str(transcript_path.relative_to(meeting_dir)),
                "type": "transcript",
                "description": "Full transcript of the meeting",
                "date": datetime.now().isoformat(),
                "content_hash": hash(full_transcript),
                "version": "1.0"
            }
            
            analysis_metadata = {
                "title": f"Meeting Analysis - {datetime.now().strftime('%Y-%m-%d')}",
                "path": str(analysis_path.relative_to(meeting_dir)),
                "type": "analysis",
                "description": analysis.get("summary", "Analysis of the meeting transcript"),
                "date": datetime.now().isoformat(),
                "content_hash": hash(str(analysis)),
                "version": "1.0"
            }
            
            self.metadata_manager.update_content_metadata(
                project["key"],
                "meetings",
                f"{meeting_id}_transcript",
                transcript_metadata
            )
            
            self.metadata_manager.update_content_metadata(
                project["key"],
                "meetings",
                f"{meeting_id}_analysis",
                analysis_metadata
            )
            
            self.metadata_manager.update_meeting_metadata(
                project["key"],
                meeting_id,
                {
                    "analysis_status": "completed",
                    "analysis_date": datetime.now().isoformat(),
                    "has_transcript": True,
                    "has_analysis": True,
                    "transcript_path": str(transcript_path.relative_to(meeting_dir)),
                    "analysis_path": str(analysis_path.relative_to(meeting_dir))
                }
            )
            
            logger.info(f"Processing complete for meeting: {meeting_id}")
            
        except Exception as e:
            logger.error(f"Failed to process meeting: {e}")
            raise

def main(
    meeting_id: str,
    project_id: Optional[str] = None
):
    """Process meeting transcripts with AI analysis.
    
    Args:
        meeting_id: ID of the meeting to process
        project_id: Optional project ID (uses default project if not provided)
    """
    try:
        print(f"Processing meeting: {meeting_id}")
        print(f"Project: {project_id or 'default'}")
        
        processor = TranscriptProcessor()
        processor.process_meeting(project_id, meeting_id)
        
        # Get meeting directory for output paths
        meeting_dir = processor.metadata_manager.get_meeting_dir(
            processor.metadata_manager.get_project(project_id)["key"],
            meeting_id
        )
        
        print(f"\nProcessing complete!")
        print(f"Transcript saved to: {meeting_dir / 'transcript.md'}")
        print(f"Analysis saved to: {meeting_dir / 'analysis.md'}")
        print(f"Metadata updated")
        
    except Exception as e:
        logger.error(f"Failed to process meeting: {e}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    fire.Fire(main)
