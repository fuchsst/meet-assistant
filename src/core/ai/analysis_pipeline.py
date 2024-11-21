"""Meeting analysis pipeline orchestration."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.storage.file_manager import FileManager
from src.core.transcription.whisper_processor import WhisperProcessor
from src.core.ai.llm_client import LLMClient

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Orchestrates the meeting analysis process."""

    def __init__(self):
        """Initialize pipeline components."""
        self.file_manager = FileManager()
        self.transcriber = WhisperProcessor()
        self.llm_client = LLMClient()

    def process_meeting(self, meeting_id: str) -> Dict[str, Any]:
        """Process a complete meeting recording."""
        try:
            logger.info(f"Starting analysis pipeline for meeting {meeting_id}")
            
            # Get meeting files
            files = self.file_manager.get_meeting_files(meeting_id)
            
            # Verify audio file exists
            if not files["audio"].exists():
                raise FileNotFoundError(f"Audio file not found for meeting {meeting_id}")
            
            # Get or create transcript
            if not files["transcript"].exists():
                logger.info("Generating transcript from audio")
                segments = self.transcriber.transcribe_file(files["audio"])
                self._save_transcript(meeting_id, segments)
            
            # Read transcript
            transcript = self._read_transcript(files["transcript"])
            
            # Generate analysis
            analysis = self.llm_client.analyze_transcript(transcript)
            
            # Save analysis results
            self.file_manager.save_analysis(meeting_id, analysis)
            
            # Update meeting metadata
            self._update_metadata(meeting_id, analysis)
            
            logger.info(f"Completed analysis pipeline for meeting {meeting_id}")
            return analysis

        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            raise

    def process_segment(self, meeting_id: str, audio_segment: bytes) -> Dict[str, Any]:
        """Process a single audio segment in real-time."""
        try:
            # Transcribe segment
            text = self.transcriber.transcribe_chunk(audio_segment)
            if not text:
                return {}

            # Save transcript chunk
            self.file_manager.save_transcript(meeting_id, text, mode='a')
            
            # Get complete transcript for context
            files = self.file_manager.get_meeting_files(meeting_id)
            full_transcript = self._read_transcript(files["transcript"])
            
            # Generate incremental analysis
            analysis = self.llm_client.analyze_transcript(full_transcript)
            
            # Save updated analysis
            self.file_manager.save_analysis(meeting_id, analysis)
            
            return {
                "transcript": text,
                "analysis": analysis
            }

        except Exception as e:
            logger.error(f"Segment processing failed: {str(e)}")
            raise

    def execute_custom_task(self, meeting_id: str, task_description: str) -> Dict[str, Any]:
        """Execute a custom analysis task on the meeting transcript."""
        try:
            logger.info(f"Executing custom task for meeting {meeting_id}")
            
            # Get meeting files
            files = self.file_manager.get_meeting_files(meeting_id)
            
            # Read transcript
            transcript = self._read_transcript(files["transcript"])
            
            # Execute custom task using LLM
            result = self.llm_client.execute_custom_task(transcript, task_description)
            
            # Save result in analysis file with task info
            analysis = {
                "custom_task": {
                    "description": task_description,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Update the analysis file
            if files["analysis"].exists():
                with open(files["analysis"], 'r', encoding='utf-8') as f:
                    existing_analysis = json.load(f)
                
                # Add or update custom tasks section
                if "custom_tasks" not in existing_analysis:
                    existing_analysis["custom_tasks"] = []
                existing_analysis["custom_tasks"].append(analysis["custom_task"])
                
                self.file_manager.save_analysis(meeting_id, existing_analysis)
            else:
                self.file_manager.save_analysis(meeting_id, {"custom_tasks": [analysis["custom_task"]]})
            
            logger.info(f"Completed custom task for meeting {meeting_id}")
            return analysis

        except Exception as e:
            logger.error(f"Custom task execution failed: {str(e)}")
            raise

    def generate_summary(self, meeting_id: str) -> Dict[str, Any]:
        """Generate a meeting summary."""
        try:
            # Get meeting files
            files = self.file_manager.get_meeting_files(meeting_id)
            
            # Read transcript
            transcript = self._read_transcript(files["transcript"])
            
            # Generate summary
            analysis = self.llm_client.analyze_transcript(transcript)
            
            return {
                "summary": analysis["summary"],
                "key_points": self._extract_key_points(analysis),
                "duration": self._calculate_duration(transcript)
            }

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            raise

    def extract_tasks(self, meeting_id: str) -> list:
        """Extract action items and tasks."""
        try:
            # Get existing analysis
            files = self.file_manager.get_meeting_files(meeting_id)
            
            if not files["analysis"].exists():
                # Generate new analysis if none exists
                return self.process_meeting(meeting_id)["action_items"]
            
            # Read existing analysis
            with open(files["analysis"], 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            return analysis.get("action_items", [])

        except Exception as e:
            logger.error(f"Task extraction failed: {str(e)}")
            raise

    def identify_decisions(self, meeting_id: str) -> list:
        """Extract key decisions made during the meeting."""
        try:
            # Get existing analysis
            files = self.file_manager.get_meeting_files(meeting_id)
            
            if not files["analysis"].exists():
                # Generate new analysis if none exists
                return self.process_meeting(meeting_id)["decisions"]
            
            # Read existing analysis
            with open(files["analysis"], 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            return analysis.get("decisions", [])

        except Exception as e:
            logger.error(f"Decision identification failed: {str(e)}")
            raise

    def _save_transcript(self, meeting_id: str, segments: list) -> None:
        """Save transcript segments to file."""
        try:
            transcript_text = ""
            for segment in segments:
                transcript_text += f"[{segment['start']}] {segment['text']}\n"
            
            self.file_manager.save_transcript(meeting_id, transcript_text, mode='w')
            
        except Exception as e:
            logger.error(f"Failed to save transcript: {str(e)}")
            raise

    def _read_transcript(self, transcript_file: Path) -> str:
        """Read transcript from file."""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read transcript: {str(e)}")
            raise

    def _update_metadata(self, meeting_id: str, analysis: Dict[str, Any]) -> None:
        """Update meeting metadata with analysis results."""
        try:
            metadata = self.file_manager.get_metadata(meeting_id)
            
            metadata.update({
                "last_analyzed": datetime.now().isoformat(),
                "summary_length": len(analysis["summary"]),
                "action_items_count": len(analysis["action_items"]),
                "decisions_count": len(analysis["decisions"])
            })
            
            self.file_manager.save_metadata(meeting_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")
            raise

    def _extract_key_points(self, analysis: Dict[str, Any]) -> list:
        """Extract key points from analysis."""
        # This could be enhanced with more sophisticated extraction logic
        return [
            point.strip()
            for point in analysis["summary"].split('.')
            if point.strip()
        ]

    def _calculate_duration(self, transcript: str) -> int:
        """Calculate meeting duration from transcript timestamps."""
        try:
            # Extract timestamps and find the last one
            import re
            timestamps = re.findall(r'\[(\d{2}:\d{2}:\d{2})\]', transcript)
            if not timestamps:
                return 0
            
            # Convert last timestamp to seconds
            last_time = timestamps[-1]
            h, m, s = map(int, last_time.split(':'))
            return h * 3600 + m * 60 + s
            
        except Exception:
            logger.warning("Failed to calculate duration, returning 0")
            return 0
