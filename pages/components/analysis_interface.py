"""Analysis interface component for Meeting Assistant."""
import logging
import json
import tempfile
import pyperclip
from datetime import datetime
from pathlib import Path
import streamlit as st
from typing import Optional, Dict, Any

from src.core.ai.analysis_pipeline import AnalysisPipeline
from src.core.storage.file_manager import FileManager
from src.core.audio.voice_generator import VoiceGenerator
from config.config import VOICE_CONFIG

logger = logging.getLogger(__name__)

class AnalysisInterface:
    """Handles the analysis interface and controls."""

    def __init__(self, analysis_pipeline: AnalysisPipeline, file_manager: FileManager):
        """Initialize analysis interface."""
        self.analysis_pipeline = analysis_pipeline
        self.file_manager = file_manager
        # Initialize with selected voice from session state if available
        selected_voice = getattr(st.session_state, 'selected_voice', None)
        self.voice_generator = VoiceGenerator(voice_id=selected_voice)

    def _update_voice_generator(self) -> None:
        """Update voice generator with current selected voice."""
        if hasattr(st.session_state, 'selected_voice'):
            if (not hasattr(self, '_current_voice') or 
                self._current_voice != st.session_state.selected_voice):
                self.voice_generator.set_voice(st.session_state.selected_voice)
                self._current_voice = st.session_state.selected_voice

    def render_analysis_controls(self, meeting_id: str) -> None:
        """Render analysis control buttons and results."""
        st.caption("Analysis Controls")

        # Analysis buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Generate Minutes", type="primary"):
                try:
                    with st.spinner("Generating meeting minutes..."):
                        summary = self.analysis_pipeline.generate_summary(meeting_id)
                    self._display_summary(summary)
                except Exception as e:
                    logger.error(f"Failed to generate minutes: {str(e)}")
                    st.error(f"Failed to generate minutes: {str(e)}")
        
        with col2:
            if st.button("Extract Tasks", type="primary"):
                try:
                    with st.spinner("Extracting tasks..."):
                        tasks = self.analysis_pipeline.extract_tasks(meeting_id)
                    self._display_tasks(tasks)
                except Exception as e:
                    logger.error(f"Failed to extract tasks: {str(e)}")
                    st.error(f"Failed to extract tasks: {str(e)}")
        
        with col3:
            if st.button("Identify Decisions", type="primary"):
                try:
                    with st.spinner("Identifying decisions..."):
                        decisions = self.analysis_pipeline.identify_decisions(meeting_id)
                    self._display_decisions(decisions)
                except Exception as e:
                    logger.error(f"Failed to identify decisions: {str(e)}")
                    st.error(f"Failed to identify decisions: {str(e)}")

    def render_custom_task(self, meeting_id: str) -> None:
        """Render custom task input and execution."""
        st.caption("Custom Analysis Task")

        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Text", "Microphone"],
            horizontal=True
        )

        if input_method == "Text":
            task = st.text_area(
                "Enter your analysis task",
                placeholder="e.g., Summarize the key points about project timeline"
            )
            submit_button = st.button("Run Analysis", type="primary")
        else:
            # Microphone input
            task = self._handle_voice_input()
            submit_button = task is not None

        if submit_button and task:
            try:
                with st.spinner("Processing custom task..."):
                    result = self._process_custom_task(meeting_id, task)
                self._display_custom_result(result)
            except Exception as e:
                logger.error(f"Failed to process custom task: {str(e)}")
                st.error(f"Failed to process custom task: {str(e)}")

    def _display_summary(self, summary: Dict[str, Any]) -> None:
        """Display meeting summary with export options."""
        st.subheader("Meeting Summary")
        
        # Display summary components
        st.markdown("### Key Points")
        for point in summary.get("key_points", []):
            st.markdown(f"- {point}")

        st.markdown("### Summary")
        st.write(summary.get("summary", ""))

        if "duration" in summary:
            st.metric("Duration", f"{summary['duration']} seconds")

        # Export options
        self._add_export_options(
            "summary",
            self._format_summary_markdown(summary)
        )

    def _display_tasks(self, tasks: list) -> None:
        """Display extracted tasks with export options."""
        st.subheader("Action Items")
        
        for task in tasks:
            with st.expander(task.get("task", "Unnamed Task")):
                st.write(f"**Assignee:** {task.get('assignee', 'Unassigned')}")
                st.write(f"**Due Date:** {task.get('due_date', 'No date')}")
                st.write(f"**Status:** {task.get('status', 'Pending')}")
                if "context" in task:
                    st.write(f"**Context:** {task['context']}")

        # Export options
        self._add_export_options(
            "tasks",
            self._format_tasks_markdown(tasks)
        )

    def _display_decisions(self, decisions: list) -> None:
        """Display identified decisions with export options."""
        st.subheader("Key Decisions")
        
        for decision in decisions:
            st.markdown(f"- {decision}")

        # Export options
        self._add_export_options(
            "decisions",
            self._format_decisions_markdown(decisions)
        )

    def _display_custom_result(self, result: Dict[str, Any]) -> None:
        """Display custom analysis result with export options."""
        st.subheader("Custom Analysis Result")
        
        result_text = result.get("custom_task", {}).get("result", "")
        st.write(result_text)
        
        # Update voice generator with current selected voice
        self._update_voice_generator()
        
        # Add read response button
        if st.button("Read Response", key="read_response"):
            try:
                with st.spinner("Generating audio response..."):
                    # Generate speech to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = Path(temp_file.name)
                        self.voice_generator.generate_speech(result_text, temp_path)
                        # Read the audio file and display it
                        audio_bytes = temp_path.read_bytes()
                        st.audio(audio_bytes, format='audio/wav')
                        # Clean up
                        temp_path.unlink()
            except Exception as e:
                logger.error(f"Failed to generate speech: {str(e)}")
                st.error(f"Failed to generate speech: {str(e)}")
        
        # Export options
        self._add_export_options(
            "custom_analysis",
            self._format_custom_markdown(result)
        )

    def _add_export_options(self, prefix: str, markdown_content: str) -> None:
        """Add export options for analysis results."""
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Copy to Clipboard", key=f"copy_{prefix}"):
                try:
                    pyperclip.copy(markdown_content)
                    st.success("Copied to clipboard!")
                except Exception as e:
                    st.error(f"Failed to copy: {str(e)}")
        
        with col2:
            if st.button(f"Download Markdown", key=f"download_{prefix}"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{prefix}_{timestamp}.md"
                    st.download_button(
                        label="Click to Download",
                        data=markdown_content,
                        file_name=filename,
                        mime="text/markdown",
                        key=f"download_button_{prefix}"
                    )
                except Exception as e:
                    st.error(f"Failed to prepare download: {str(e)}")

    def _handle_voice_input(self) -> Optional[str]:
        """Handle voice input for custom task."""
        # This is a placeholder for voice input functionality
        # Will be implemented with F5-TTS integration
        st.warning("Voice input will be available soon!")
        return None

    def _process_custom_task(self, meeting_id: str, task: str) -> Dict[str, Any]:
        """Process a custom analysis task."""
        try:
            # Execute custom task using analysis pipeline
            result = self.analysis_pipeline.execute_custom_task(meeting_id, task)
            return result

        except Exception as e:
            logger.error(f"Custom task processing failed: {str(e)}")
            raise

    def _format_summary_markdown(self, summary: Dict[str, Any]) -> str:
        """Format summary as markdown."""
        lines = [
            "# Meeting Summary\n",
            "## Key Points\n"
        ]
        
        for point in summary.get("key_points", []):
            lines.append(f"- {point}\n")
        
        lines.extend([
            "\n## Summary\n",
            summary.get("summary", ""),
            f"\nDuration: {summary.get('duration', 0)} seconds"
        ])
        
        return "\n".join(lines)

    def _format_tasks_markdown(self, tasks: list) -> str:
        """Format tasks as markdown."""
        lines = ["# Action Items\n"]
        
        for task in tasks:
            lines.extend([
                f"## {task.get('task', 'Unnamed Task')}\n",
                f"- **Assignee:** {task.get('assignee', 'Unassigned')}",
                f"- **Due Date:** {task.get('due_date', 'No date')}",
                f"- **Status:** {task.get('status', 'Pending')}"
            ])
            if "context" in task:
                lines.append(f"- **Context:** {task['context']}")
            lines.append("\n")
        
        return "\n".join(lines)

    def _format_decisions_markdown(self, decisions: list) -> str:
        """Format decisions as markdown."""
        lines = ["# Key Decisions\n"]
        
        for decision in decisions:
            lines.append(f"- {decision}")
        
        return "\n".join(lines)

    def _format_custom_markdown(self, result: Dict[str, Any]) -> str:
        """Format custom analysis result as markdown."""
        custom_task = result.get("custom_task", {})
        lines = [
            "# Custom Analysis Result\n",
            "## Task Description",
            custom_task.get("description", "No task description"),
            "\n## Analysis Result",
            custom_task.get("result", "No result available"),
            "\n## Analysis Details",
            f"- Generated: {custom_task.get('timestamp', datetime.now().isoformat())}"
        ]
        
        return "\n".join(lines)
