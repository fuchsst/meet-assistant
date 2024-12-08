"""LLM client for text analysis using CrewAI."""
import logging
from typing import Dict, Any, Optional
import json
from pathlib import Path
import time
from crewai import Agent, Task, Crew, Process
from crewai import LLM

from config.config import LLM_CONFIG

logger = logging.getLogger(__name__)

class LLMClient:
    """Handles LLM-based text analysis using CrewAI."""

    def __init__(self):
        """Initialize LLM client with agents."""
        self.llm = LLM(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
            top_p=LLM_CONFIG["top_p"],
            frequency_penalty=LLM_CONFIG["frequency_penalty"],
            presence_penalty=LLM_CONFIG["presence_penalty"]
        )
        # Initialize specialized agents
        self.agents = {
            "summarizer": Agent(
                role="Meeting Summarizer",
                goal="Create concise, accurate meeting summaries",
                backstory="Expert in distilling key information from conversations",
                llm=self.llm,
                verbose=True
            ),
            "task_extractor": Agent(
                role="Task Manager",
                goal="Identify and organize action items and responsibilities",
                backstory="Experienced project manager focused on task tracking",
                llm=self.llm,
                verbose=True
            ),
            "decision_tracker": Agent(
                role="Decision Analyst",
                goal="Track and document key decisions made during meetings",
                backstory="Strategic analyst specializing in decision documentation",
                llm=self.llm,
                verbose=True
            )
        }
        
    def analyze_text(self, text: str) -> str:
        """Analyze text and return structured insights using CrewAI agents."""
        try:
            # Create an analyst agent
            analyst = Agent(
                role="Meeting Analyst",
                goal="Analyze meeting transcripts and extract key information",
                backstory="""You are an expert meeting analyst with years of experience
                in extracting valuable insights from conversations. You excel at identifying
                main points, action items, and key decisions.""",
                llm=self.llm,
                verbose=True
            )

            # Create analysis task
            analysis_task = Task(
                description="""Analyze the provided meeting transcript and extract:
                1. A concise summary of the main points
                2. Action items or tasks that were assigned
                3. Key decisions that were made
                Format your response as a JSON object with these three keys.""",
                expected_output="A JSON object containing summary, action_items, and decisions",
                agent=analyst
            )

            # Create and run crew
            crew = Crew(
                agents=[analyst],
                tasks=[analysis_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute analysis
            result = crew.kickoff(inputs={"transcript": text})
            
            # Ensure the result is valid JSON
            try:
                if isinstance(result, str):
                    json.loads(result)
                    return result
                else:
                    # If result is already a dict/object, convert to JSON string
                    return json.dumps(result)
            except json.JSONDecodeError:
                # If the result isn't valid JSON, try to extract and format it
                logger.warning("LLM response was not valid JSON, attempting to format")
                formatted_result = {
                    "summary": str(result).split("Action items:")[0].strip(),
                    "action_items": [],
                    "decisions": []
                }
                return json.dumps(formatted_result)
                
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            raise

    def execute_custom_task(self, transcript: str, task_description: str) -> str:
        """Execute a custom analysis task on the transcript using CrewAI."""
        try:
            # Create a custom task agent
            custom_agent = Agent(
                role="Custom Task Analyst",
                goal="Execute specific analysis tasks on meeting transcripts",
                backstory="""You are a specialized analyst capable of performing
                custom analysis tasks on meeting transcripts. You adapt your approach
                based on the specific requirements of each task.""",
                llm=self.llm,
                verbose=True
            )

            # Create custom task
            custom_task = Task(
                description=task_description,
                expected_output="A detailed response addressing the task requirements in English.",
                agent=custom_agent
            )

            # Create and run crew
            crew = Crew(
                agents=[custom_agent],
                tasks=[custom_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute task
            result = crew.kickoff(inputs={"transcript": transcript})
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Custom task execution failed: {str(e)}")
            raise

    def analyze_transcript(self, transcript: str, max_retries: int = 3) -> Dict[str, Any]:
        """Analyze meeting transcript using multiple agents."""
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Create analysis tasks
                tasks = [
                    Task(
                        description="Generate a comprehensive meeting summary",
                        agent=self.agents["summarizer"],
                        context={"transcript": transcript}
                    ),
                    Task(
                        description="Extract and organize action items",
                        agent=self.agents["task_extractor"],
                        context={"transcript": transcript}
                    ),
                    Task(
                        description="Identify key decisions made",
                        agent=self.agents["decision_tracker"],
                        context={"transcript": transcript}
                    )
                ]

                # Create and run crew
                crew = Crew(
                    agents=list(self.agents.values()),
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=True
                )

                result = crew.kickoff()
                
                # Parse and validate results
                analysis = self._parse_results(result)
                if self._validate_analysis(analysis):
                    return analysis
                
                raise ValueError("Analysis validation failed")

            except Exception as e:
                logger.error(f"Analysis attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    raise RuntimeError(f"Analysis failed after {max_retries} attempts") from e

    def _parse_results(self, raw_results: str) -> Dict[str, Any]:
        """Parse and structure the analysis results."""
        try:
            # Parse the JSON string from analyze_text
            results = json.loads(raw_results)
            
            # Extract different components from the results
            analysis = {
                "summary": results.get("summary", "No summary available"),
                "action_items": results.get("action_items", []),
                "decisions": results.get("decisions", []),
                "metadata": {
                    "timestamp": time.time(),
                    "version": "1.0"
                }
            }
            return analysis
        except Exception as e:
            logger.error(f"Failed to parse analysis results: {str(e)}")
            raise

    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate the structure and content of the analysis."""
        try:
            required_keys = ["summary", "action_items", "decisions", "metadata"]
            if not all(key in analysis for key in required_keys):
                logger.error("Missing required keys in analysis")
                return False

            # Validate summary
            if not isinstance(analysis["summary"], str) or len(analysis["summary"]) < 10:
                logger.error("Invalid summary format or length")
                return False

            # Validate action items
            if not isinstance(analysis["action_items"], list):
                logger.error("Action items must be a list")
                return False

            # Validate decisions
            if not isinstance(analysis["decisions"], list):
                logger.error("Decisions must be a list")
                return False

            # Validate metadata
            metadata = analysis["metadata"]
            if not (isinstance(metadata, dict) and 
                   "timestamp" in metadata and 
                   "version" in metadata):
                logger.error("Invalid metadata format")
                return False

            return True

        except Exception as e:
            logger.error(f"Analysis validation failed: {str(e)}")
            return False

    def save_analysis(self, analysis: Dict[str, Any], filepath: Path) -> None:
        """Save analysis results to a file."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved analysis to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            raise
