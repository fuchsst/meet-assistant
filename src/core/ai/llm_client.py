"""LLM client for text analysis using CrewAI."""
import logging
from typing import Dict, Any, Optional, List
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
        # Initialize specialized agents with more detailed context
        self.agents = {
            "summarizer": Agent(
                role="Meeting Summarizer",
                goal="Create concise, accurate meeting summaries with context awareness",
                backstory="""Expert in distilling key information from conversations while maintaining 
                context from related documents and previous discussions. Skilled at identifying themes,
                patterns and connecting information across different sources.""",
                llm=self.llm,
                verbose=True
            ),
            "task_extractor": Agent(
                role="Task Manager",
                goal="Identify and organize action items and responsibilities with full context",
                backstory="""Experienced project manager focused on task tracking and delegation.
                Specializes in understanding task dependencies, priorities, and connecting tasks
                to broader project context and documentation.""",
                llm=self.llm,
                verbose=True
            ),
            "decision_tracker": Agent(
                role="Decision Analyst",
                goal="Track and document key decisions made during meetings with contextual understanding",
                backstory="""Strategic analyst specializing in decision documentation and impact analysis.
                Expert at connecting decisions to previous context, related documents, and understanding
                implications for future actions.""",
                llm=self.llm,
                verbose=True
            )
        }
        
    def analyze_text(self, text: str, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """Analyze text and return structured insights using CrewAI agents."""
        try:
            # Create an analyst agent with enhanced context awareness
            analyst = Agent(
                role="Meeting Analyst",
                goal="Analyze meeting transcripts and extract key information with full context",
                backstory="""You are an expert meeting analyst with years of experience
                in extracting valuable insights from conversations while considering broader
                context from related documents and previous discussions. You excel at identifying
                main points, action items, and key decisions while maintaining contextual relevance.""",
                llm=self.llm,
                verbose=True
            )

            # Create analysis task with embedded context
            analysis_task = Task(
                description=f"""Analyze the following meeting transcript considering all available context.
                Extract and structure the following information:
                1. A concise summary of the main points, incorporating relevant context
                2. Action items or tasks that were assigned, including ownership and deadlines
                3. Key decisions that were made, including rationale and implications
                
                <transcript>
                {text}
                </transcript>
                
                <context>
                {json.dumps(additional_context, indent=2) if additional_context else ""}
                </context>
                
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
            result = crew.kickoff()
            
            # Get task output
            if hasattr(result, 'tasks_output') and result.tasks_output:
                task_output = result.tasks_output[0].raw
            else:
                task_output = result.raw if hasattr(result, 'raw') else str(result)
            
            # Ensure the result is valid JSON
            try:
                if isinstance(task_output, str):
                    json.loads(task_output)
                    return task_output
                else:
                    # If result is already a dict/object, convert to JSON string
                    return json.dumps(task_output)
            except json.JSONDecodeError:
                logger.warning("LLM response was not valid JSON, attempting to format")
                formatted_result = {
                    "summary": str(task_output).split("Action items:")[0].strip(),
                    "action_items": [],
                    "decisions": []
                }
                return json.dumps(formatted_result)
                
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            raise

    def execute_custom_task(self, prompt: str, task_description: str, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """Execute a custom analysis task on the transcript using CrewAI."""
        try:
            # Create a custom task agent with enhanced context handling
            custom_agent = Agent(
                role="Custom Task Analyst",
                goal="Execute specific analysis tasks on meeting transcripts with full context awareness",
                backstory="""You are a specialized analyst capable of performing
                custom analysis tasks on meeting transcripts while maintaining awareness
                of broader context from related documents and previous discussions. You
                adapt your approach based on the specific requirements of each task while
                ensuring contextual relevance.""",
                llm=self.llm,
                verbose=True
            )

            # Format context based on task type
            context_str = ""
            if additional_context:
                if "transcript_chunk" in additional_context:
                    # Format for transcript cleaning
                    context_str = f"""
                    <context>
                    Previous chunks: {additional_context.get('previous_chunks', [])}
                    Parameters: {json.dumps(additional_context.get('cleaning_parameters', {}), indent=2)}
                    </context>
                    """
                elif "doc_metadata" in additional_context:
                    # Format for document selection
                    context_str = f"""
                    <context>
                    Meeting metadata: {json.dumps(additional_context.get('meeting_metadata', {}), indent=2)}
                    Selection criteria: {json.dumps(additional_context.get('selection_criteria', {}), indent=2)}
                    </context>
                    """
                else:
                    # Generic context format
                    context_str = f"""
                    <context>
                    {json.dumps(additional_context, indent=2)}
                    </context>
                    """

            # Create custom task with embedded context
            custom_task = Task(
                description=f"""{task_description}

                <input>
                {prompt}
                </input>
                
                {context_str}""",
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

            # Execute task and get output
            result = crew.kickoff()
            
            # Get task output
            if hasattr(result, 'tasks_output') and result.tasks_output:
                return result.tasks_output[0].raw
            return result.raw if hasattr(result, 'raw') else str(result)
            
        except Exception as e:
            logger.error(f"Custom task execution failed: {str(e)}")
            raise

    def analyze_transcript(self, transcript: str, analysis_type: str, role: str, prompt: str, additional_context: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> str:
        """Analyze meeting transcript using multiple agents.
        
        Returns:
            str: A markdown-formatted string containing the analysis. This string will be saved
                 as a markdown file by the TranscriptProcessor.
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Format context sections
                context_sections = []
                
                if additional_context:
                    if "doc_context" in additional_context:
                        context_sections.append(f"""
                        <documents>
                        {additional_context['doc_context']}
                        </documents>
                        """)
                    
                    if "metadata_context" in additional_context:
                        context_sections.append(f"""
                        <metadata>
                        {additional_context['metadata_context']}
                        </metadata>
                        """)
                    
                    if "previous_analyses" in additional_context:
                        prev_analyses = []
                        for analysis_type, content in additional_context["previous_analyses"].items():
                            prev_analyses.append(f"<analysis type='{analysis_type}'>\n{content}\n</analysis>")
                        if prev_analyses:
                            context_sections.append(f"""
                            <previous_analyses>
                            {"\n".join(prev_analyses)}
                            </previous_analyses>
                            """)
                    
                    # Add any remaining context
                    other_context = {k: v for k, v in additional_context.items() 
                                   if k not in {"doc_context", "metadata_context", "previous_analyses"}}
                    if other_context:
                        context_sections.append(f"""
                        <additional_context>
                        {json.dumps(other_context, indent=2)}
                        </additional_context>
                        """)

                # Create analysis task with embedded context
                task = Task(
                    description=f"""{prompt}

                    <transcript>
                    {transcript}
                    </transcript>
                    
                    {"\n".join(context_sections)}
                    
                    Format your response in markdown following the example in the prompt.""",
                    expected_output=f"A detailed {analysis_type} analysis in markdown format.",
                    agent=Agent(
                        role=analysis_type.title(),
                        goal=f"Generate a comprehensive {analysis_type} analysis",
                        backstory=role,
                        llm=self.llm,
                        verbose=True
                    )
                )

                # Create and run crew
                crew = Crew(
                    agents=[task.agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )

                # Execute analysis and get task output
                result = crew.kickoff()
                
                # Get markdown content from task output
                if hasattr(result, 'tasks_output') and result.tasks_output:
                    markdown_content = result.tasks_output[0].raw
                else:
                    markdown_content = result.raw if hasattr(result, 'raw') else str(result)
                
                # Validate markdown format
                if not markdown_content.strip().startswith('#'):
                    logger.warning("Analysis result does not start with markdown heading, attempting to format")
                    markdown_content = f"# Meeting {analysis_type.title()}\n\n{markdown_content}"
                
                return markdown_content

            except Exception as e:
                logger.error(f"Analysis attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    raise RuntimeError(f"Analysis failed after {max_retries} attempts") from e
