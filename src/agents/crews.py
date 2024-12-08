# src/agents/crews.py
from typing import Dict, List
from crewai import Task, Crew, Process
from crewai.tasks import TaskOutput

from src.agents.roles import AgentRoles

class TaskSpecs:
    """Specifications for different types of tasks."""
    
    @staticmethod
    def create_preparation_tasks(context: Dict) -> List[Task]:
        """Create tasks for meeting preparation."""
        project_metadata = context.get("project_metadata", {})
        meeting_metadata = context.get("meeting_metadata", {})
        
        return [
            Task(
                description="""Review all available meeting documents and prepare a briefing.
                1. Analyze the meeting transcript and any related documents
                2. Identify key topics and themes
                3. Note any important context from previous discussions
                4. Highlight areas that need special attention
                
                Expected Output: A comprehensive briefing document in markdown format that 
                summarizes the key points and provides context for the meeting.""",
                agent=AgentRoles.documentation_expert(project_metadata, meeting_metadata),
                context=context
            ),
            Task(
                description="""Create a structured agenda based on the document review.
                1. Identify main discussion points
                2. List open questions that need addressing
                3. Note any decisions that need to be made
                4. Suggest time allocations for each topic
                
                Expected Output: A structured agenda in markdown format with clear topics,
                questions, and time allocations.""",
                agent=AgentRoles.meeting_analyst(project_metadata, meeting_metadata),
                context=context
            )
        ]
    
    @staticmethod
    def create_summary_tasks(context: Dict) -> List[Task]:
        """Create tasks for meeting summarization."""
        project_metadata = context.get("project_metadata", {})
        meeting_metadata = context.get("meeting_metadata", {})
        
        return [
            Task(
                description="""Analyze the meeting transcript and identify key information.
                1. Extract main discussion points
                2. List all decisions made
                3. Capture action items and assignments
                4. Note any important insights or concerns raised
                
                Expected Output: A detailed analysis in markdown format covering all key
                points from the meeting.""",
                agent=AgentRoles.meeting_analyst(project_metadata, meeting_metadata),
                context=context
            ),
            Task(
                description="""Create a structured summary document.
                1. Organize the key points into clear sections
                2. Link to relevant background information
                3. Highlight next steps and action items
                4. Include references to any mentioned documents
                
                Expected Output: A well-organized summary document in markdown format that
                captures all essential information.""",
                agent=AgentRoles.documentation_expert(project_metadata, meeting_metadata),
                context=context
            )
        ]
    
    @staticmethod
    def create_task_breakdown_tasks(context: Dict) -> List[Task]:
        """Create tasks for work breakdown."""
        project_metadata = context.get("project_metadata", {})
        meeting_metadata = context.get("meeting_metadata", {})
        
        return [
            Task(
                description="""Review the meeting content and identify all tasks.
                1. List all work items mentioned
                2. Group related tasks together
                3. Identify dependencies between tasks
                4. Note any constraints or requirements
                
                Expected Output: A JSON document listing all tasks with their relationships
                and requirements.""",
                agent=AgentRoles.task_planner(project_metadata, meeting_metadata),
                context=context
            ),
            Task(
                description="""Analyze technical aspects of identified tasks.
                1. Assess technical complexity
                2. Identify technical dependencies
                3. Note potential challenges
                4. Suggest implementation approaches
                
                Expected Output: A JSON document with technical analysis for each task,
                including complexity estimates and recommendations.""",
                agent=AgentRoles.technical_analyst(project_metadata, meeting_metadata),
                context=context
            )
        ]
    
    @staticmethod
    def create_question_tasks(context: Dict) -> List[Task]:
        """Create tasks for identifying questions."""
        project_metadata = context.get("project_metadata", {})
        meeting_metadata = context.get("meeting_metadata", {})
        
        return [
            Task(
                description="""Review the meeting content and identify all questions.
                1. List explicit questions raised
                2. Identify implicit questions from discussions
                3. Note areas of uncertainty
                4. Highlight decisions needing clarification
                
                Expected Output: A markdown document listing all identified questions
                with context and priority.""",
                agent=AgentRoles.meeting_analyst(project_metadata, meeting_metadata),
                context=context
            ),
            Task(
                description="""Organize and categorize identified questions.
                1. Group related questions
                2. Prioritize based on impact
                3. Link to relevant documentation
                4. Suggest approaches for resolution
                
                Expected Output: A structured markdown document with categorized
                questions and recommended next steps.""",
                agent=AgentRoles.documentation_expert(project_metadata, meeting_metadata),
                context=context
            )
        ]

class CrewFactory:
    """Factory for creating specialized crews."""
    
    @staticmethod
    def create_preparation_crew(context: Dict) -> Crew:
        """Create a crew for meeting preparation."""
        project_metadata = context.get("project_metadata", {})
        meeting_metadata = context.get("meeting_metadata", {})
        
        agents = [
            AgentRoles.documentation_expert(project_metadata, meeting_metadata),
            AgentRoles.meeting_analyst(project_metadata, meeting_metadata)
        ]
        tasks = TaskSpecs.create_preparation_tasks(context)
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
    
    @staticmethod
    def create_summary_crew(context: Dict) -> Crew:
        """Create a crew for meeting summarization."""
        project_metadata = context.get("project_metadata", {})
        meeting_metadata = context.get("meeting_metadata", {})
        
        agents = [
            AgentRoles.meeting_analyst(project_metadata, meeting_metadata),
            AgentRoles.documentation_expert(project_metadata, meeting_metadata)
        ]
        tasks = TaskSpecs.create_summary_tasks(context)
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
    
    @staticmethod
    def create_task_breakdown_crew(context: Dict) -> Crew:
        """Create a crew for task breakdown."""
        project_metadata = context.get("project_metadata", {})
        meeting_metadata = context.get("meeting_metadata", {})
        
        agents = [
            AgentRoles.task_planner(project_metadata, meeting_metadata),
            AgentRoles.technical_analyst(project_metadata, meeting_metadata)
        ]
        tasks = TaskSpecs.create_task_breakdown_tasks(context)
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
    
    @staticmethod
    def create_question_crew(context: Dict) -> Crew:
        """Create a crew for identifying questions."""
        project_metadata = context.get("project_metadata", {})
        meeting_metadata = context.get("meeting_metadata", {})
        
        agents = [
            AgentRoles.meeting_analyst(project_metadata, meeting_metadata),
            AgentRoles.documentation_expert(project_metadata, meeting_metadata)
        ]
        tasks = TaskSpecs.create_question_tasks(context)
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
