# src/agents/roles.py
from typing import Dict, List
from crewai import Agent
from langchain.tools import BaseTool

from src.tools.document_tools import DocumentSelector, DocumentCombiner

class AgentRoles:
    """Factory for creating specialized agents with specific roles and capabilities."""
    
    @staticmethod
    def get_document_tools(project_metadata: Dict, meeting_metadata: Dict) -> List[BaseTool]:
        """Get the document tools initialized with metadata."""
        return [
            DocumentSelector(project_metadata, meeting_metadata),
            DocumentCombiner(project_metadata, meeting_metadata)
        ]
    
    @staticmethod
    def meeting_analyst(project_metadata: Dict, meeting_metadata: Dict, additional_tools: List[BaseTool] = None) -> Agent:
        """Create a Meeting Analyst agent specialized in understanding meeting content."""
        tools = AgentRoles.get_document_tools(project_metadata, meeting_metadata)
        if additional_tools:
            tools.extend(additional_tools)
            
        return Agent(
            role='Meeting Analyst',
            goal='Analyze meeting transcripts and extract key information',
            backstory="""You are an expert in analyzing meeting discussions and transcripts.
            Your strength lies in identifying key points, decisions, and action items while
            maintaining the context from related documentation. You excel at summarizing
            complex discussions into clear, actionable insights.""",
            tools=tools,
            verbose=True,
            allow_delegation=True
        )
    
    @staticmethod
    def documentation_expert(project_metadata: Dict, meeting_metadata: Dict, additional_tools: List[BaseTool] = None) -> Agent:
        """Create a Documentation Expert agent specialized in document organization."""
        tools = AgentRoles.get_document_tools(project_metadata, meeting_metadata)
        if additional_tools:
            tools.extend(additional_tools)
            
        return Agent(
            role='Documentation Expert',
            goal='Organize and synthesize information from multiple documents',
            backstory="""You are a documentation specialist with expertise in technical
            writing and information organization. You excel at finding connections between
            different documents and presenting information in a clear, structured format.
            You understand how to maintain context while summarizing complex documents.""",
            tools=tools,
            verbose=True,
            allow_delegation=True
        )
    
    @staticmethod
    def task_planner(project_metadata: Dict, meeting_metadata: Dict, additional_tools: List[BaseTool] = None) -> Agent:
        """Create a Task Planner agent specialized in work breakdown."""
        tools = AgentRoles.get_document_tools(project_metadata, meeting_metadata)
        if additional_tools:
            tools.extend(additional_tools)
            
        return Agent(
            role='Task Planner',
            goal='Create detailed task breakdowns and project plans',
            backstory="""You are an experienced project manager with a talent for breaking
            down complex work into manageable tasks. You understand dependencies,
            resource requirements, and how to sequence work effectively. You excel at
            identifying risks and ensuring all aspects of work are properly planned.""",
            tools=tools,
            verbose=True,
            allow_delegation=True
        )
    
    @staticmethod
    def technical_analyst(project_metadata: Dict, meeting_metadata: Dict, additional_tools: List[BaseTool] = None) -> Agent:
        """Create a Technical Analyst agent specialized in technical requirements."""
        tools = AgentRoles.get_document_tools(project_metadata, meeting_metadata)
        if additional_tools:
            tools.extend(additional_tools)
            
        return Agent(
            role='Technical Analyst',
            goal='Analyze technical aspects and requirements',
            backstory="""You are a technical expert who excels at understanding and
            breaking down technical requirements and implementation details. You can
            identify technical dependencies, potential challenges, and provide
            implementation recommendations. You have a strong background in software
            development and system architecture.""",
            tools=tools,
            verbose=True,
            allow_delegation=True
        )
