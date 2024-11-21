"""Test suite for AI analysis functionality."""
import pytest
from pathlib import Path
import json
from datetime import datetime

from core.ai.llm_client import LLMClient
from core.ai.analysis_pipeline import AnalysisPipeline

def test_llm_client_initialization():
    """Test LLMClient initialization."""
    client = LLMClient()
    assert client is not None
    assert client.agents is not None
    assert "summarizer" in client.agents
    assert "task_extractor" in client.agents
    assert "decision_tracker" in client.agents

def test_agent_creation():
    """Test agent creation and configuration."""
    client = LLMClient()
    agents = client._create_agents()
    
    # Verify all required agents exist
    assert len(agents) == 3
    assert all(agent.role for agent in agents.values())
    assert all(agent.goal for agent in agents.values())
    assert all(agent.backstory for agent in agents.values())
    assert all(agent.tools for agent in agents.values())

def test_transcript_analysis(mock_transcription):
    """Test basic transcript analysis."""
    client = LLMClient()
    
    # Analyze transcript
    analysis = client.analyze_transcript(mock_transcription)
    
    # Verify analysis structure
    assert "summary" in analysis
    assert "action_items" in analysis
    assert "decisions" in analysis
    assert "metadata" in analysis
    
    # Verify content types
    assert isinstance(analysis["summary"], str)
    assert isinstance(analysis["action_items"], list)
    assert isinstance(analysis["decisions"], list)
    assert isinstance(analysis["metadata"], dict)

def test_analysis_validation(mock_analysis_result):
    """Test analysis result validation."""
    client = LLMClient()
    
    # Test valid analysis
    assert client._validate_analysis(mock_analysis_result)
    
    # Test invalid analysis
    invalid_analysis = {
        "summary": "",  # Empty summary
        "action_items": "not a list",  # Wrong type
        "decisions": []
    }
    assert not client._validate_analysis(invalid_analysis)

def test_error_handling():
    """Test error handling and retries."""
    client = LLMClient()
    
    # Test with empty transcript
    with pytest.raises(ValueError):
        client.analyze_transcript("")
    
    # Test with invalid input
    with pytest.raises(Exception):
        client.analyze_transcript(None)

def test_analysis_pipeline_initialization():
    """Test AnalysisPipeline initialization."""
    pipeline = AnalysisPipeline()
    assert pipeline is not None
    assert pipeline.file_manager is not None
    assert pipeline.transcriber is not None
    assert pipeline.llm_client is not None

def test_complete_meeting_analysis(mock_meeting_dir):
    """Test complete meeting analysis process."""
    pipeline = AnalysisPipeline()
    meeting_id = mock_meeting_dir.name
    
    # Process meeting
    analysis = pipeline.process_meeting(meeting_id)
    
    # Verify analysis results
    assert isinstance(analysis, dict)
    assert "summary" in analysis
    assert "action_items" in analysis
    assert "decisions" in analysis

def test_real_time_processing(mock_audio_data):
    """Test real-time segment processing."""
    pipeline = AnalysisPipeline()
    meeting_id = "test_meeting"
    
    # Process audio segment
    result = pipeline.process_segment(meeting_id, mock_audio_data)
    
    # Verify result structure
    assert "transcript" in result
    assert "analysis" in result

def test_summary_generation(mock_meeting_dir):
    """Test meeting summary generation."""
    pipeline = AnalysisPipeline()
    meeting_id = mock_meeting_dir.name
    
    # Generate summary
    summary = pipeline.generate_summary(meeting_id)
    
    # Verify summary structure
    assert "summary" in summary
    assert "key_points" in summary
    assert "duration" in summary
    assert isinstance(summary["key_points"], list)
    assert isinstance(summary["duration"], int)

def test_task_extraction(mock_meeting_dir):
    """Test action item extraction."""
    pipeline = AnalysisPipeline()
    meeting_id = mock_meeting_dir.name
    
    # Extract tasks
    tasks = pipeline.extract_tasks(meeting_id)
    
    # Verify tasks
    assert isinstance(tasks, list)
    if tasks:
        task = tasks[0]
        assert "task" in task
        assert "assignee" in task
        assert isinstance(task["task"], str)

def test_decision_identification(mock_meeting_dir):
    """Test decision identification."""
    pipeline = AnalysisPipeline()
    meeting_id = mock_meeting_dir.name
    
    # Identify decisions
    decisions = pipeline.identify_decisions(meeting_id)
    
    # Verify decisions
    assert isinstance(decisions, list)
    assert all(isinstance(d, str) for d in decisions)

@pytest.mark.integration
def test_analysis_pipeline_integration(mock_meeting_dir, mock_audio_data):
    """Test complete analysis pipeline integration."""
    pipeline = AnalysisPipeline()
    meeting_id = mock_meeting_dir.name
    
    # Test real-time processing
    segment_result = pipeline.process_segment(meeting_id, mock_audio_data)
    assert "transcript" in segment_result
    assert "analysis" in segment_result
    
    # Test complete analysis
    analysis = pipeline.process_meeting(meeting_id)
    assert "summary" in analysis
    assert "action_items" in analysis
    assert "decisions" in analysis
    
    # Test summary generation
    summary = pipeline.generate_summary(meeting_id)
    assert "summary" in summary
    assert "key_points" in summary
    
    # Verify file creation
    files = pipeline.file_manager.get_meeting_files(meeting_id)
    assert files["analysis"].exists()

@pytest.mark.performance
def test_analysis_performance(mock_meeting_dir, mock_transcription):
    """Test analysis performance metrics."""
    pipeline = AnalysisPipeline()
    meeting_id = mock_meeting_dir.name
    
    import time
    
    # Test analysis speed
    start_time = time.time()
    pipeline.process_meeting(meeting_id)
    analysis_time = time.time() - start_time
    
    # Test real-time processing speed
    start_time = time.time()
    pipeline.process_segment(meeting_id, mock_audio_data)
    segment_time = time.time() - start_time
    
    # Verify performance
    assert analysis_time < 30.0, "Complete analysis too slow"
    assert segment_time < 5.0, "Real-time processing too slow"

def test_custom_task_processing(mock_meeting_dir):
    """Test custom analysis task processing."""
    pipeline = AnalysisPipeline()
    meeting_id = mock_meeting_dir.name
    
    # Test custom task
    custom_task = "Summarize the key points about project timeline"
    result = pipeline._process_custom_task(
        pipeline.file_manager.get_meeting_files(meeting_id)["transcript"],
        custom_task
    )
    
    # Verify result
    assert isinstance(result, dict)
    assert "response" in result
    assert isinstance(result["response"], str)
