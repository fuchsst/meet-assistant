"""Test script for LLMClient functionality."""
import logging
from pathlib import Path
import json

from config.config import DATA_DIR
from llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_analysis():
    """Test LLM analysis functionality."""
    # Initialize
    client = LLMClient()
    test_data_dir = DATA_DIR / "test_llm"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Sample transcript for testing
    sample_transcript = """
    [00:00:00] Alice: Good morning everyone. Let's discuss the Q1 project timeline.
    [00:00:10] Bob: I've prepared the budget estimates as requested.
    [00:00:20] Alice: Great. We need to decide on the resource allocation.
    [00:00:30] Charlie: I suggest we increase the development team by two people.
    [00:00:40] Alice: Agreed. Bob, can you update the budget to reflect this?
    [00:00:50] Bob: I'll have it done by tomorrow.
    [00:01:00] Alice: Perfect. Let's also set up weekly progress reviews.
    [00:01:10] Charlie: I'll schedule those starting next week.
    """

    try:
        # Test 1: Agent Creation
        logger.info("Testing agent creation...")
        agents = client._create_agents()
        assert len(agents) == 3, "Expected 3 agents"
        assert "summarizer" in agents, "Missing summarizer agent"
        assert "task_extractor" in agents, "Missing task extractor agent"
        assert "decision_tracker" in agents, "Missing decision tracker agent"
        logger.info("Agent creation successful")

        # Test 2: Analysis Pipeline
        logger.info("\nTesting analysis pipeline...")
        analysis = client.analyze_transcript(sample_transcript)
        
        # Verify analysis structure
        assert "summary" in analysis, "Missing summary in analysis"
        assert "action_items" in analysis, "Missing action items in analysis"
        assert "decisions" in analysis, "Missing decisions in analysis"
        assert "metadata" in analysis, "Missing metadata in analysis"
        
        logger.info("Analysis structure verified")

        # Test 3: Result Validation
        logger.info("\nTesting result validation...")
        assert client._validate_analysis(analysis), "Analysis validation failed"
        
        # Test invalid analysis
        invalid_analysis = {
            "summary": "",  # Invalid: empty summary
            "action_items": "not a list",  # Invalid: should be a list
            "decisions": [],
            "metadata": {"timestamp": 123}  # Invalid: missing version
        }
        assert not client._validate_analysis(invalid_analysis), "Invalid analysis passed validation"
        
        logger.info("Validation testing successful")

        # Test 4: Save Analysis
        logger.info("\nTesting analysis saving...")
        output_file = test_data_dir / "test_analysis.json"
        client.save_analysis(analysis, output_file)
        
        # Verify saved file
        assert output_file.exists(), "Analysis file not created"
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_analysis = json.load(f)
        assert saved_analysis == analysis, "Saved analysis does not match original"
        
        logger.info("Analysis saved successfully")

        # Test 5: Error Handling
        logger.info("\nTesting error handling...")
        try:
            # Test with empty transcript
            client.analyze_transcript("")
            assert False, "Should have raised an error for empty transcript"
        except Exception as e:
            logger.info(f"Successfully caught error: {str(e)}")

        logger.info("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if test_data_dir.exists():
            import shutil
            shutil.rmtree(test_data_dir)
        logger.info("Test cleanup completed")

if __name__ == "__main__":
    test_llm_analysis()
