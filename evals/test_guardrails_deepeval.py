import os
import sys
import pytest
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric

# Import Agent code
from src.agent import get_agent, Recommendation
from src.clustering import ClusterInfo
from src.vectordb import get_vectordb_service
import json

# Retrieve API Key
from src.config import OPENAI_API_KEY, RUNBOOKS_DIR
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def _load_runbooks():
    """Helper to load runbooks"""
    print("\n📦 Loading Runbooks for Testing...")
    runbooks_file = RUNBOOKS_DIR / 'runbooks.json'
    if runbooks_file.exists():
        with open(runbooks_file, 'r') as f:
            runbooks = json.load(f)
        vectordb = get_vectordb_service()
        vectordb.add_runbooks(runbooks)
        print(f"   - Loaded {len(runbooks)} runbooks")
    else:
        raise FileNotFoundError("Runbooks file not found!")

@pytest.fixture(scope="module", autouse=True)
def setup_knowledge_base():
    """Ensure vector DB has runbooks loaded before testing"""
    _load_runbooks()

def test_disk_space_incident():
    """
    Integration Test: 
    Simulate a 'Disk Full' incident and ensure the agent recommends 
    cleaning files or running balancer (based on runbooks).
    """
    
    # 1. Create a Synthetic Incident
    # We pretend DBSCAN found this cluster of logs
    cluster = ClusterInfo(
        cluster_id=1,
        size=50,
        representative_logs=[
            "ERROR dfs.DataNode: Disk check failed for volume /data/hdfs/dn - No space left on device",
            "WARN dfs.DataNode: DataNode /data/hdfs/dn running low on disk space: 2% remaining"
        ],
        error_keywords=["disk", "space", "full", "datanode"],
        severity_hint="high"
    )
    
    # 2. Run the Agent
    agent = get_agent()
    print("\n🤖 Agent Analyzing Cluster...")
    recommendation: Recommendation = agent.analyze_incident(cluster)
    
    print(f"   - Root Cause: {recommendation.root_cause}")
    print(f"   - Recommendation: {recommendation.recommendation}")
    
    # 3. Define the Test Case for DeepEval
    # Input: The context of the incident (logs)
    input_query = f"Incident logs: {cluster.representative_logs}. Keywords: {cluster.error_keywords}"
    
    test_case = LLMTestCase(
        input=input_query,
        actual_output=recommendation.recommendation,
        context=recommendation.retrieved_context 
    )

    # 4. Metrics
    
    # Metric A: Hallucination
    # Ensure the command (e.g., `hdfs balancer`) actually came from the runbook context
    hallucination_metric = HallucinationMetric(threshold=0.5)
    
    # Metric B: Relevancy
    # Ensure the answer is about "Disk Space"
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

    # 5. Assert
    assert_test(test_case, [hallucination_metric, relevancy_metric])

if __name__ == "__main__":
    # Make it easy to debug without pytest
    try:
        _load_runbooks()
        test_disk_space_incident()
        print("✅ integration Test Passed")
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")
    except Exception as e:
        print(f"⚠️ Error: {e}")
