import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Retrieve API Key
from src.config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def run_ragas_evaluation():
    """
    Runs a Ragas evaluation on a sample 'Golden Dataset' of SRE incidents.
    This simulates assessing:
    1. Did we retrieve the right runbook? (Context Precision)
    2. Did the agent give a relevant answer? (Answer Relevancy)
    3. Was the answer faithful to the runbook? (Faithfulness)
    """
    print("🚀  Starting Ragas Evaluation for SRE Agent...")

    # Sample Data (In production, this would be your test set)
    data_samples = {
        'question': [
            "My disk usage is at 98% on data node 2. What should I do?",
            "Namenode connection refused. How do I fix it?",
        ],
        'answer': [
            "You should check the disk usage with 'hdfs dfs -df -h' and then run the balancer command 'hdfs balancer -threshold 10'.",
            "You need to restart the namenode service using 'hdfs --daemon start namenode' and check if it is in safe mode.",
        ],
        'contexts': [
            ["## symptoms\n- logs show 'no space left on device'\n## resolution\n1. Check usage: `hdfs dfs -df -h`\n2. Run balancer: `hdfs balancer -threshold 10`"],  # Correct context for Q1
            ["## symptoms\n- logs show 'connection refused'\n## resolution\n1. Check status: `hdfs dfsadmin -report`\n2. Restart: `hdfs --daemon start namenode`"],      # Correct context for Q2
        ],
        'ground_truth': [
             "Run hdfs balancer with threshold 10 to clear space.",
             "Restart the namenode daemon."
        ]
    }

    # Create Dataset
    dataset = Dataset.from_dict(data_samples)

    # Run Evaluation
    # Note: Ragas uses OpenAI by default for grading
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
    )

    print("\n📊 Ragas Evaluation Results:")
    print(results)
    
    # Export for CI/CD
    df = results.to_pandas()
    print("\nDetailed breakdown:")
    print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])

if __name__ == "__main__":
    run_ragas_evaluation()
