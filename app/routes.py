"""
Flask routes for the Autonomous SRE Dashboard
"""
import os
import json
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import embed_logs
from src.clustering import cluster_embeddings, summarize_clusters, get_cluster_statistics
from src.vectordb import get_vectordb_service
from src.agent import get_agent, Recommendation
from src.config import LOGS_DIR, RUNBOOKS_DIR

# Create blueprint
main = Blueprint('main', __name__)

# Store current incidents in memory (for demo)
current_incidents = {}


@main.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html', incidents=list(current_incidents.values()))


@main.route('/api/stats')
def get_stats():
    """Get system statistics"""
    vectordb = get_vectordb_service()
    stats = vectordb.get_collection_stats()
    stats['active_incidents'] = len(current_incidents)
    return jsonify(stats)


@main.route('/api/incidents')
def get_incidents():
    """Get all current incidents"""
    return jsonify(list(current_incidents.values()))


@main.route('/api/incidents/<incident_id>')
def get_incident(incident_id):
    """Get a specific incident"""
    if incident_id in current_incidents:
        return jsonify(current_incidents[incident_id])
    return jsonify({"error": "Incident not found"}), 404


@main.route('/api/ingest', methods=['POST'])
def ingest_logs():
    """
    Ingest and analyze logs
    Accepts either file upload or JSON body with logs
    """
    logs = []
    
    # Check for file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename:
            content = file.read().decode('utf-8')
            logs = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Check for JSON body
    elif request.is_json:
        data = request.get_json()
        logs = data.get('logs', [])
    
    if not logs:
        return jsonify({"error": "No logs provided"}), 400
    
    try:
        # Process logs
        result = process_logs(logs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback on an incident recommendation
    """
    data = request.get_json()
    
    incident_id = data.get('incident_id')
    was_correct = data.get('was_correct', True)
    correct_answer = data.get('correct_answer', '')
    
    if not incident_id or incident_id not in current_incidents:
        return jsonify({"error": "Incident not found"}), 404
    
    incident = current_incidents[incident_id]
    
    # Store feedback in vector DB
    vectordb = get_vectordb_service()
    feedback_id = vectordb.add_feedback(
        incident_id=incident_id,
        cluster_summary=incident['cluster_summary'],
        original_recommendation=incident['recommendation'],
        was_correct=was_correct,
        correct_answer=correct_answer if not was_correct else None
    )
    
    # Mark incident as resolved
    incident['resolved'] = True
    incident['feedback'] = {
        'was_correct': was_correct,
        'correct_answer': correct_answer,
        'feedback_id': feedback_id
    }
    
    return jsonify({
        "success": True,
        "feedback_id": feedback_id,
        "message": "Feedback recorded. The agent will learn from this!"
    })


@main.route('/api/load-sample', methods=['POST'])
def load_sample_logs():
    """Load sample HDFS logs for demo"""
    # Check if sample logs exist
    sample_file = LOGS_DIR / 'hdfs_sample.log'
    
    if not sample_file.exists():
        # Create sample logs for demo
        sample_logs = generate_sample_logs()
        with open(sample_file, 'w') as f:
            f.write('\n'.join(sample_logs))
    
    with open(sample_file, 'r') as f:
        logs = [line.strip() for line in f.readlines() if line.strip()]
    
    result = process_logs(logs)
    return jsonify(result)


@main.route('/api/init-runbooks', methods=['POST'])
def init_runbooks():
    """Initialize runbooks in vector DB"""
    runbooks_file = RUNBOOKS_DIR / 'runbooks.json'
    
    if not runbooks_file.exists():
        return jsonify({"error": "Runbooks file not found"}), 404
    
    with open(runbooks_file, 'r') as f:
        runbooks = json.load(f)
    
    vectordb = get_vectordb_service()
    count = vectordb.add_runbooks(runbooks)
    
    return jsonify({
        "success": True,
        "message": f"Loaded {count} runbooks into vector database"
    })


def process_logs(logs):
    """Process logs through the full pipeline"""
    global current_incidents
    
    # Step 1: Generate embeddings
    embeddings = embed_logs(logs)
    
    # Step 2: Cluster
    labels = cluster_embeddings(embeddings)
    stats = get_cluster_statistics(labels)
    
    # Step 3: Summarize clusters
    clusters = summarize_clusters(logs, labels)
    
    # Step 4: Analyze each cluster
    agent = get_agent()
    new_incidents = []
    
    for cluster in clusters:
        recommendation = agent.analyze_incident(cluster)
        incident_dict = recommendation.to_dict()
        incident_dict['resolved'] = False
        
        current_incidents[recommendation.incident_id] = incident_dict
        new_incidents.append(incident_dict)
    
    return {
        "success": True,
        "stats": stats,
        "incidents_created": len(new_incidents),
        "incidents": new_incidents
    }


def generate_sample_logs():
    """Generate sample HDFS-like logs for demo"""
    import random
    from datetime import datetime, timedelta
    
    log_templates = [
        # Connection errors (cluster 1)
        "ERROR dfs.DataNode: Error connecting to NameNode at namenode.cluster.local:9000 - Connection refused",
        "WARN dfs.DataNode: Failed to report blocks to NameNode - java.net.ConnectException: Connection refused",
        "ERROR dfs.DataNode: Unable to connect to NameNode - timeout after 30000ms",
        "WARN hadoop.ipc.Client: Retrying connect to NameNode. Already tried 0 time(s)",
        "ERROR dfs.DataNode: Block report failed - namenode.cluster.local:9000 unreachable",
        
        # Block replication issues (cluster 2)
        "WARN dfs.BlockStateChange: BLOCK* BlockMissing: blk_1234567890 is missing on data01.cluster.local",
        "WARN hdfs.StateChange: Block blk_9876543210 has only 1 replicas below minimum 3",
        "ERROR dfs.DataNode: Block replication failed for blk_1111111111 - no available DataNodes",
        "WARN dfs.BlockStateChange: Under-replicated block count: 156 blocks below minimum replication",
        
        # Disk space issues (cluster 3)
        "ERROR dfs.DataNode: Disk check failed for volume /data/hdfs/dn - No space left on device",
        "WARN dfs.DataNode: DataNode /data/hdfs/dn running low on disk space: 2% remaining",
        "ERROR dfs.DataNode: Unable to create new block - all volumes are full",
        "WARN dfs.DataNode: Volume /data/hdfs/dn2 has crossed threshold: 95% used",
        
        # Normal operations (noise)
        "INFO dfs.DataNode: Received block blk_2222222222 from /192.168.1.100:50010",
        "INFO dfs.DataNode: Successfully sent block blk_3333333333 to /192.168.1.101:50010",
        "INFO hadoop.hdfs: File /user/hadoop/data.txt closed successfully",
    ]
    
    logs = []
    base_time = datetime.now() - timedelta(hours=1)
    
    # Generate ~200 logs
    for i in range(200):
        template = random.choice(log_templates)
        timestamp = (base_time + timedelta(seconds=i*18)).strftime('%Y-%m-%d %H:%M:%S')
        logs.append(f"{timestamp},{i+1000} {template}")
    
    return logs
