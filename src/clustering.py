"""
Clustering module using HDBSCAN
Groups similar log entries into clusters for incident detection
Includes anomaly detection for flagging unusual patterns
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

# Try HDBSCAN first, fall back to DBSCAN if not available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    from sklearn.cluster import DBSCAN

from collections import Counter

from .config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES
from .anomaly import get_anomaly_detector


@dataclass
class ClusterInfo:
    """Information about a log cluster"""
    cluster_id: int
    size: int
    representative_logs: List[str]
    error_keywords: List[str]
    severity_hint: str  # Based on keywords
    has_anomalies: bool = False  # NEW: Flag if cluster contains anomalies
    anomaly_count: int = 0  # NEW: Number of anomalies in cluster


def cluster_embeddings(
    embeddings: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    detect_anomalies: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster embeddings using HDBSCAN (or DBSCAN fallback)
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        eps: Maximum distance (only used for DBSCAN fallback)
        min_samples: Minimum samples in a cluster
        detect_anomalies: Whether to run anomaly detection
        
    Returns:
        Tuple of (cluster_labels, anomaly_mask)
        - cluster_labels: Array of cluster labels (-1 for noise)
        - anomaly_mask: Boolean array (True = anomaly)
    """
    # Run anomaly detection first
    anomaly_mask = np.zeros(len(embeddings), dtype=bool)
    if detect_anomalies and len(embeddings) >= 10:
        detector = get_anomaly_detector()
        anomaly_mask = detector.get_anomaly_mask(embeddings)
    
    # Cluster using HDBSCAN or DBSCAN
    if HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, min_samples),
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass
            prediction_data=True
        )
        labels = clusterer.fit_predict(embeddings)
    else:
        # Fallback to DBSCAN
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='cosine'
        )
        labels = clusterer.fit_predict(embeddings)
    
    return labels, anomaly_mask


def summarize_clusters(
    logs: List[str],
    labels: np.ndarray,
    anomaly_mask: np.ndarray = None,
    max_representatives: int = 5
) -> List[ClusterInfo]:
    """
    Create summaries for each cluster
    
    Args:
        logs: Original log lines
        labels: Cluster labels from HDBSCAN/DBSCAN
        anomaly_mask: Boolean array indicating anomalies
        max_representatives: Max number of representative logs per cluster
        
    Returns:
        List of ClusterInfo objects
    """
    if anomaly_mask is None:
        anomaly_mask = np.zeros(len(logs), dtype=bool)
    clusters = []
    unique_labels = set(labels)
    
    # Error keywords for severity detection
    high_severity_keywords = {'error', 'exception', 'failed', 'failure', 'critical', 'fatal', 'crash'}
    medium_severity_keywords = {'warning', 'warn', 'timeout', 'retry', 'slow'}
    
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        
        # Get logs in this cluster
        mask = labels == label
        cluster_logs = [logs[i] for i in range(len(logs)) if mask[i]]
        
        # Get representative samples (evenly distributed)
        step = max(1, len(cluster_logs) // max_representatives)
        representatives = cluster_logs[::step][:max_representatives]
        
        # Extract common keywords
        all_words = ' '.join(cluster_logs).lower().split()
        word_counts = Counter(all_words)
        
        # Filter to meaningful words (length > 3, not common)
        stop_words = {'the', 'and', 'for', 'this', 'that', 'with', 'from', 'are', 'was', 'were'}
        keywords = [
            word for word, count in word_counts.most_common(20)
            if len(word) > 3 and word not in stop_words
        ][:10]
        
        # Determine severity hint based on keywords
        cluster_text = ' '.join(cluster_logs).lower()
        if any(kw in cluster_text for kw in high_severity_keywords):
            severity = "high"
        elif any(kw in cluster_text for kw in medium_severity_keywords):
            severity = "medium"
        else:
            severity = "low"
        
        # Count anomalies in this cluster
        cluster_anomalies = [i for i in range(len(logs)) if mask[i] and anomaly_mask[i]]
        anomaly_count = len(cluster_anomalies)
        
        clusters.append(ClusterInfo(
            cluster_id=int(label),
            size=len(cluster_logs),
            representative_logs=representatives,
            error_keywords=keywords,
            severity_hint=severity,
            has_anomalies=anomaly_count > 0,
            anomaly_count=anomaly_count
        ))
    
    # Sort by size (largest clusters first), then by anomaly presence
    clusters.sort(key=lambda x: (x.has_anomalies, x.size), reverse=True)
    
    return clusters


def get_cluster_statistics(labels: np.ndarray) -> Dict[str, Any]:
    """
    Get statistics about the clustering results
    
    Args:
        labels: Cluster labels from DBSCAN
        
    Returns:
        Dictionary with clustering statistics
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    cluster_sizes = []
    for label in unique_labels:
        if label != -1:
            cluster_sizes.append(list(labels).count(label))
    
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_total": len(labels),
        "noise_ratio": n_noise / len(labels) if len(labels) > 0 else 0,
        "cluster_sizes": sorted(cluster_sizes, reverse=True),
        "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0
    }
