"""
Clustering module using DBSCAN
Groups similar log entries into clusters for incident detection
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

from .config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES


@dataclass
class ClusterInfo:
    """Information about a log cluster"""
    cluster_id: int
    size: int
    representative_logs: List[str]
    error_keywords: List[str]
    severity_hint: str  # Based on keywords


def cluster_embeddings(
    embeddings: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES
) -> np.ndarray:
    """
    Cluster embeddings using DBSCAN
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        eps: Maximum distance between two samples to be in same neighborhood
        min_samples: Minimum number of samples in a neighborhood for a core point
        
    Returns:
        Array of cluster labels (-1 for noise)
    """
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='cosine'  # Better for high-dimensional embeddings
    )
    
    labels = clustering.fit_predict(embeddings)
    return labels


def summarize_clusters(
    logs: List[str],
    labels: np.ndarray,
    max_representatives: int = 5
) -> List[ClusterInfo]:
    """
    Create summaries for each cluster
    
    Args:
        logs: Original log lines
        labels: Cluster labels from DBSCAN
        max_representatives: Max number of representative logs per cluster
        
    Returns:
        List of ClusterInfo objects
    """
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
        
        clusters.append(ClusterInfo(
            cluster_id=int(label),
            size=len(cluster_logs),
            representative_logs=representatives,
            error_keywords=keywords,
            severity_hint=severity
        ))
    
    # Sort by size (largest clusters first)
    clusters.sort(key=lambda x: x.size, reverse=True)
    
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
