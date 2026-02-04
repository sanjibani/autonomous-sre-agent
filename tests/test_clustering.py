"""
Tests for the clustering module
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestClustering:
    """Tests for DBSCAN clustering"""
    
    def test_cluster_embeddings_returns_labels(self):
        """Test that cluster_embeddings returns valid labels"""
        from src.clustering import cluster_embeddings
        
        # Create fake embeddings (3 clusters)
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Group 1
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.0],  # Group 2
            [0.1, 1.0, 0.0],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 1.0],  # Group 3
            [0.1, 0.0, 1.0],
        ])
        
        labels = cluster_embeddings(embeddings, eps=0.3, min_samples=2)
        
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(embeddings)
    
    def test_noise_points_labeled_minus_one(self):
        """Test that noise points are labeled as -1"""
        from src.clustering import cluster_embeddings
        
        # Create embeddings with a clear outlier (opposite direction)
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Group pointing in +x direction
            [0.95, 0.05, 0.0],
            [0.9, 0.1, 0.0],
            [-1.0, -1.0, -1.0],  # Outlier pointing in opposite direction
        ])
        
        labels = cluster_embeddings(embeddings, eps=0.3, min_samples=2)
        
        # The outlier should be noise (-1)
        assert labels[-1] == -1
    
    def test_summarize_clusters(self):
        """Test cluster summarization"""
        from src.clustering import summarize_clusters, ClusterInfo
        
        logs = [
            "ERROR: Connection refused",
            "ERROR: Connection timeout",
            "ERROR: Unable to connect",
            "INFO: Normal operation",
            "INFO: All systems ok",
        ]
        labels = np.array([0, 0, 0, 1, 1])
        
        clusters = summarize_clusters(logs, labels)
        
        assert isinstance(clusters, list)
        assert len(clusters) == 2
        assert all(isinstance(c, ClusterInfo) for c in clusters)
        
        # Check cluster with errors detected as high severity
        error_cluster = next(c for c in clusters if c.size == 3)
        assert error_cluster.severity_hint == "high"
    
    def test_get_cluster_statistics(self):
        """Test cluster statistics calculation"""
        from src.clustering import get_cluster_statistics
        
        labels = np.array([0, 0, 0, 1, 1, -1, -1])
        
        stats = get_cluster_statistics(labels)
        
        assert stats["n_clusters"] == 2
        assert stats["n_noise"] == 2
        assert stats["n_total"] == 7
        assert 3 in stats["cluster_sizes"]
        assert 2 in stats["cluster_sizes"]


class TestClusterSeverity:
    """Tests for severity detection in clusters"""
    
    def test_high_severity_keywords(self):
        """Test detection of high severity keywords"""
        from src.clustering import summarize_clusters
        
        logs = [
            "FATAL: System crash detected",
            "CRITICAL: Database failure",
            "ERROR: Exception in main thread",
        ]
        labels = np.array([0, 0, 0])
        
        clusters = summarize_clusters(logs, labels)
        
        assert clusters[0].severity_hint == "high"
    
    def test_medium_severity_keywords(self):
        """Test detection of medium severity keywords"""
        from src.clustering import summarize_clusters
        
        logs = [
            "WARNING: Memory usage high",
            "WARN: Retry attempt 3",
            "WARNING: Slow response time",
        ]
        labels = np.array([0, 0, 0])
        
        clusters = summarize_clusters(logs, labels)
        
        assert clusters[0].severity_hint == "medium"
    
    def test_low_severity_for_info_logs(self):
        """Test that info logs are low severity"""
        from src.clustering import summarize_clusters
        
        logs = [
            "INFO: Server started",
            "INFO: User logged in",
            "DEBUG: Processing request",
        ]
        labels = np.array([0, 0, 0])
        
        clusters = summarize_clusters(logs, labels)
        
        assert clusters[0].severity_hint == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
