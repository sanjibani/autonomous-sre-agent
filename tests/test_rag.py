"""
Tests for the RAG module
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVectorDB:
    """Tests for VectorDB service"""
    
    def test_vectordb_initializes(self):
        """Test that VectorDB initializes correctly"""
        from src.vectordb import get_vectordb_service
        
        service = get_vectordb_service()
        
        assert service is not None
        assert service.runbooks is not None
        assert service.feedback is not None
    
    def test_add_and_search_runbooks(self):
        """Test adding and searching runbooks"""
        from src.vectordb import get_vectordb_service
        
        service = get_vectordb_service()
        
        # Add test runbooks
        test_runbooks = [
            {
                "title": "Test Connection Error",
                "content": "When you see connection refused errors, check if the service is running",
                "category": "network"
            },
            {
                "title": "Test Disk Space",
                "content": "Disk space issues require cleanup or expansion",
                "category": "storage"
            }
        ]
        
        count = service.add_runbooks(test_runbooks)
        assert count == 2
        
        # Search for connection issues
        results = service.search_runbooks("connection refused error", n_results=2)
        
        assert len(results) > 0
        # First result should be about connection
        assert "connection" in results[0]["document"].lower()
    
    def test_add_and_search_feedback(self):
        """Test storing and retrieving feedback"""
        from src.vectordb import get_vectordb_service
        
        service = get_vectordb_service()
        
        # Add feedback
        feedback_id = service.add_feedback(
            incident_id="TEST-001",
            cluster_summary="Connection errors",
            original_recommendation="Restart service",
            was_correct=True
        )
        
        assert feedback_id is not None
        
        # Search feedback
        results = service.search_feedback("connection", n_results=1)
        
        # Should find the feedback
        assert len(results) > 0


class TestRAGService:
    """Tests for RAG retrieval"""
    
    def test_retrieve_context(self):
        """Test context retrieval for a cluster"""
        from src.rag import get_rag_service, RAGContext
        from src.clustering import ClusterInfo
        
        rag = get_rag_service()
        
        # Create a test cluster
        cluster = ClusterInfo(
            cluster_id=0,
            size=100,
            representative_logs=[
                "ERROR: Connection refused to database",
                "ERROR: Unable to connect to mysql:3306",
            ],
            error_keywords=["connection", "refused", "database", "error"],
            severity_hint="high"
        )
        
        context = rag.retrieve_context(cluster)
        
        assert isinstance(context, RAGContext)
        assert context.combined_context is not None
        assert context.confidence_boost >= 0.0
    
    def test_confidence_boost_calculation(self):
        """Test that confidence boost is calculated correctly"""
        from src.rag import RAGService
        
        rag = RAGService()
        
        # High similarity match should give boost
        runbook_matches = [{"similarity": 0.8, "document": "test"}]
        feedback_matches = [{"similarity": 0.7, "metadata": {"was_correct": True}}]
        
        boost = rag._calculate_confidence_boost(runbook_matches, feedback_matches)
        
        assert boost > 0.0
        assert boost <= 0.2  # Max boost is 0.2


class TestQueryBuilding:
    """Tests for query building"""
    
    def test_build_query_from_cluster(self):
        """Test building search query from cluster info"""
        from src.rag import RAGService
        from src.clustering import ClusterInfo
        
        rag = RAGService()
        
        cluster = ClusterInfo(
            cluster_id=0,
            size=50,
            representative_logs=[
                "Connection timeout after 30s",
                "Failed to connect to remote host",
            ],
            error_keywords=["timeout", "connection", "failed"],
            severity_hint="medium"
        )
        
        query = rag._build_query(cluster)
        
        assert "timeout" in query
        assert "connection" in query
        assert "Connection timeout" in query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
