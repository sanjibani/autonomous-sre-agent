"""
Tests for the embedding module
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbeddingService:
    """Tests for EmbeddingService"""
    
    def test_model_loads(self):
        """Test that the embedding model loads successfully"""
        from src.embeddings import get_embedding_service
        service = get_embedding_service()
        assert service is not None
        assert service._model is not None
    
    def test_singleton_pattern(self):
        """Test that the same instance is returned"""
        from src.embeddings import get_embedding_service
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        assert service1 is service2
    
    def test_embed_single_text(self):
        """Test embedding a single text"""
        from src.embeddings import embed_texts
        from src.config import EMBEDDING_DIMENSION
        
        result = embed_texts("This is a test log message")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, EMBEDDING_DIMENSION)  # 384 dimensions
    
    def test_embed_multiple_texts(self):
        """Test embedding multiple texts"""
        from src.embeddings import embed_texts
        from src.config import EMBEDDING_DIMENSION
        
        texts = [
            "Error: Connection refused",
            "Warning: Disk space low",
            "Info: Server started"
        ]
        result = embed_texts(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, EMBEDDING_DIMENSION)
    
    def test_embed_logs_preprocessing(self):
        """Test that log preprocessing works"""
        from src.embeddings import get_embedding_service
        
        service = get_embedding_service()
        
        # Log with timestamp
        log = "2024-01-15T10:30:45.123 ERROR Connection refused"
        processed = service._preprocess_log(log)
        
        # Timestamp should be removed
        assert "2024-01-15" not in processed
        assert "ERROR" in processed
        assert "Connection" in processed
    
    def test_similar_logs_have_similar_embeddings(self):
        """Test that similar logs produce similar embeddings"""
        from src.embeddings import embed_texts
        from sklearn.metrics.pairwise import cosine_similarity
        
        logs = [
            "ERROR: Connection refused to database server",
            "ERROR: Unable to connect to database - connection refused",
            "INFO: User logged in successfully"
        ]
        
        embeddings = embed_texts(logs)
        
        # Compute cosine similarity
        similarity = cosine_similarity(embeddings)
        
        # First two logs should be more similar to each other than to the third
        assert similarity[0, 1] > similarity[0, 2]
        assert similarity[0, 1] > similarity[1, 2]


class TestEmbeddingPerformance:
    """Performance tests for embedding"""
    
    def test_batch_embedding_is_fast(self):
        """Test that batch embedding is reasonably fast"""
        import time
        from src.embeddings import embed_texts
        
        # Generate 100 sample logs
        logs = [f"Log message number {i} with some content" for i in range(100)]
        
        start = time.time()
        embed_texts(logs)
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds on CPU
        assert elapsed < 5.0, f"Batch embedding took {elapsed:.2f}s, expected < 5s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
