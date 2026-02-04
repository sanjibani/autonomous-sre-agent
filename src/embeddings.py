"""
Embedding module using sentence-transformers
Uses all-MiniLM-L6-v2 for CPU-efficient embeddings
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, USE_OPENAI_EMBEDDINGS, OPENAI_API_KEY


class EmbeddingService:
    """Service for generating embeddings from text"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to reuse model across requests"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            if USE_OPENAI_EMBEDDINGS:
                print(f"Initializing OpenAI embedding service: {EMBEDDING_MODEL}")
                if not OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY environment variable is not set")
                from openai import OpenAI
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                self._mode = "openai"
            else:
                print(f"Loading local embedding model: {EMBEDDING_MODEL}")
                self._model = SentenceTransformer(EMBEDDING_MODEL)
                self._mode = "local"
            print("Embedding service initialized successfully")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more texts
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            numpy array of shape (n_texts, EMBEDDING_DIMENSION)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self._mode == "openai":
            # Batching is handled by user but let's do simple batching here if needed
            # OpenAI limit is usually 2048 dimensions or specific token counts, but for list of strings it's fine
            processed_texts = [t.replace("\n", " ") for t in texts]
            response = self._client.embeddings.create(input=processed_texts, model=EMBEDDING_MODEL)
            # OpenAI returns list of objects, we need to preserve order
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)
            
        else:
            # Local model
            return self._model.encode(
                texts,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
    
    def embed_logs(self, logs: List[str]) -> np.ndarray:
        """
        Embed a list of log lines
        
        Args:
            logs: List of log line strings
            
        Returns:
            numpy array of embeddings
        """
        # Preprocess logs (remove timestamps, normalize)
        processed_logs = [self._preprocess_log(log) for log in logs]
        return self.embed(processed_logs)
    
    def _preprocess_log(self, log: str) -> str:
        """
        Preprocess a log line for embedding
        - Remove common timestamp patterns
        - Normalize whitespace
        """
        import re
        
        # Remove common timestamp patterns
        # ISO format: 2024-01-15T10:30:45
        log = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?', '', log)
        # Common log format: [15/Jan/2024:10:30:45]
        log = re.sub(r'\[\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}[^\]]*\]', '', log)
        # Unix timestamp
        log = re.sub(r'\b\d{10,13}\b', '', log)
        # Simple date/time
        log = re.sub(r'\d{2}:\d{2}:\d{2}', '', log)
        
        # Remove excessive whitespace
        log = ' '.join(log.split())
        
        return log.strip()


# Global instance for easy access
def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance"""
    return EmbeddingService()


def embed_texts(texts: Union[str, List[str]]) -> np.ndarray:
    """Convenience function to embed texts"""
    return get_embedding_service().embed(texts)


def embed_logs(logs: List[str]) -> np.ndarray:
    """Convenience function to embed logs"""
    return get_embedding_service().embed_logs(logs)
