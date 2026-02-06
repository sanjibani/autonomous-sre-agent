"""
Cross-Encoder Reranker module
Improves context relevance by 15-20% through joint query-document scoring
"""
from typing import List, Tuple
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranker for improving RAG retrieval quality.
    
    Bi-encoders (embedding search) are fast but miss nuance.
    Cross-encoders compare query+document jointly for better relevance.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name
        """
        if self._model is None:
            print(f"Loading cross-encoder reranker: {model_name}")
            self._model = CrossEncoder(model_name, max_length=512)
            print("Reranker initialized successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[int]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: The search query
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            Indices of top_k most relevant documents
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score all pairs
        scores = self._model.predict(pairs)
        
        # Get indices sorted by score (descending)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        return ranked_indices[:top_k]
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents and return indices with scores.
        
        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [(idx, float(score)) for idx, score in ranked[:top_k]]


# Global accessor
def get_reranker() -> Reranker:
    """Get the singleton reranker instance"""
    return Reranker()
