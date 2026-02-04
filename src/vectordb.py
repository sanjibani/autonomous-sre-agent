"""
Vector Database module using ChromaDB
Manages runbooks, feedback, and incidents collections
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from .config import CHROMADB_DIR, COLLECTION_RUNBOOKS, COLLECTION_FEEDBACK, COLLECTION_INCIDENTS
from .embeddings import get_embedding_service


class VectorDBService:
    """Service for ChromaDB operations"""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            print(f"Initializing ChromaDB at {CHROMADB_DIR}")
            self._client = chromadb.PersistentClient(
                path=str(CHROMADB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            self._embedding_service = get_embedding_service()
            self._init_collections()
            print("ChromaDB initialized successfully")
    
    def _init_collections(self):
        """Initialize all required collections"""
        self.runbooks = self._client.get_or_create_collection(
            name=COLLECTION_RUNBOOKS,
            metadata={"description": "Runbooks and troubleshooting guides"}
        )
        self.feedback = self._client.get_or_create_collection(
            name=COLLECTION_FEEDBACK,
            metadata={"description": "Human feedback on incidents"}
        )
        self.incidents = self._client.get_or_create_collection(
            name=COLLECTION_INCIDENTS,
            metadata={"description": "Past resolved incidents"}
        )
    
    def add_runbooks(self, runbooks: List[Dict[str, str]]) -> int:
        """
        Add runbooks to the vector store
        
        Args:
            runbooks: List of dicts with 'title', 'content', 'category' keys
            
        Returns:
            Number of runbooks added
        """
        if not runbooks:
            return 0
        
        ids = [f"runbook_{i}" for i in range(len(runbooks))]
        documents = [f"{r['title']}\n\n{r['content']}" for r in runbooks]
        metadatas = [{"title": r["title"], "category": r.get("category", "general")} for r in runbooks]
        
        # Generate embeddings
        embeddings = self._embedding_service.embed(documents).tolist()
        
        # Upsert to handle duplicates
        self.runbooks.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return len(runbooks)
    
    def search_runbooks(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant runbooks
        
        Args:
            query: Search query (e.g., error description)
            n_results: Number of results to return
            
        Returns:
            List of matching runbooks with similarity scores
        """
        query_embedding = self._embedding_service.embed(query).tolist()
        
        results = self.runbooks.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"][0]:
            return []
        
        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def add_feedback(
        self,
        incident_id: str,
        cluster_summary: str,
        original_recommendation: str,
        was_correct: bool,
        correct_answer: Optional[str] = None
    ) -> str:
        """
        Store human feedback for learning
        
        Args:
            incident_id: ID of the incident
            cluster_summary: Summary of the log cluster
            original_recommendation: The agent's recommendation
            was_correct: Whether the recommendation was correct
            correct_answer: If wrong, the correct answer
            
        Returns:
            Feedback ID
        """
        feedback_id = f"feedback_{incident_id}"
        
        # Create the feedback document
        if was_correct:
            document = f"CONFIRMED: {cluster_summary}\nResolution: {original_recommendation}"
        else:
            document = f"CORRECTED: {cluster_summary}\nWrong: {original_recommendation}\nCorrect: {correct_answer}"
        
        embedding = self._embedding_service.embed(document).tolist()
        
        self.feedback.upsert(
            ids=[feedback_id],
            documents=[document],
            embeddings=embedding,
            metadatas=[{
                "incident_id": incident_id,
                "was_correct": was_correct,
                "timestamp": str(__import__('datetime').datetime.now())
            }]
        )
        
        return feedback_id
    
    def search_feedback(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search past feedback for similar issues"""
        query_embedding = self._embedding_service.embed(query).tolist()
        
        results = self.feedback.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"][0]:
            return []
        
        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get counts for all collections"""
        return {
            "runbooks": self.runbooks.count(),
            "feedback": self.feedback.count(),
            "incidents": self.incidents.count()
        }


# Global instance
def get_vectordb_service() -> VectorDBService:
    """Get the singleton VectorDB service instance"""
    return VectorDBService()
