"""
RAG (Retrieval Augmented Generation) module
Retrieves relevant context from runbooks and past incidents
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .vectordb import get_vectordb_service
from .clustering import ClusterInfo


@dataclass
class RAGContext:
    """Context retrieved for incident analysis"""
    runbook_matches: List[Dict[str, Any]]
    feedback_matches: List[Dict[str, Any]]
    combined_context: str
    confidence_boost: float  # How much the context should boost confidence


class RAGService:
    """Service for RAG-based context retrieval"""
    
    def __init__(self):
        self.vectordb = get_vectordb_service()
    
    def retrieve_context(
        self,
        cluster_info: ClusterInfo,
        n_runbooks: int = 3,
        n_feedback: int = 2
    ) -> RAGContext:
        """
        Retrieve relevant context for a cluster
        
        Args:
            cluster_info: Information about the log cluster
            n_runbooks: Number of runbooks to retrieve
            n_feedback: Number of past feedback entries to retrieve
            
        Returns:
            RAGContext with relevant information
        """
        # Build query from cluster info
        query = self._build_query(cluster_info)
        
        # Search runbooks
        runbook_matches = self.vectordb.search_runbooks(query, n_results=n_runbooks)
        
        # Search past feedback
        feedback_matches = self.vectordb.search_feedback(query, n_results=n_feedback)
        
        # Build combined context
        combined_context = self._build_combined_context(runbook_matches, feedback_matches)
        
        # Calculate confidence boost based on match quality
        confidence_boost = self._calculate_confidence_boost(runbook_matches, feedback_matches)
        
        return RAGContext(
            runbook_matches=runbook_matches,
            feedback_matches=feedback_matches,
            combined_context=combined_context,
            confidence_boost=confidence_boost
        )
    
    def _build_query(self, cluster_info: ClusterInfo) -> str:
        """Build a search query from cluster information"""
        parts = []
        
        # Add keywords
        if cluster_info.error_keywords:
            parts.append(f"Error keywords: {', '.join(cluster_info.error_keywords[:5])}")
        
        # Add representative logs (first 2)
        for log in cluster_info.representative_logs[:2]:
            # Truncate long logs
            log_snippet = log[:200] if len(log) > 200 else log
            parts.append(log_snippet)
        
        return "\n".join(parts)
    
    def _build_combined_context(
        self,
        runbook_matches: List[Dict[str, Any]],
        feedback_matches: List[Dict[str, Any]]
    ) -> str:
        """Combine all matches into a single context string"""
        parts = []
        
        # Add runbook context
        if runbook_matches:
            parts.append("=== RELEVANT RUNBOOKS ===")
            for match in runbook_matches:
                title = match.get("metadata", {}).get("title", "Unknown")
                similarity = match.get("similarity", 0)
                # Only include high-relevance matches
                if similarity > 0.3:
                    parts.append(f"\n[Runbook: {title}] (Relevance: {similarity:.0%})")
                    # Truncate long content
                    content = match.get("document", "")
                    if len(content) > 500:
                        content = content[:500] + "..."
                    parts.append(content)
        
        # Add feedback context
        if feedback_matches:
            parts.append("\n=== PAST INCIDENTS ===")
            for match in feedback_matches:
                similarity = match.get("similarity", 0)
                if similarity > 0.3:
                    was_correct = match.get("metadata", {}).get("was_correct", True)
                    status = "✓ Confirmed" if was_correct else "✗ Corrected"
                    parts.append(f"\n[Past Incident: {status}] (Relevance: {similarity:.0%})")
                    content = match.get("document", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    parts.append(content)
        
        return "\n".join(parts) if parts else "No relevant context found in knowledge base."
    
    def _calculate_confidence_boost(
        self,
        runbook_matches: List[Dict[str, Any]],
        feedback_matches: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate how much the context should boost confidence
        
        High similarity matches = higher confidence boost
        Past confirmed feedback = higher boost
        Past corrected feedback = be careful, slightly lower
        """
        boost = 0.0
        
        # Runbook matches contribute up to 0.15
        if runbook_matches:
            top_similarity = max(m.get("similarity", 0) for m in runbook_matches)
            boost += min(0.15, top_similarity * 0.2)
        
        # Feedback matches contribute up to 0.1
        if feedback_matches:
            for match in feedback_matches:
                similarity = match.get("similarity", 0)
                was_correct = match.get("metadata", {}).get("was_correct", True)
                
                if was_correct and similarity > 0.5:
                    boost += 0.05  # Confirmed similar case
                elif not was_correct and similarity > 0.5:
                    boost -= 0.02  # Similar case was wrong before, be careful
        
        return max(0.0, min(0.2, boost))  # Cap at 0.2


# Global instance
_rag_service = None

def get_rag_service() -> RAGService:
    """Get the RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
