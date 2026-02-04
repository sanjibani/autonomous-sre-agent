"""
Agent module - Core logic for analyzing incidents and generating recommendations
Uses Ollama for LLM-based reasoning
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
import uuid
from datetime import datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .config import OLLAMA_MODEL, SEVERITY_THRESHOLDS
from .clustering import ClusterInfo
from .rag import get_rag_service, RAGContext


@dataclass
class Recommendation:
    """Agent's recommendation for an incident"""
    incident_id: str
    severity: str  # high, medium, low
    confidence: float  # 0.0 to 1.0
    root_cause: str
    recommendation: str
    evidence: List[str]
    cluster_summary: str
    timestamp: str
    retrieved_context: List[str]  # For evaluation/faithfulness checks
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SREAgent:
    """
    Autonomous SRE Agent
    Analyzes log clusters and generates recommendations using RAG + LLM
    """
    
    def __init__(self):
        self.rag_service = get_rag_service()
        self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        if not OLLAMA_AVAILABLE:
            print("WARNING: Ollama package not installed. Using fallback mode.")
            return False
        
        try:
            # Try to list models to check connectivity
            ollama.list()
            return True
        except Exception as e:
            print(f"WARNING: Ollama not available: {e}")
            print("Falling back to rule-based recommendations.")
            return False
    
    def analyze_incident(self, cluster_info: ClusterInfo) -> Recommendation:
        """
        Analyze a cluster and generate a recommendation
        
        Args:
            cluster_info: Information about the log cluster
            
        Returns:
            Recommendation with root cause analysis and suggested fix
        """
        # Generate incident ID
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        # Get cluster summary
        cluster_summary = self._summarize_cluster(cluster_info)
        
        # Retrieve relevant context via RAG
        rag_context = self.rag_service.retrieve_context(cluster_info)
        
        # Extract raw context strings for evaluation
        context_strings = []
        for match in rag_context.runbook_matches:
            if match.get("similarity", 0) > 0.3:
                context_strings.append(match.get("document", ""))
        
        # Generate recommendation using LLM or fallback
        if OLLAMA_AVAILABLE:
            try:
                result = self._analyze_with_llm(cluster_info, rag_context)
            except Exception as e:
                print(f"LLM analysis failed: {e}. Using fallback.")
                result = self._analyze_with_rules(cluster_info, rag_context)
        else:
            result = self._analyze_with_rules(cluster_info, rag_context)
        
        # Calculate confidence
        base_confidence = result.get("confidence", 0.6)
        final_confidence = min(1.0, base_confidence + rag_context.confidence_boost)
        
        # Determine severity
        severity = self._determine_severity(cluster_info, final_confidence)
        
        return Recommendation(
            incident_id=incident_id,
            severity=severity,
            confidence=round(final_confidence, 2),
            root_cause=result.get("root_cause", "Unknown"),
            recommendation=result.get("recommendation", "Investigate further"),
            evidence=result.get("evidence", cluster_info.representative_logs[:3]),
            cluster_summary=cluster_summary,
            timestamp=datetime.now().isoformat(),
            retrieved_context=context_strings
        )
    
    def _summarize_cluster(self, cluster_info: ClusterInfo) -> str:
        """Create a human-readable cluster summary"""
        keywords = ", ".join(cluster_info.error_keywords[:5]) if cluster_info.error_keywords else "various issues"
        return f"{cluster_info.size} log entries related to: {keywords}"
    
    def _analyze_with_llm(
        self,
        cluster_info: ClusterInfo,
        rag_context: RAGContext
    ) -> Dict[str, Any]:
        """Use Ollama LLM for analysis"""
        
        # Build the prompt
        prompt = self._build_analysis_prompt(cluster_info, rag_context)
        
        # Call Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert SRE (Site Reliability Engineer) analyzing production incidents.
Your task is to:
1. Identify the most likely root cause based on the logs and context provided
2. Suggest a clear, actionable resolution
3. Provide a confidence score (0.0 to 1.0) based on how certain you are

Respond in JSON format:
{
    "root_cause": "Brief description of the root cause",
    "recommendation": "Step-by-step resolution",
    "confidence": 0.X,
    "evidence": ["Key log line 1", "Key log line 2"]
}"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json"
        )
        
        # Parse response
        try:
            result = json.loads(response['message']['content'])
            return result
        except json.JSONDecodeError:
            # Try to extract info from non-JSON response
            content = response['message']['content']
            return {
                "root_cause": content[:200],
                "recommendation": "See full analysis above",
                "confidence": 0.5,
                "evidence": cluster_info.representative_logs[:3]
            }
    
    def _build_analysis_prompt(
        self,
        cluster_info: ClusterInfo,
        rag_context: RAGContext
    ) -> str:
        """Build the prompt for LLM analysis"""
        parts = [
            "## Incident Analysis Request",
            "",
            f"**Cluster Size:** {cluster_info.size} log entries",
            f"**Keywords:** {', '.join(cluster_info.error_keywords[:10])}",
            f"**Initial Severity Assessment:** {cluster_info.severity_hint}",
            "",
            "### Sample Log Lines:",
        ]
        
        for i, log in enumerate(cluster_info.representative_logs[:5], 1):
            # Truncate very long logs
            log_display = log[:300] if len(log) > 300 else log
            parts.append(f"{i}. {log_display}")
        
        if rag_context.combined_context:
            parts.extend([
                "",
                "### Relevant Knowledge Base Context:",
                rag_context.combined_context
            ])
        
        parts.extend([
            "",
            "Based on the above information, analyze this incident and provide your recommendation."
        ])
        
        return "\n".join(parts)
    
    def _analyze_with_rules(
        self,
        cluster_info: ClusterInfo,
        rag_context: RAGContext
    ) -> Dict[str, Any]:
        """Fallback rule-based analysis when LLM is not available"""
        
        keywords = set(k.lower() for k in cluster_info.error_keywords)
        logs_text = " ".join(cluster_info.representative_logs).lower()
        
        # Rule-based root cause detection
        root_cause = "Unknown error pattern"
        recommendation = "Investigate the logs manually"
        confidence = 0.4
        
        # Connection issues
        if any(kw in keywords for kw in ['connection', 'refused', 'timeout', 'socket']):
            root_cause = "Network connectivity or service availability issue"
            recommendation = "1. Check if the target service is running\n2. Verify network connectivity\n3. Check firewall rules\n4. Review service logs"
            confidence = 0.7
        
        # Disk/space issues
        elif any(kw in keywords for kw in ['disk', 'space', 'full', 'storage']):
            root_cause = "Disk space exhaustion"
            recommendation = "1. Check disk usage: df -h\n2. Identify large files: du -sh /*\n3. Clean up logs and temp files\n4. Consider expanding storage"
            confidence = 0.8
        
        # Memory issues
        elif any(kw in keywords for kw in ['memory', 'heap', 'outofmemory', 'oom']):
            root_cause = "Memory exhaustion"
            recommendation = "1. Check memory usage: free -h\n2. Identify memory-hungry processes\n3. Increase heap size if applicable\n4. Check for memory leaks"
            confidence = 0.75
        
        # Block/replication issues (HDFS specific)
        elif any(kw in keywords for kw in ['block', 'replica', 'datanode']):
            root_cause = "HDFS block replication issue"
            recommendation = "1. Check cluster health: hdfs dfsadmin -report\n2. Identify under-replicated blocks\n3. Check DataNode status\n4. Run: hdfs fsck / -files -blocks"
            confidence = 0.7
        
        # NameNode issues
        elif 'namenode' in logs_text or 'safemode' in logs_text:
            root_cause = "HDFS NameNode issue"
            recommendation = "1. Check NameNode status\n2. Check if in safe mode: hdfs dfsadmin -safemode get\n3. Review NameNode logs\n4. Check heap usage"
            confidence = 0.7
        
        # Boost confidence if we have good RAG matches
        if rag_context.confidence_boost > 0.1:
            confidence += 0.1
            # Try to extract recommendation from runbook
            if rag_context.runbook_matches:
                top_match = rag_context.runbook_matches[0]
                if top_match.get("similarity", 0) > 0.5:
                    recommendation = f"Based on runbook '{top_match['metadata'].get('title', 'Unknown')}':\n{recommendation}"
        
        return {
            "root_cause": root_cause,
            "recommendation": recommendation,
            "confidence": confidence,
            "evidence": cluster_info.representative_logs[:3]
        }
    
    def _determine_severity(
        self,
        cluster_info: ClusterInfo,
        confidence: float
    ) -> str:
        """Determine incident severity"""
        # Use cluster hint as base
        base_severity = cluster_info.severity_hint
        
        # Adjust based on cluster size
        if cluster_info.size > 1000:
            return "high"
        elif cluster_info.size > 100:
            base_severity = "high" if base_severity == "medium" else base_severity
        
        return base_severity


# Global instance
_agent = None

def get_agent() -> SREAgent:
    """Get the SRE Agent instance"""
    global _agent
    if _agent is None:
        _agent = SREAgent()
    return _agent


def analyze_cluster(cluster_info: ClusterInfo) -> Recommendation:
    """Convenience function to analyze a cluster"""
    return get_agent().analyze_incident(cluster_info)
