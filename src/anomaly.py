"""
Anomaly Detection module using Isolation Forest
Flags truly unusual log patterns before clustering
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Detects anomalous log entries using Isolation Forest.
    Anomalies are flagged before clustering to highlight truly unusual patterns.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, contamination: float = 0.05):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (default 5%)
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._contamination = contamination
            self._model = None
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict anomalies.
        
        Args:
            embeddings: Log embeddings (n_samples, n_features)
            
        Returns:
            Array of labels: -1 for anomalies, 1 for normal
        """
        if len(embeddings) < 10:
            # Too few samples for reliable anomaly detection
            return np.ones(len(embeddings))
        
        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )
        return self._model.fit_predict(embeddings)
    
    def get_anomaly_scores(
        self,
        embeddings: np.ndarray,
        logs: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get logs with their anomaly scores.
        
        Args:
            embeddings: Log embeddings
            logs: Original log lines
            
        Returns:
            List of dicts with log, score, and is_anomaly flag
        """
        if len(embeddings) < 10:
            return []
        
        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )
        self._model.fit(embeddings)
        scores = self._model.decision_function(embeddings)
        labels = self._model.predict(embeddings)
        
        results = []
        for i, (score, label) in enumerate(zip(scores, labels)):
            if label == -1:  # Anomaly
                results.append({
                    "log": logs[i],
                    "score": float(score),
                    "is_anomaly": True,
                    "index": i
                })
        
        # Sort by score (most anomalous first)
        return sorted(results, key=lambda x: x["score"])
    
    def get_anomaly_mask(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get a boolean mask indicating anomalies.
        
        Returns:
            Boolean array where True = anomaly
        """
        labels = self.fit_predict(embeddings)
        return labels == -1


# Global accessor
def get_anomaly_detector() -> AnomalyDetector:
    """Get the singleton anomaly detector instance"""
    return AnomalyDetector()
