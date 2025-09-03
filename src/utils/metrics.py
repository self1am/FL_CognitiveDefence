# src/utils/metrics.py
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MetricsCalculator:
    """Calculate various metrics for federated learning evaluation"""
    
    @staticmethod
    def calculate_model_similarity(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
        """Calculate cosine similarity between model parameters"""
        flat1 = np.concatenate([p.flatten() for p in params1])
        flat2 = np.concatenate([p.flatten() for p in params2])
        
        dot_product = np.dot(flat1, flat2)
        norms = np.linalg.norm(flat1) * np.linalg.norm(flat2)
        
        return dot_product / (norms + 1e-8)
    
    @staticmethod
    def calculate_parameter_drift(current_params: List[np.ndarray], 
                                previous_params: List[np.ndarray]) -> float:
        """Calculate L2 norm of parameter changes"""
        drift = 0.0
        for curr, prev in zip(current_params, previous_params):
            drift += np.linalg.norm(curr - prev) ** 2
        return np.sqrt(drift)
    
    @staticmethod
    def calculate_client_contribution_variance(client_updates: Dict[str, List[np.ndarray]]) -> float:
        """Calculate variance in client contributions"""
        norms = []
        for client_id, params in client_updates.items():
            norm = sum(np.linalg.norm(p) for p in params)
            norms.append(norm)
        
        return np.var(norms) if norms else 0.0
    
    @staticmethod
    def evaluate_model_performance(predictions: np.ndarray, 
                                 true_labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics"""
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='weighted', zero_division=0),
            'f1': f1_score(true_labels, predictions, average='weighted', zero_division=0)
        }