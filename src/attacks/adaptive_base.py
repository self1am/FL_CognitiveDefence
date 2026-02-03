# src/attacks/adaptive_base.py
"""
Base class for adaptive attacks that learn from defense mechanisms.
All adaptive attacks inherit from this class to get feedback tracking and adaptation capabilities.
"""
from abc import abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from .base_attack import BaseAttack


class AdaptiveAttack(BaseAttack):
    """
    Base class for adaptive attacks that modify their strategy based on feedback.
    
    Adaptive attacks observe defense responses and adjust their parameters to:
    1. Evade detection
    2. Maximize attack impact
    3. Maintain stealth over multiple rounds
    """
    
    def __init__(self, intensity: float = 0.1, target_clients: Optional[List[int]] = None):
        super().__init__(intensity, target_clients)
        self.feedback_history: List[Dict[str, Any]] = []
        self.round_number = 0
        self.detection_count = 0
        self.acceptance_count = 0
        
    def update_feedback(self, round_num: int, was_accepted: bool, 
                       global_accuracy: Optional[float] = None,
                       anomaly_score: Optional[float] = None,
                       additional_info: Optional[Dict[str, Any]] = None):
        """
        Update attack strategy based on feedback from the defense mechanism.
        
        Args:
            round_num: Current federated learning round number
            was_accepted: Whether the update was accepted (True) or rejected/downweighted (False)
            global_accuracy: Global model accuracy after aggregation (if available)
            anomaly_score: Anomaly score assigned by defense (if available)
            additional_info: Any additional defense-specific information
        """
        feedback = {
            'round': round_num,
            'accepted': was_accepted,
            'accuracy': global_accuracy,
            'anomaly_score': anomaly_score,
            'intensity': self.intensity,
            'additional_info': additional_info or {}
        }
        
        self.feedback_history.append(feedback)
        self.round_number = round_num
        
        if was_accepted:
            self.acceptance_count += 1
        else:
            self.detection_count += 1
            
        # Trigger adaptation
        self.adapt_strategy()
    
    @abstractmethod
    def adapt_strategy(self):
        """
        Adapt attack strategy based on accumulated feedback.
        Must be implemented by subclasses.
        """
        pass
    
    def get_detection_rate(self) -> float:
        """Calculate the fraction of updates that were detected/rejected"""
        total = self.detection_count + self.acceptance_count
        return self.detection_count / total if total > 0 else 0.0
    
    def get_acceptance_rate(self) -> float:
        """Calculate the fraction of updates that were accepted"""
        total = self.detection_count + self.acceptance_count
        return self.acceptance_count / total if total > 0 else 0.0
    
    def get_recent_feedback(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent n feedback entries"""
        return self.feedback_history[-n:] if len(self.feedback_history) >= n else self.feedback_history
    
    def reset_adaptation_state(self):
        """Reset adaptation state (useful for new experiment runs)"""
        self.feedback_history = []
        self.round_number = 0
        self.detection_count = 0
        self.acceptance_count = 0
        
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get a summary of adaptation behavior"""
        return {
            'total_rounds': self.round_number,
            'detection_rate': self.get_detection_rate(),
            'acceptance_rate': self.get_acceptance_rate(),
            'current_intensity': self.intensity,
            'feedback_count': len(self.feedback_history),
            'detection_count': self.detection_count,
            'acceptance_count': self.acceptance_count
        }
