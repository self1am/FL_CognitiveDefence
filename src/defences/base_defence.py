# src/defences/base_defence.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import deque
from ..utils.logging_utils import ExplainableDecision

class Basedefence(ABC):
    """Base class for all defence strategies"""
    
    def __init__(self, **kwargs):
        self.defence_history = deque(maxlen=kwargs.get('history_size', 100))
        self.client_reputation = {}
        self.round_number = 0
    
    @abstractmethod
    def aggregate_updates(self, 
                         client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
                        ) -> Tuple[List[np.ndarray], List[ExplainableDecision]]:
        """Aggregate client updates with defence mechanisms"""
        pass
    
    @abstractmethod
    def get_defence_description(self) -> str:
        """Return human-readable description of the defence"""
        pass
    
    def update_client_reputation(self, client_id: str, reputation_change: float):
        """Update client reputation score"""
        if client_id not in self.client_reputation:
            self.client_reputation[client_id] = 1.0
        
        self.client_reputation[client_id] = max(0.0, min(1.0, 
            self.client_reputation[client_id] + reputation_change))
    
    def get_client_reputation(self, client_id: str) -> float:
        """Get client reputation score"""
        return self.client_reputation.get(client_id, 1.0)
    
    def increment_round(self):
        """Increment round counter"""
        self.round_number += 1