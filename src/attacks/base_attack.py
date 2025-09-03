# src/attacks/base_attack.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.logging_utils import ExplainableDecision

class BaseAttack(ABC):
    """Base class for all attacks"""
    
    def __init__(self, intensity: float = 0.1, target_clients: Optional[List[int]] = None):
        self.intensity = intensity
        self.target_clients = target_clients or []
        self.attack_history = []
    
    @abstractmethod
    def attack_data(self, dataset: Dataset, client_id: int) -> Dataset:
        """Apply attack to training data"""
        pass
    
    @abstractmethod
    def attack_parameters(self, parameters: List[np.ndarray], client_id: int) -> List[np.ndarray]:
        """Apply attack to model parameters"""
        pass
    
    @abstractmethod
    def get_attack_description(self) -> str:
        """Return human-readable description of the attack"""
        pass
    
    def should_attack_client(self, client_id: int) -> bool:
        """Determine if this client should be attacked"""
        if not self.target_clients:
            return True  # Attack all clients if no specific targets
        return client_id in self.target_clients
    
    def log_attack(self, client_id: int, attack_type: str, details: Dict[str, Any]):
        """Log attack details"""
        self.attack_history.append({
            'client_id': client_id,
            'attack_type': attack_type,
            'intensity': self.intensity,
            'details': details,
            'timestamp': str(torch.initial_seed())  # Use for deterministic logging
        })