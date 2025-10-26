# src/defences/no_defence.py
import numpy as np
from typing import Dict, List, Tuple, Any
from .base_defence import Basedefence
from ..utils.logging_utils import ExplainableDecision

class NoDefenceStrategy(Basedefence):
    """
    Simple FedAvg aggregation with no defense mechanisms.
    Used for baseline and attack-only experiments.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def aggregate_updates(self, 
                         client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
                        ) -> Tuple[List[np.ndarray], List[ExplainableDecision]]:
        """Perform simple weighted averaging of client updates"""
        
        if not client_updates:
            return None, []
        
        # Simple FedAvg: weighted average by number of samples
        weighted_updates = []
        total_samples = 0
        
        for client_id, (parameters, num_samples, metrics) in client_updates.items():
            weighted_updates.append((parameters, num_samples))
            total_samples += num_samples
        
        # Aggregate parameters
        if weighted_updates and total_samples > 0:
            aggregated_params = []
            num_params = len(weighted_updates[0][0])
            
            for param_idx in range(num_params):
                weighted_sum = sum(params[param_idx] * weight 
                                 for params, weight in weighted_updates)
                aggregated_params.append(weighted_sum / total_samples)
        else:
            aggregated_params = None
        
        # Create simple decisions for logging (all clients accepted equally)
        decisions = []
        for client_id in client_updates.keys():
            decision = ExplainableDecision(
                decision="accept",
                confidence=1.0,
                reasoning="No defense applied - simple FedAvg aggregation",
                evidence={'weight': 1.0, 'aggregation_method': 'fedavg'}
            )
            decisions.append(decision)
        
        self.increment_round()
        
        return aggregated_params, decisions
    
    def get_defence_description(self) -> str:
        return "No Defense (Simple FedAvg)"
