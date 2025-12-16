# src/defences/trimmed_mean_defence.py
import numpy as np
from typing import Dict, List, Tuple, Any
from .base_defence import Basedefence
from ..utils.logging_utils import ExplainableDecision

class TrimmedMeanDefenceStrategy(Basedefence):
    """
    Trimmed Mean defence strategy for Byzantine-robust aggregation.
    
    Trimmed Mean removes the largest and smallest beta fraction of values
    for each parameter dimension, then averages the remaining values.
    This provides robustness against Byzantine attacks by removing outliers.
    
    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards 
    Optimal Statistical Rates", ICML 2018
    """
    
    def __init__(self, beta: float = 0.2, **kwargs):
        """
        Initialize Trimmed Mean defence strategy.
        
        Args:
            beta: Fraction of values to trim from each end (0 < beta < 0.5)
                  E.g., beta=0.2 means trim 20% largest and 20% smallest values
        """
        super().__init__(**kwargs)
        if not (0 < beta < 0.5):
            raise ValueError(f"beta must be between 0 and 0.5, got {beta}")
        self.beta = beta
    
    def _trim_and_aggregate_parameter(self, param_values: List[np.ndarray]) -> np.ndarray:
        """
        Apply trimmed mean to a single parameter across all clients.
        
        Args:
            param_values: List of parameter arrays from different clients (same shape)
            
        Returns:
            Aggregated parameter array using trimmed mean
        """
        if not param_values:
            return None
        
        # Stack parameters along new axis (clients axis)
        stacked = np.stack(param_values, axis=0)  # Shape: (n_clients, *param_shape)
        n_clients = stacked.shape[0]
        
        # Calculate number of values to trim from each end
        n_trim = int(np.floor(n_clients * self.beta))
        
        if n_trim == 0 or n_clients - 2 * n_trim <= 0:
            # Not enough clients to trim, fall back to mean
            return np.mean(stacked, axis=0)
        
        # Sort along client axis
        sorted_params = np.sort(stacked, axis=0)
        
        # Trim from both ends
        if n_trim > 0:
            trimmed = sorted_params[n_trim:-n_trim]
        else:
            trimmed = sorted_params
        
        # Compute mean of remaining values
        return np.mean(trimmed, axis=0)
    
    def _identify_trimmed_clients(self, 
                                 client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
                                ) -> Dict[str, Dict[str, Any]]:
        """
        Identify which clients were trimmed and provide statistics.
        
        Returns:
            Dictionary mapping client_id to statistics about trimming
        """
        client_ids = list(client_updates.keys())
        n_clients = len(client_ids)
        n_trim = int(np.floor(n_clients * self.beta))
        
        # Compute average norm for each client
        client_norms = {}
        for client_id in client_ids:
            parameters, _, _ = client_updates[client_id]
            total_norm = sum(np.linalg.norm(param) for param in parameters)
            client_norms[client_id] = total_norm
        
        # Sort clients by norm
        sorted_clients = sorted(client_norms.items(), key=lambda x: x[1])
        
        # Identify trimmed clients (smallest and largest)
        trimmed_low = set(client_id for client_id, _ in sorted_clients[:n_trim])
        trimmed_high = set(client_id for client_id, _ in sorted_clients[-n_trim:]) if n_trim > 0 else set()
        
        # Build statistics
        stats = {}
        for client_id, norm in client_norms.items():
            if client_id in trimmed_low:
                stats[client_id] = {
                    'trimmed': True,
                    'reason': 'low_norm',
                    'norm': float(norm),
                    'n_trim': n_trim
                }
            elif client_id in trimmed_high:
                stats[client_id] = {
                    'trimmed': True,
                    'reason': 'high_norm',
                    'norm': float(norm),
                    'n_trim': n_trim
                }
            else:
                stats[client_id] = {
                    'trimmed': False,
                    'norm': float(norm),
                    'n_trim': n_trim
                }
        
        return stats
    
    def aggregate_updates(self, 
                         client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
                        ) -> Tuple[List[np.ndarray], List[ExplainableDecision]]:
        """
        Aggregate client updates using Trimmed Mean algorithm.
        
        The algorithm:
        1. For each parameter dimension independently:
           - Sort values from all clients
           - Remove beta fraction from top and bottom
           - Average remaining values
        2. This is applied element-wise across all parameters
        """
        if not client_updates:
            return None, []
        
        client_ids = list(client_updates.keys())
        n_clients = len(client_ids)
        
        # Get client statistics before aggregation
        trim_stats = self._identify_trimmed_clients(client_updates)
        
        # Extract parameters from all clients
        all_parameters = []
        for client_id in client_ids:
            parameters, _, _ = client_updates[client_id]
            all_parameters.append(parameters)
        
        # Aggregate each parameter using trimmed mean
        aggregated_params = []
        num_params = len(all_parameters[0])
        
        for param_idx in range(num_params):
            # Collect this parameter from all clients
            param_values = [client_params[param_idx] for client_params in all_parameters]
            
            # Apply trimmed mean
            aggregated_param = self._trim_and_aggregate_parameter(param_values)
            aggregated_params.append(aggregated_param)
        
        # Create explainable decisions
        decisions = []
        n_trim = int(np.floor(n_clients * self.beta))
        n_kept = max(n_clients - 2 * n_trim, 1)
        
        for client_id in client_ids:
            stat = trim_stats[client_id]
            
            if stat['trimmed']:
                decision = ExplainableDecision(
                    decision="partial_reject",
                    confidence=0.8,
                    reasoning=f"Client trimmed due to {stat['reason']} (norm: {stat['norm']:.2f}). "
                             f"Trimmed {n_trim} clients from each end, kept {n_kept} clients.",
                    evidence={
                        'trimmed': True,
                        'reason': stat['reason'],
                        'norm': stat['norm'],
                        'beta': self.beta,
                        'n_trim': n_trim,
                        'n_kept': n_kept,
                        'method': 'trimmed_mean'
                    }
                )
            else:
                decision = ExplainableDecision(
                    decision="accept",
                    confidence=1.0,
                    reasoning=f"Client included in trimmed mean aggregation (norm: {stat['norm']:.2f}). "
                             f"Kept {n_kept} out of {n_clients} clients.",
                    evidence={
                        'trimmed': False,
                        'norm': stat['norm'],
                        'beta': self.beta,
                        'n_trim': n_trim,
                        'n_kept': n_kept,
                        'method': 'trimmed_mean'
                    }
                )
            
            decisions.append(decision)
        
        self.increment_round()
        
        return aggregated_params, decisions
    
    def get_defence_description(self) -> str:
        return f"Trimmed Mean Defence Strategy (beta={self.beta})"
