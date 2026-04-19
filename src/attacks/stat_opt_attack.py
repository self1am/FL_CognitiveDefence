# src/attacks/stat_opt_attack.py
"""
Statistical Optimization Attack (stat-opt)

Crafts malicious updates that stay within statistical bounds of benign updates
to evade detection by statistical defenses (trimmed mean, Krum, median).

Reference: Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust 
Federated Learning" (USENIX Security 2020)
"""
import numpy as np
from typing import List, Optional, Dict, Any
from torch.utils.data import Dataset
from .adaptive_base import AdaptiveAttack


class StatOptAttack(AdaptiveAttack):
    """
    Statistical Optimization Attack that mimics benign update statistics.
    
    The attack computes the mean and standard deviation of benign updates,
    then crafts a malicious update that stays within k*sigma of the mean
    while maximizing attack impact.
    
    Parameters:
        intensity: Base attack strength (0.0-1.0)
        constraint_factor: Multiplier for standard deviation bound (default: 1.5)
        adaptive_learning_rate: Rate of constraint adjustment (default: 0.1)
        target_clients: List of client IDs to attack
    """
    
    def __init__(self, 
                 intensity: float = 0.1,
                 constraint_factor: float = 1.5,
                 adaptive_learning_rate: float = 0.1,
                 target_clients: Optional[List[int]] = None):
        super().__init__(intensity, target_clients)
        self.constraint_factor = constraint_factor
        self.initial_constraint_factor = constraint_factor
        self.adaptive_learning_rate = adaptive_learning_rate
        self.benign_stats: Dict[str, Any] = {}
        
    def attack_data(self, dataset: Dataset, client_id: int) -> Dataset:
        """stat-opt doesn't modify training data, only parameters"""
        return dataset
    
    def attack_parameters(self, parameters: List[np.ndarray], client_id: int) -> List[np.ndarray]:
        """
        Apply statistical optimization attack to model parameters.
        
        The attack:
        1. Generates a base malicious update (gradient sign flip)
        2. Projects it to stay within statistical bounds
        3. Logs attack details
        """
        if not self.should_attack_client(client_id):
            return parameters
        
        attacked_params = []
        total_adjustment = 0.0
        
        for param in parameters:
            # Generate base malicious update (sign flip with scaling)
            base_malicious = -self.intensity * param
            
            # If we have benign statistics, constrain to those bounds
            if self.benign_stats:
                # Project to statistical bounds
                param_mean = self.benign_stats.get('mean', 0.0)
                param_std = self.benign_stats.get('std', 1.0)
                
                # Direction from mean to malicious update
                direction = base_malicious - param_mean
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    # Normalize direction
                    direction = direction / direction_norm
                    
                    # Constrain magnitude to k*sigma
                    max_magnitude = self.constraint_factor * param_std
                    actual_magnitude = min(direction_norm, max_magnitude)
                    
                    # Craft constrained malicious update
                    constrained_malicious = param_mean + direction * actual_magnitude
                    attacked_params.append(constrained_malicious.astype(param.dtype))
                    total_adjustment += actual_magnitude
                else:
                    attacked_params.append(param)
            else:
                # No statistics available, use base malicious update
                attacked_params.append(base_malicious.astype(param.dtype))
                total_adjustment += np.linalg.norm(base_malicious)
        
        self.log_attack(client_id, "stat_opt", {
            'constraint_factor': self.constraint_factor,
            'total_adjustment': float(total_adjustment),
            'num_parameters': len(parameters),
            'has_benign_stats': bool(self.benign_stats)
        })
        
        return attacked_params
    
    def update_benign_statistics(self, benign_parameters: List[List[np.ndarray]]):
        """
        Update statistics of benign client updates.
        This should be called with parameters from known benign clients.
        
        Args:
            benign_parameters: List of parameter lists from benign clients
        """
        if not benign_parameters:
            return
        
        # Flatten all parameters
        all_params = []
        for client_params in benign_parameters:
            for param in client_params:
                all_params.append(param.flatten())
        
        if all_params:
            all_params_concat = np.concatenate(all_params)
            self.benign_stats = {
                'mean': np.mean(all_params_concat),
                'std': np.std(all_params_concat),
                'min': np.min(all_params_concat),
                'max': np.max(all_params_concat),
                'num_samples': len(benign_parameters)
            }
    
    def adapt_strategy(self):
        """
        Adapt constraint factor based on detection feedback.
        
        If detected frequently, reduce constraint factor (be more conservative).
        If accepted frequently, increase constraint factor (be more aggressive).
        """
        if len(self.feedback_history) < 3:
            return  # Need some history before adapting
        
        detection_rate = self.get_detection_rate()
        
        # Reduce constraint if detection rate is high
        if detection_rate > 0.5:
            # Being detected too often, be more conservative
            adjustment = -self.adaptive_learning_rate * self.constraint_factor
            self.constraint_factor = max(0.5, self.constraint_factor + adjustment)
        elif detection_rate < 0.2:
            # Rarely detected, can be more aggressive
            adjustment = self.adaptive_learning_rate * self.constraint_factor
            self.constraint_factor = min(3.0, self.constraint_factor + adjustment)
        
        # Log adaptation
        self.log_attack(-1, "stat_opt_adaptation", {
            'new_constraint_factor': self.constraint_factor,
            'detection_rate': detection_rate,
            'round': self.round_number
        })
    
    def get_attack_description(self) -> str:
        return (f"Statistical Optimization Attack "
                f"(intensity={self.intensity}, "
                f"constraint_factor={self.constraint_factor:.2f})")
