# src/attacks/min_sum_attack.py
"""
Minimum Sum Attack (min-sum)

Crafts malicious updates that minimize the sum of distances to all benign
updates while maintaining attack effectiveness. This makes the attack appear
as a "centrist" update, highly trusted by distance-based defenses.

Reference: Baruch et al., "A Little Is Enough: Circumventing Defenses 
For Distributed Learning" (NeurIPS 2019)
"""
import numpy as np
from typing import List, Optional, Dict, Any
from torch.utils.data import Dataset
from .adaptive_base import AdaptiveAttack


class MinSumAttack(AdaptiveAttack):
    """
    Minimum Sum Attack that minimizes total distance to benign updates.
    
    The attack:
    1. Estimates the centroid of benign updates
    2. Chooses an attack direction (e.g., toward target objective)
    3. Optimizes magnitude to balance distance minimization and attack impact
    
    This makes the malicious update appear as a "consensus" among clients,
    evading distance-based defenses like Krum and Multi-Krum.
    
    Parameters:
        intensity: Attack strength (magnitude in attack direction)
        distance_weight: Balance between minimizing distance vs. maximizing impact (0-1)
        optimization_lr: Learning rate for gradient descent optimization
        max_iterations: Maximum optimization steps
        convergence_threshold: Stopping criterion for optimization
        target_clients: List of client IDs to attack
    """
    
    def __init__(self,
                 intensity: float = 0.1,
                 distance_weight: float = 0.7,
                 optimization_lr: float = 0.01,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-5,
                 target_clients: Optional[List[int]] = None):
        super().__init__(intensity, target_clients)
        self.distance_weight = distance_weight
        self.optimization_lr = optimization_lr
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Estimated centroid of benign updates
        self.benign_centroid: Optional[np.ndarray] = None
        self.benign_updates: List[np.ndarray] = []
        
        # Optimized attack magnitude
        self.optimized_magnitude = intensity
    
    def update_benign_estimates(self, benign_parameters: List[List[np.ndarray]]):
        """
        Update estimate of benign client updates.
        
        Args:
            benign_parameters: List of parameter lists from benign clients
        """
        if not benign_parameters:
            return
        
        # Store benign updates for distance calculation
        self.benign_updates = []
        for client_params in benign_parameters:
            # Flatten all parameters into single vector
            flat_params = np.concatenate([p.flatten() for p in client_params])
            self.benign_updates.append(flat_params)
        
        # Compute centroid
        if self.benign_updates:
            self.benign_centroid = np.mean(self.benign_updates, axis=0)
    
    def optimize_attack_magnitude(self, attack_direction: np.ndarray) -> float:
        """
        Optimize attack magnitude to minimize sum of distances to benign updates.
        
        Uses gradient descent to solve:
            minimize: distance_weight * Σᵢ ||centroid + α·direction - uᵢ||²
            subject to: α ≥ attack_threshold
        
        Args:
            attack_direction: Normalized direction of attack
            
        Returns:
            Optimized magnitude α
        """
        if not self.benign_updates or self.benign_centroid is None:
            return self.intensity
        
        # Initialize magnitude
        alpha = self.intensity
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Compute current attack vector
            attack_vector = self.benign_centroid + alpha * attack_direction
            
            # Compute loss: sum of squared distances
            distance_sum = 0.0
            for benign_update in self.benign_updates:
                # Ensure same size for distance calculation
                min_len = min(len(attack_vector), len(benign_update))
                if min_len != len(attack_vector) or min_len != len(benign_update):
                    # Log size mismatch warning
                    import warnings
                    warnings.warn(f"Parameter size mismatch in min-sum optimization: "
                                f"attack_vector={len(attack_vector)}, benign_update={len(benign_update)}")
                distance = np.linalg.norm(attack_vector[:min_len] - benign_update[:min_len])
                distance_sum += distance ** 2
            
            # Add penalty for low impact (encourage sufficient attack magnitude)
            impact_penalty = max(0, self.intensity - alpha) ** 2
            loss = self.distance_weight * distance_sum + (1 - self.distance_weight) * impact_penalty
            
            # Check convergence
            if abs(prev_loss - loss) < self.convergence_threshold:
                break
            
            # Compute gradient (simplified numerical gradient)
            epsilon = 1e-6
            attack_vector_plus = self.benign_centroid + (alpha + epsilon) * attack_direction
            
            distance_sum_plus = 0.0
            for benign_update in self.benign_updates:
                min_len = min(len(attack_vector_plus), len(benign_update))
                distance = np.linalg.norm(attack_vector_plus[:min_len] - benign_update[:min_len])
                distance_sum_plus += distance ** 2
            
            impact_penalty_plus = max(0, self.intensity - (alpha + epsilon)) ** 2
            loss_plus = self.distance_weight * distance_sum_plus + (1 - self.distance_weight) * impact_penalty_plus
            
            # Gradient
            gradient = (loss_plus - loss) / epsilon
            
            # Gradient descent step
            alpha = alpha - self.optimization_lr * gradient
            
            # Ensure alpha stays positive and reasonable
            alpha = max(0.01, min(1.0, alpha))
            
            prev_loss = loss
        
        return alpha
    
    def attack_data(self, dataset: Dataset, client_id: int) -> Dataset:
        """min-sum doesn't modify training data, only parameters"""
        return dataset
    
    def attack_parameters(self, parameters: List[np.ndarray], client_id: int) -> List[np.ndarray]:
        """
        Apply minimum sum attack to model parameters.
        
        Crafts update that minimizes sum of distances to benign updates
        while maintaining attack effectiveness.
        """
        if not self.should_attack_client(client_id):
            return parameters
        
        # Flatten parameters
        flat_params = np.concatenate([p.flatten() for p in parameters])
        
        # Define attack direction (sign flip toward target)
        if self.benign_centroid is not None and len(flat_params) == len(self.benign_centroid):
            # Direction from centroid
            attack_direction = -flat_params  # Sign flip as base attack
            direction_norm = np.linalg.norm(attack_direction)
            if direction_norm > 0:
                attack_direction = attack_direction / direction_norm
            
            # Optimize magnitude
            self.optimized_magnitude = self.optimize_attack_magnitude(attack_direction)
            
            # Create optimized attack vector
            if len(self.benign_centroid) == len(flat_params):
                optimized_attack_flat = self.benign_centroid + self.optimized_magnitude * attack_direction
            else:
                # Fallback if size mismatch
                optimized_attack_flat = flat_params - self.optimized_magnitude * flat_params
        else:
            # No benign centroid available, use simple sign flip
            optimized_attack_flat = flat_params - self.intensity * flat_params
            self.optimized_magnitude = self.intensity
        
        # Reshape back to original parameter shapes
        attacked_params = []
        offset = 0
        for param in parameters:
            param_size = param.size
            param_flat = optimized_attack_flat[offset:offset + param_size]
            attacked_param = param_flat.reshape(param.shape)
            attacked_params.append(attacked_param.astype(param.dtype))
            offset += param_size
        
        # Calculate total distance to benign updates
        total_distance = 0.0
        if self.benign_updates:
            for benign_update in self.benign_updates:
                min_len = min(len(optimized_attack_flat), len(benign_update))
                distance = np.linalg.norm(optimized_attack_flat[:min_len] - benign_update[:min_len])
                total_distance += distance
        
        self.log_attack(client_id, "min_sum", {
            'optimized_magnitude': float(self.optimized_magnitude),
            'total_distance_to_benign': float(total_distance),
            'num_benign_estimates': len(self.benign_updates),
            'num_parameters': len(parameters),
            'has_centroid': self.benign_centroid is not None
        })
        
        return attacked_params
    
    def adapt_strategy(self):
        """
        Adapt distance weight based on detection feedback.
        
        If detected frequently, increase distance_weight (minimize distance more).
        If accepted frequently, decrease distance_weight (maximize impact more).
        """
        if len(self.feedback_history) < 3:
            return
        
        detection_rate = self.get_detection_rate()
        
        # Adjust distance weight based on detection rate
        if detection_rate > 0.6:
            # Being detected, prioritize minimizing distance
            self.distance_weight = min(0.95, self.distance_weight + 0.05)
        elif detection_rate < 0.3:
            # Rarely detected, can prioritize impact
            self.distance_weight = max(0.3, self.distance_weight - 0.05)
        
        self.log_attack(-1, "min_sum_adaptation", {
            'new_distance_weight': self.distance_weight,
            'detection_rate': detection_rate,
            'round': self.round_number
        })
    
    def get_attack_description(self) -> str:
        return (f"Minimum Sum Attack "
                f"(magnitude={self.optimized_magnitude:.2f}, "
                f"distance_weight={self.distance_weight:.2f})")
