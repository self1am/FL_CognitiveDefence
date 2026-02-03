# src/attacks/min_max_attack.py
"""
Minimax Attack (min-max)

Game-theoretic attack that finds the optimal attack strategy assuming
the defender will respond optimally. Considers multiple defense strategies
and crafts updates that work well against all of them.

Reference: Bhagoji et al., "Analyzing Federated Learning through an 
Adversarial Lens" (ICML 2019)
"""
import numpy as np
from typing import List, Optional, Dict, Any, Callable, Tuple
from torch.utils.data import Dataset
from .adaptive_base import AdaptiveAttack


class MinMaxAttack(AdaptiveAttack):
    """
    Minimax Attack using game-theoretic optimization.
    
    Formulates attack as a two-player game:
    - Attacker: chooses malicious update to maximize damage
    - Defender: chooses aggregation strategy to minimize damage
    
    The attack finds an update that performs well under worst-case defense.
    
    Parameters:
        intensity: Base attack strength (0.0-1.0)
        defense_models: List of defense names to consider
        optimization_steps: Iterations for finding minimax solution
        threat_model_weights: Prior probabilities over defense strategies
        target_clients: List of client IDs to attack
    """
    
    def __init__(self,
                 intensity: float = 0.1,
                 defense_models: Optional[List[str]] = None,
                 optimization_steps: int = 10,
                 threat_model_weights: Optional[Dict[str, float]] = None,
                 target_clients: Optional[List[int]] = None):
        super().__init__(intensity, target_clients)
        
        # Default defense models to consider
        self.defense_models = defense_models or ['trimmed_mean', 'krum', 'median', 'mean']
        self.optimization_steps = optimization_steps
        
        # Default uniform weights over defense models
        if threat_model_weights is None:
            uniform_weight = 1.0 / len(self.defense_models)
            self.threat_model_weights = {
                defense: uniform_weight for defense in self.defense_models
            }
        else:
            self.threat_model_weights = threat_model_weights
        
        # Track which defenses we encounter
        self.observed_defenses: Dict[str, int] = {d: 0 for d in self.defense_models}
        
        # Best attack strategy found
        self.best_attack_strategy = 'sign_flip'
        self.best_attack_intensity = intensity
    
    def estimate_defense_impact(self, attack_vector: np.ndarray, 
                                defense_type: str) -> float:
        """
        Estimate how much impact an attack vector would have under a given defense.
        
        Args:
            attack_vector: The malicious update vector
            defense_type: Type of defense ('trimmed_mean', 'krum', etc.)
            
        Returns:
            Estimated impact score (higher = more effective attack)
        """
        # Simulate defense response (simplified model)
        if defense_type == 'trimmed_mean':
            # Trimmed mean defense: effectiveness depends on how much we're an outlier
            # Stay closer to mean for less detection
            magnitude = np.linalg.norm(attack_vector)
            # Trade-off: higher magnitude = more impact but easier detection
            impact = magnitude * 0.5  # Reduced impact due to trimming
            
        elif defense_type == 'krum':
            # Krum defense: selects update with smallest distance to neighbors
            # Being too far from cluster results in rejection
            magnitude = np.linalg.norm(attack_vector)
            # Exponential decay in effectiveness as we get further from cluster
            impact = magnitude * np.exp(-magnitude / self.intensity)
            
        elif defense_type == 'median':
            # Median defense: similar to trimmed mean but more robust
            magnitude = np.linalg.norm(attack_vector)
            impact = magnitude * 0.3  # Even more reduced impact
            
        elif defense_type == 'mean':
            # Simple mean aggregation: no defense
            magnitude = np.linalg.norm(attack_vector)
            impact = magnitude  # Full impact
            
        else:
            # Unknown defense: assume moderate robustness
            magnitude = np.linalg.norm(attack_vector)
            impact = magnitude * 0.5
        
        return impact
    
    def find_minimax_attack(self, base_parameters: List[np.ndarray]) -> Tuple[str, float]:
        """
        Find the minimax optimal attack strategy.
        
        For each attack strategy, evaluate worst-case performance across defenses.
        Choose the strategy with the best worst-case performance.
        
        Returns:
            (best_strategy, best_intensity)
        """
        attack_strategies = ['sign_flip', 'gradient_noise', 'scaling', 'targeted_noise']
        # Intensity range: start at 5% intensity, up to 50% or 3x base intensity (whichever is smaller)
        # These bounds ensure attacks are detectable but not so strong as to be trivially rejected
        min_intensity = 0.05  # Minimum to have measurable impact
        max_intensity = min(0.5, self.intensity * 3)  # Cap at 50% or 3x base
        intensity_levels = np.linspace(min_intensity, max_intensity, self.optimization_steps)
        
        best_worst_case_value = float('-inf')
        best_strategy = attack_strategies[0]
        best_intensity = self.intensity
        
        for strategy in attack_strategies:
            for intensity in intensity_levels:
                # Generate attack vector for this strategy
                attack_vector = self._generate_attack_vector(base_parameters[0], strategy, intensity)
                
                # Evaluate under each defense (worst-case)
                min_impact = float('inf')
                for defense in self.defense_models:
                    # Weight by threat model
                    weight = self.threat_model_weights.get(defense, 0.25)
                    impact = self.estimate_defense_impact(attack_vector, defense)
                    weighted_impact = weight * impact
                    min_impact = min(min_impact, weighted_impact)
                
                # Track best worst-case performance
                if min_impact > best_worst_case_value:
                    best_worst_case_value = min_impact
                    best_strategy = strategy
                    best_intensity = intensity
        
        return best_strategy, best_intensity
    
    def _generate_attack_vector(self, param: np.ndarray, strategy: str, intensity: float) -> np.ndarray:
        """Generate attack vector based on strategy"""
        if strategy == 'sign_flip':
            return -intensity * param
        elif strategy == 'gradient_noise':
            noise = np.random.normal(0, intensity, param.shape)
            return param + noise
        elif strategy == 'scaling':
            return param * (1 + intensity)
        elif strategy == 'targeted_noise':
            # Add noise only to high-magnitude components
            mask = np.abs(param) > np.percentile(np.abs(param), 75)
            noise = np.random.normal(0, intensity, param.shape)
            return param + noise * mask
        else:
            return param
    
    def attack_data(self, dataset: Dataset, client_id: int) -> Dataset:
        """min-max doesn't modify training data, only parameters"""
        return dataset
    
    def attack_parameters(self, parameters: List[np.ndarray], client_id: int) -> List[np.ndarray]:
        """
        Apply minimax attack to model parameters.
        
        Uses the best strategy found via minimax optimization.
        """
        if not self.should_attack_client(client_id):
            return parameters
        
        # Find best attack strategy
        self.best_attack_strategy, self.best_attack_intensity = self.find_minimax_attack(parameters)
        
        attacked_params = []
        total_magnitude = 0.0
        
        for param in parameters:
            attacked = self._generate_attack_vector(
                param, 
                self.best_attack_strategy, 
                self.best_attack_intensity
            )
            attacked_params.append(attacked.astype(param.dtype))
            total_magnitude += np.linalg.norm(attacked - param)
        
        self.log_attack(client_id, "min_max", {
            'strategy': self.best_attack_strategy,
            'intensity': self.best_attack_intensity,
            'total_magnitude': float(total_magnitude),
            'num_parameters': len(parameters),
            'defense_models': self.defense_models
        })
        
        return attacked_params
    
    def adapt_strategy(self):
        """
        Adapt threat model based on observed defense behavior.
        
        If we can infer which defense is being used, update weights.
        """
        if len(self.feedback_history) < 3:
            return
        
        # Analyze recent feedback to infer likely defense
        recent_feedback = self.get_recent_feedback(5)
        
        # Heuristic: high rejection rate suggests robust defense (Krum, Trimmed Mean)
        # Low rejection rate suggests weak defense (Mean) or successful evasion
        rejection_rate = sum(1 for f in recent_feedback if not f['accepted']) / len(recent_feedback)
        
        if rejection_rate > 0.7:
            # Likely facing robust defense, increase weight on Krum/Trimmed Mean
            self.threat_model_weights['krum'] = min(0.5, self.threat_model_weights.get('krum', 0.25) + 0.1)
            self.threat_model_weights['trimmed_mean'] = min(0.5, self.threat_model_weights.get('trimmed_mean', 0.25) + 0.1)
        elif rejection_rate < 0.3:
            # Likely facing weak defense or successful evasion
            self.threat_model_weights['mean'] = min(0.5, self.threat_model_weights.get('mean', 0.25) + 0.1)
        
        # Normalize weights
        total_weight = sum(self.threat_model_weights.values())
        self.threat_model_weights = {
            k: v / total_weight for k, v in self.threat_model_weights.items()
        }
        
        self.log_attack(-1, "min_max_adaptation", {
            'threat_model_weights': self.threat_model_weights,
            'rejection_rate': rejection_rate,
            'round': self.round_number
        })
    
    def get_attack_description(self) -> str:
        return (f"Minimax Attack "
                f"(intensity={self.best_attack_intensity:.2f}, "
                f"strategy={self.best_attack_strategy})")
