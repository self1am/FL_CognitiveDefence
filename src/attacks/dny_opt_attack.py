# src/attacks/dny_opt_attack.py
"""
Dynamic Optimization Attack (dny-opt)

Continuously adapts attack parameters based on real-time feedback using
reinforcement learning (Q-learning). Treats different attack strategies
as actions and learns which work best against the defense.

Reference: Shejwalkar & Houmansadr, "Manipulating the Byzantine" (NDSS 2021)
"""
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from torch.utils.data import Dataset
from .adaptive_base import AdaptiveAttack


class DnyOptAttack(AdaptiveAttack):
    """
    Dynamic Optimization Attack using reinforcement learning.
    
    Uses Q-learning to select attack intensity and technique based on
    feedback about detection and impact. Implements epsilon-greedy
    exploration to discover effective strategies.
    
    Parameters:
        intensity: Base attack strength (0.0-1.0)
        learning_rate: Q-learning update rate (default: 0.1)
        exploration_rate: Epsilon for epsilon-greedy policy (default: 0.1)
        discount_factor: Gamma for future reward discounting (default: 0.95)
        intensity_levels: List of intensity levels to choose from
        detection_threshold: Threshold to trigger defensive mode (default: 0.7)
        target_clients: List of client IDs to attack
    """
    
    def __init__(self,
                 intensity: float = 0.1,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 intensity_levels: Optional[List[float]] = None,
                 detection_threshold: float = 0.7,
                 target_clients: Optional[List[int]] = None):
        super().__init__(intensity, target_clients)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        self.detection_threshold = detection_threshold
        
        # Define action space: (intensity_level, attack_technique)
        self.intensity_levels = intensity_levels or [0.05, 0.1, 0.15, 0.2, 0.25]
        self.attack_techniques = ['sign_flip', 'gradient_noise', 'scaling']
        
        # Q-table: state -> action -> Q-value
        # State is discretized detection rate: {low, medium, high}
        self.q_table: Dict[Tuple[str, str, str], float] = {}
        self.initialize_q_table()
        
        # Current action
        self.current_intensity_level = intensity
        self.current_technique = 'sign_flip'
        
        # Track state
        self.previous_state = None
        self.previous_action = None
    
    def initialize_q_table(self):
        """Initialize Q-table with zero values for all state-action pairs"""
        states = ['low_detection', 'medium_detection', 'high_detection']
        for state in states:
            for intensity in self.intensity_levels:
                for technique in self.attack_techniques:
                    self.q_table[(state, str(intensity), technique)] = 0.0
    
    def get_state(self) -> str:
        """Discretize current detection rate into state"""
        detection_rate = self.get_detection_rate()
        
        if detection_rate < 0.3:
            return 'low_detection'
        elif detection_rate < 0.6:
            return 'medium_detection'
        else:
            return 'high_detection'
    
    def select_action(self, state: str) -> Tuple[float, str]:
        """
        Select action using epsilon-greedy policy.
        
        Returns:
            (intensity_level, attack_technique)
        """
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            intensity = np.random.choice(self.intensity_levels)
            technique = np.random.choice(self.attack_techniques)
        else:
            # Exploit: best known action
            best_value = float('-inf')
            best_intensity = self.intensity_levels[0]
            best_technique = self.attack_techniques[0]
            
            for intensity in self.intensity_levels:
                for technique in self.attack_techniques:
                    q_value = self.q_table.get((state, str(intensity), technique), 0.0)
                    if q_value > best_value:
                        best_value = q_value
                        best_intensity = intensity
                        best_technique = technique
            
            intensity = best_intensity
            technique = best_technique
        
        return intensity, technique
    
    def calculate_reward(self, was_accepted: bool) -> float:
        """
        Calculate reward based on attack outcome.
        
        Reward structure:
        - Accepted: positive reward (stealth + impact)
        - Rejected: negative reward
        - Bonus for maintaining low detection rate
        """
        if was_accepted:
            # Accepted: base reward + intensity bonus
            reward = 1.0 + (self.current_intensity_level * 2.0)
        else:
            # Rejected: penalty proportional to intensity
            reward = -2.0 - self.current_intensity_level
        
        # Bonus for maintaining stealth
        detection_rate = self.get_detection_rate()
        if detection_rate < 0.3:
            reward += 0.5
        
        return reward
    
    def update_q_value(self, state: str, action: Tuple[float, str], 
                      reward: float, next_state: str):
        """Update Q-value using Q-learning update rule"""
        intensity, technique = action
        state_action_key = (state, str(intensity), technique)
        
        # Get current Q-value
        current_q = self.q_table.get(state_action_key, 0.0)
        
        # Get max Q-value for next state
        max_next_q = max(
            self.q_table.get((next_state, str(i), t), 0.0)
            for i in self.intensity_levels
            for t in self.attack_techniques
        )
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_action_key] = new_q
    
    def attack_data(self, dataset: Dataset, client_id: int) -> Dataset:
        """dny-opt doesn't modify training data, only parameters"""
        return dataset
    
    def attack_parameters(self, parameters: List[np.ndarray], client_id: int) -> List[np.ndarray]:
        """
        Apply dynamic optimization attack to model parameters.
        
        Uses current strategy (intensity + technique) selected by Q-learning.
        """
        if not self.should_attack_client(client_id):
            return parameters
        
        attacked_params = []
        total_magnitude = 0.0
        
        for param in parameters:
            if self.current_technique == 'sign_flip':
                # Flip sign and scale
                attacked = -self.current_intensity_level * param
            elif self.current_technique == 'gradient_noise':
                # Add Gaussian noise
                noise = np.random.normal(0, self.current_intensity_level, param.shape)
                attacked = param + noise.astype(param.dtype)
            elif self.current_technique == 'scaling':
                # Scale up gradients
                attacked = param * (1 + self.current_intensity_level)
            else:
                attacked = param
            
            attacked_params.append(attacked.astype(param.dtype))
            total_magnitude += np.linalg.norm(attacked - param)
        
        self.log_attack(client_id, "dny_opt", {
            'intensity': self.current_intensity_level,
            'technique': self.current_technique,
            'total_magnitude': float(total_magnitude),
            'num_parameters': len(parameters),
            'detection_rate': self.get_detection_rate()
        })
        
        return attacked_params
    
    def adapt_strategy(self):
        """
        Adapt attack strategy using Q-learning.
        
        Updates Q-values based on feedback and selects next action.
        """
        if not self.feedback_history:
            return
        
        # Get latest feedback
        latest_feedback = self.feedback_history[-1]
        was_accepted = latest_feedback['accepted']
        
        # Get current state
        current_state = self.get_state()
        
        # Calculate reward
        reward = self.calculate_reward(was_accepted)
        
        # Update Q-value if we have a previous state-action
        if self.previous_state is not None and self.previous_action is not None:
            self.update_q_value(
                self.previous_state,
                self.previous_action,
                reward,
                current_state
            )
        
        # Select next action
        next_intensity, next_technique = self.select_action(current_state)
        
        # Update current strategy
        self.current_intensity_level = next_intensity
        self.current_technique = next_technique
        # Note: We update the base intensity to reflect current strategy,
        # but the original intensity is preserved in attack history
        self.intensity = next_intensity
        
        # Store for next update
        self.previous_state = current_state
        self.previous_action = (next_intensity, next_technique)
        
        # Log adaptation
        self.log_attack(-1, "dny_opt_adaptation", {
            'state': current_state,
            'new_intensity': next_intensity,
            'new_technique': next_technique,
            'reward': reward,
            'round': self.round_number
        })
    
    def get_attack_description(self) -> str:
        return (f"Dynamic Optimization Attack "
                f"(intensity={self.current_intensity_level:.2f}, "
                f"technique={self.current_technique})")
