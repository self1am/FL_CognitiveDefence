# src/utils/config.py
import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import torch
import numpy as np
import random

@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    experiment_name: str
    seed: int = 42
    num_rounds: int = 10
    min_clients: int = 2
    min_available_clients: int = 2
    server_address: str = "127.0.0.1:8080"
    
@dataclass
class ClientConfig:
    """Client configuration"""
    batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
@dataclass
class AttackConfig:
    """Attack configuration with support for adaptive attack parameters"""
    enabled: bool = False
    attack_type: str = "label_flip"
    intensity: float = 0.1
    target_clients: List[int] = None  # None means random selection
    
    # Parameters for stat-opt attack
    constraint_factor: float = 1.5
    adaptive_learning_rate: float = 0.1
    
    # Parameters for dny-opt attack
    learning_rate: float = 0.1
    exploration_rate: float = 0.1
    discount_factor: float = 0.95
    detection_threshold: float = 0.7
    
    # Parameters for min-max attack
    defense_models: List[str] = None
    optimization_steps: int = 10
    threat_model_weights: Dict[str, float] = None
    
    # Parameters for min-sum attack
    distance_weight: float = 0.7
    optimization_lr: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 1e-5
    
    def __post_init__(self):
        """Initialize default values for lists and dicts"""
        if self.target_clients is None:
            self.target_clients = []
        if self.defense_models is None:
            self.defense_models = ['krum', 'trimmed_mean']
        if self.threat_model_weights is None:
            # Initialize with uniform weights over defense models
            uniform_weight = 1.0 / len(self.defense_models)
            self.threat_model_weights = {d: uniform_weight for d in self.defense_models}
    
@dataclass
class defenceConfig:
    """defence configuration - supports all defense strategies"""
    strategy: str = "cognitive_defence"
    
    # Cognitive defense parameters
    anomaly_threshold: float = 0.7
    reputation_decay: float = 0.8
    history_size: int = 100
    
    # Krum defense parameters
    num_byzantine: int = 2
    multi_krum: bool = False
    
    # Trimmed Mean defense parameters
    beta: float = 0.2
    
    # VERT defense parameters
    kappa: int = 5
    projection_dim: int = 100
    learning_rate: float = 0.01
    min_history_rounds: int = 3

class DeterministicEnvironment:
    """Ensures deterministic behavior across experiments"""
    
    @staticmethod
    def setup_seeds(seed: int = 42):
        """Set seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    @staticmethod
    def get_device():
        """Get appropriate device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        else:
            return torch.device("cpu")

class ConfigManager:
    """Manage experiment configurations"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def create_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Create ExperimentConfig from dictionary"""
        return ExperimentConfig(**config_dict.get('experiment', {}))