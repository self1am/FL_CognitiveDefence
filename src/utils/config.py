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
    server_address: str = "0.0.0.0:8080"
    
@dataclass
class ClientConfig:
    """Client configuration"""
    batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
@dataclass
class AttackConfig:
    """Attack configuration"""
    enabled: bool = False
    attack_type: str = "label_flip"
    intensity: float = 0.1
    target_clients: List[int] = None  # None means random selection
    
@dataclass
class DefenseConfig:
    """Defense configuration"""
    strategy: str = "cognitive_defense"
    anomaly_threshold: float = 0.7
    reputation_decay: float = 0.8
    history_size: int = 100

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