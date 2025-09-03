# src/attacks/gradient_noise.py
import numpy as np
from .base_attack import BaseAttack
from torch.utils.data import Dataset

class GradientNoiseAttack(BaseAttack):
    """Gradient noise attack implementation"""
    
    def __init__(self, intensity: float = 0.1, target_clients=None, noise_type: str = "gaussian"):
        super().__init__(intensity, target_clients)
        self.noise_type = noise_type
    
    def attack_data(self, dataset: Dataset, client_id: int) -> Dataset:
        """Gradient noise doesn't modify data"""
        return dataset
    
    def attack_parameters(self, parameters: list[np.ndarray], client_id: int) -> list[np.ndarray]:
        """Apply noise to model parameters"""
        if not self.should_attack_client(client_id):
            return parameters
        
        noisy_params = []
        total_noise_magnitude = 0
        
        for param in parameters:
            if self.noise_type == "gaussian":
                noise = np.random.normal(0, self.intensity, param.shape)
            elif self.noise_type == "uniform":
                noise = np.random.uniform(-self.intensity, self.intensity, param.shape)
            else:
                noise = np.random.normal(0, self.intensity, param.shape)  # Default to gaussian
            
            noisy_param = param + noise.astype(param.dtype)
            noisy_params.append(noisy_param)
            total_noise_magnitude += np.linalg.norm(noise)
        
        self.log_attack(client_id, "gradient_noise", {
            'noise_type': self.noise_type,
            'total_noise_magnitude': float(total_noise_magnitude),
            'num_parameters': len(parameters)
        })
        
        return noisy_params
    
    def get_attack_description(self) -> str:
        return f"Gradient Noise Attack (intensity={self.intensity}, type={self.noise_type})"
