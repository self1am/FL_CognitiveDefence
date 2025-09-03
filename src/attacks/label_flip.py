# src/attacks/label_flip.py
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from .base_attack import BaseAttack

class LabelFlipAttack(BaseAttack):
    """Label flipping attack implementation"""
    
    def __init__(self, intensity: float = 0.1, target_clients=None, 
                 source_class: int = None, target_class: int = None):
        super().__init__(intensity, target_clients)
        self.source_class = source_class
        self.target_class = target_class
    
    def attack_data(self, dataset: Dataset, client_id: int) -> Dataset:
        """Apply label flipping to dataset"""
        if not self.should_attack_client(client_id):
            return dataset
        
        # Convert dataset to tensors for manipulation
        if hasattr(dataset, 'dataset'):  # Handle Subset
            base_dataset = dataset.dataset
            indices = dataset.indices
            data = torch.stack([base_dataset[i][0] for i in indices])
            labels = torch.tensor([base_dataset[i][1] for i in indices])
        else:
            data = torch.stack([dataset[i][0] for i in range(len(dataset))])
            labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        
        num_to_flip = int(len(labels) * self.intensity)
        flip_indices = np.random.choice(len(labels), num_to_flip, replace=False)
        
        flipped_labels = labels.clone()
        for idx in flip_indices:
            if self.source_class is None or labels[idx] == self.source_class:
                if self.target_class is not None:
                    flipped_labels[idx] = self.target_class
                else:
                    # Random flip to different class
                    num_classes = len(torch.unique(labels))
                    available_classes = list(range(num_classes))
                    available_classes.remove(labels[idx].item())
                    flipped_labels[idx] = np.random.choice(available_classes)
        
        self.log_attack(client_id, "label_flip", {
            'num_flipped': num_to_flip,
            'total_samples': len(labels),
            'flip_percentage': num_to_flip / len(labels)
        })
        
        return TensorDataset(data, flipped_labels)
    
    def attack_parameters(self, parameters: list[np.ndarray], client_id: int) -> list[np.ndarray]:
        """Label flip doesn't modify parameters"""
        return parameters
    
    def get_attack_description(self) -> str:
        return f"Label Flip Attack (intensity={self.intensity}, source={self.source_class}, target={self.target_class})"
