# src/datasets/mnist_handler.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict, Optional
import threading
import hashlib

class MNISTDataHandler:
    """Handle MNIST dataset loading and client distribution with global caching"""
    
    # Class-level cache for shared datasets and splits
    _cache: Dict[str, any] = {}
    _cache_lock = threading.Lock()
    
    def __init__(self, data_path: str = "./data", batch_size: int = 32):
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def load_datasets(self) -> Tuple[datasets.MNIST, datasets.MNIST]:
        """Load train and test datasets with caching"""
        cache_key = f"datasets_{self.data_path}"
        
        # Check cache first
        with MNISTDataHandler._cache_lock:
            if cache_key in MNISTDataHandler._cache:
                return MNISTDataHandler._cache[cache_key]
        
        # Load if not cached
        train_dataset = datasets.MNIST(
            self.data_path, 
            train=True, 
            download=True, 
            transform=self.transform
        )
        
        test_dataset = datasets.MNIST(
            self.data_path, 
            train=False, 
            download=True, 
            transform=self.transform
        )
        
        result = (train_dataset, test_dataset)
        
        # Store in cache
        with MNISTDataHandler._cache_lock:
            MNISTDataHandler._cache[cache_key] = result
        
        return result
    
    def create_non_iid_split(self, dataset: datasets.MNIST, num_clients: int, 
                            alpha: float = 0.5) -> List[Subset]:
        """Create non-IID data split using Dirichlet distribution with caching"""
        # Create cache key based on dataset size, num_clients, and alpha
        cache_key = f"split_{len(dataset)}_{num_clients}_{alpha}"
        
        # Check cache first
        with MNISTDataHandler._cache_lock:
            if cache_key in MNISTDataHandler._cache:
                return MNISTDataHandler._cache[cache_key]
        
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        num_classes = len(np.unique(labels))
        
        # Use Dirichlet distribution for non-IID allocation
        proportions = np.random.dirichlet(alpha * np.ones(num_classes), num_clients)
        
        client_datasets = []
        for client in range(num_clients):
            client_indices = []
            for class_id in range(num_classes):
                class_indices = np.where(labels == class_id)[0]
                num_samples = int(proportions[client][class_id] * len(class_indices))
                if num_samples > 0:
                    selected_indices = np.random.choice(
                        class_indices, num_samples, replace=False
                    )
                    client_indices.extend(selected_indices)
            
            if client_indices:  # Ensure client has some data
                client_datasets.append(Subset(dataset, client_indices))
            else:
                # Fallback: give at least some random samples
                fallback_indices = np.random.choice(len(dataset), 100, replace=False)
                client_datasets.append(Subset(dataset, fallback_indices))
        
        # Store in cache
        with MNISTDataHandler._cache_lock:
            MNISTDataHandler._cache[cache_key] = client_datasets
        
        return client_datasets
    
    def create_client_dataloaders(self, num_clients: int, alpha: float = 0.5) -> Tuple[List[DataLoader], DataLoader]:
        """Create dataloaders for clients and test set"""
        train_dataset, test_dataset = self.load_datasets()
        
        # Create client datasets
        client_datasets = self.create_non_iid_split(train_dataset, num_clients, alpha)
        
        # Create dataloaders
        client_loaders = [
            DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for dataset in client_datasets
        ]
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return client_loaders, test_loader
    
    @classmethod
    def clear_cache(cls):
        """Clear the global cache - useful for cleanup between experiments"""
        with cls._cache_lock:
            cls._cache.clear()
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, str]:
        """Get information about cached items"""
        with cls._cache_lock:
            return {k: type(v).__name__ for k, v in cls._cache.items()}
