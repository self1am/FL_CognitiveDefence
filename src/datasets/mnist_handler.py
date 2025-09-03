# src/datasets/mnist_handler.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple

class MNISTDataHandler:
    """Handle MNIST dataset loading and client distribution"""
    
    def __init__(self, data_path: str = "./data", batch_size: int = 32):
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def load_datasets(self) -> Tuple[datasets.MNIST, datasets.MNIST]:
        """Load train and test datasets"""
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
        
        return train_dataset, test_dataset
    
    def create_non_iid_split(self, dataset: datasets.MNIST, num_clients: int, 
                            alpha: float = 0.5) -> List[Subset]:
        """Create non-IID data split using Dirichlet distribution"""
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
        
    