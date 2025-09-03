# src/clients/enhanced_client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from ..attacks.base_attack import BaseAttack
from ..utils.config import ClientConfig, DeterministicEnvironment
from ..utils.logging_utils import ExperimentLogger

class EnhancedFLClient(fl.client.NumPyClient):
    """Enhanced Flower client with attack simulation capabilities"""
    
    def __init__(self, 
                 client_id: int,
                 model: nn.Module,
                 trainloader: DataLoader,
                 testloader: DataLoader,
                 config: ClientConfig,
                 attack: Optional[BaseAttack] = None,
                 device: torch.device = None,
                 logger: Optional[ExperimentLogger] = None):
        
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.config = config
        self.attack = attack
        self.device = device or DeterministicEnvironment.get_device()
        self.logger = logger
        
        self.model.to(self.device)
        self.training_history = []
        self.round_number = 0
        
        if self.logger:
            self.logger.logger.info(f"Client {self.client_id} initialized with {len(trainloader.dataset)} training samples")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Extract model parameters as numpy arrays"""
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        # Apply parameter-level attacks if configured
        if self.attack:
            params = self.attack.attack_parameters(params, self.client_id)
            if self.logger:
                self.logger.logger.info(f"Client {self.client_id} applied {self.attack.get_attack_description()}")
        
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Load parameters into model"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train the model on local data"""
        self.round_number += 1
        self.set_parameters(parameters)
        
        # Apply data-level attacks if configured
        if self.attack:
            attacked_dataset = self.attack.attack_data(self.trainloader.dataset, self.client_id)
            attacked_trainloader = DataLoader(
                attacked_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
        else:
            attacked_trainloader = self.trainloader
        
        # Setup optimizer
        if self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_samples = 0
        
        for epoch in range(self.config.epochs):
            for batch_idx, (images, labels) in enumerate(attacked_trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Track accuracy during training
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        training_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # Log training results
        training_log = {
            'client_id': self.client_id,
            'round': self.round_number,
            'avg_loss': avg_loss,
            'training_accuracy': training_accuracy,
            'num_samples': len(attacked_trainloader.dataset),
            'attacked': self.attack is not None,
            'attack_type': self.attack.get_attack_description() if self.attack else None,
            'timestamp': datetime.now().isoformat()
        }
        self.training_history.append(training_log)
        
        if self.logger:
            self.logger.logger.info(
                f"Client {self.client_id} Round {self.round_number} - "
                f"Loss: {avg_loss:.4f}, Accuracy: {training_accuracy:.4f}"
            )
        
        return self.get_parameters(config), len(attacked_trainloader.dataset), {
            "loss": avg_loss,
            "accuracy": training_accuracy
        }
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model on local test data"""
        self.set_parameters(parameters)
        self.model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.testloader) if len(self.testloader) > 0 else 0.0
        
        if self.logger:
            self.logger.logger.info(
                f"Client {self.client_id} Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
        
        return avg_loss, total, {"accuracy": accuracy}