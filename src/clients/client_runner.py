# src/clients/client_runner.py
"""Standalone client runner script for orchestration"""
import argparse
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import flwr as fl
from src.clients.enhanced_client import EnhancedFLClient
from src.models.cnn_mnist import MNISTNet
from src.datasets.mnist_handler import MNISTDataHandler
from src.attacks.label_flip import LabelFlipAttack
from src.attacks.gradient_noise import GradientNoiseAttack
from src.utils.config import ClientConfig, DeterministicEnvironment
from src.utils.logging_utils import ExperimentLogger

def create_attack(attack_config: dict):
    """Create attack instance based on configuration"""
    if not attack_config.get('attack_enabled', False):
        return None
    
    attack_type = attack_config.get('attack_type', 'label_flip')
    intensity = attack_config.get('attack_intensity', 0.1)
    
    if attack_type == 'label_flip':
        return LabelFlipAttack(intensity=intensity)
    elif attack_type == 'gradient_noise':
        return GradientNoiseAttack(intensity=intensity)
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='JSON configuration')
    args = parser.parse_args()
    
    # Parse configuration
    config = json.loads(args.config)
    client_id = config['client_id']
    server_address = config['server_address']
    
    # Setup deterministic environment
    DeterministicEnvironment.setup_seeds(config['seed'])
    device = DeterministicEnvironment.get_device()
    
    # Setup logging
    experiment_name = config.get('experiment_name', 'default')
    logger = ExperimentLogger(f"{experiment_name}_client_{client_id}")
    logger.logger.info(f"Starting client {client_id} with config: {config}")
    
    try:
        # Create model
        model = MNISTNet()
        
        # Load data
        data_handler = MNISTDataHandler(batch_size=config.get('batch_size', 32))
        client_loaders, test_loader = data_handler.create_client_dataloaders(
            num_clients=10,  # Fixed for consistency
            alpha=0.5
        )
        
        # Get client's data
        if client_id < len(client_loaders):
            train_loader = client_loaders[client_id]
        else:
            # Fallback for extra clients
            all_loaders, test_loader = data_handler.create_client_dataloaders(
                num_clients=client_id + 1, alpha=0.5
            )
            train_loader = all_loaders[client_id]
        
        # Create attack if specified
        attack = create_attack(config)
        
        # Create client configuration
        client_config = ClientConfig(
            batch_size=config.get('batch_size', 32),
            epochs=config.get('epochs', 2),
            learning_rate=config.get('learning_rate', 0.001),
            optimizer=config.get('optimizer', 'adam')
        )
        
        # Create enhanced client
        fl_client = EnhancedFLClient(
            client_id=client_id,
            model=model,
            trainloader=train_loader,
            testloader=test_loader,
            config=client_config,
            attack=attack,
            device=device,
            logger=logger
        )
        
        logger.logger.info(f"Client {client_id} connecting to {server_address}")
        
        # Start Flower client
        fl.client.start_client(
            server_address=server_address,
            client=fl_client.to_client()
        )
        
        # Save training logs
        log_file = f"client_{client_id}_training_log.json"
        with open(log_file, 'w') as f:
            json.dump(fl_client.training_history, f, indent=2)
        
        logger.logger.info(f"Client {client_id} completed successfully")
        
    except Exception as e:
        logger.logger.error(f"Client {client_id} failed: {e}")
        raise

if __name__ == "__main__":
    main()