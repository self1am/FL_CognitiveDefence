#!/usr/bin/env python3
"""
Run FL server with centralized evaluation on a clean test set.
This script demonstrates the TRUE impact of attacks on the global model.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import flwr as fl

from src.models.cnn_mnist import MNISTNet
from src.datasets.mnist_handler import MNISTDataHandler
from src.server.no_defence_server import NoDefenceAggregationStrategy
from src.server.cognitive_server import CognitiveAggregationStrategy
from src.utils.config import ExperimentConfig, defenceConfig, DeterministicEnvironment
from src.utils.logging_utils import ExperimentLogger


def create_centralized_eval_fn(test_loader, device, logger):
    """Create centralized evaluation function"""
    
    def evaluate(server_round: int, parameters, config):
        """Evaluate global model on centralized clean test set"""
        try:
            # Create model
            model = MNISTNet().to(device)
            
            # Load parameters
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            # Evaluate
            model.eval()
            criterion = nn.CrossEntropyLoss()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_loss = total_loss / len(test_loader)
            accuracy = correct / total
            
            # Log with clear formatting
            logger.logger.info("=" * 80)
            logger.logger.info(
                f"🎯 ROUND {server_round} - CENTRALIZED EVALUATION (Clean Test Set)"
            )
            logger.logger.info(f"   Loss:     {avg_loss:.6f}")
            logger.logger.info(f"   Accuracy: {accuracy:.4f} ({correct}/{total} correct)")
            logger.logger.info("=" * 80)
            
            return avg_loss, {"centralized_accuracy": accuracy}
            
        except Exception as e:
            logger.logger.error(f"❌ Centralized evaluation failed: {e}")
            import traceback
            logger.logger.error(traceback.format_exc())
            return None
    
    return evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to experiment configuration YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_config = ExperimentConfig(**config.get('experiment', {}))
    defence_config = defenceConfig(**config.get('defence', {}))
    
    # Setup logging
    logger = ExperimentLogger(experiment_config.experiment_name)
    logger.logger.info(f"Starting server for experiment: {experiment_config.experiment_name}")
    
    # Setup deterministic environment
    DeterministicEnvironment.setup_seeds(experiment_config.seed)
    device = DeterministicEnvironment.get_device()
    
    # Load centralized test data
    logger.logger.info("Loading centralized test dataset...")
    data_handler = MNISTDataHandler(batch_size=64)
    _, test_loader = data_handler.create_client_dataloaders(num_clients=2, alpha=0.5)
    logger.logger.info(f"Test set loaded: {len(test_loader.dataset)} samples")
    
    # Create evaluation function
    evaluate_fn = create_centralized_eval_fn(test_loader, device, logger)
    
    # Create strategy
    if defence_config.strategy == 'cognitive_defence':
        logger.logger.info("Using Cognitive Defense Strategy")
        strategy = CognitiveAggregationStrategy(
            config=experiment_config,
            anomaly_threshold=defence_config.anomaly_threshold,
            reputation_decay=defence_config.reputation_decay,
            history_size=defence_config.history_size,
            logger=logger,
            evaluate_fn=evaluate_fn,
            min_fit_clients=experiment_config.min_clients,
            min_evaluate_clients=experiment_config.min_clients,
            min_available_clients=experiment_config.min_available_clients,
            fraction_evaluate=1.0,
        )
    else:
        logger.logger.info("Using No Defense Strategy (Simple FedAvg)")
        strategy = NoDefenceAggregationStrategy(
            config=experiment_config,
            logger=logger,
            evaluate_fn=evaluate_fn,
            min_fit_clients=experiment_config.min_clients,
            min_evaluate_clients=experiment_config.min_clients,
            min_available_clients=experiment_config.min_available_clients,
            fraction_evaluate=1.0,
        )
    
    # Start server
    logger.logger.info(f"Starting Flower server on {experiment_config.server_address}")
    logger.logger.info(f"Number of rounds: {experiment_config.num_rounds}")
    logger.logger.info("✅ Centralized evaluation ENABLED - will show TRUE global model performance")
    
    fl.server.start_server(
        server_address=experiment_config.server_address,
        config=fl.server.ServerConfig(num_rounds=experiment_config.num_rounds),
        strategy=strategy,
    )
    
    logger.logger.info("Server stopped")


if __name__ == "__main__":
    main()
