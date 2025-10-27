# src/orchestration/experiment_runner.py
"""Main experiment runner"""
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import subprocess
import time

from .client_orchestrator import ClientOrchestrator
from ..server.cognitive_server import CognitiveAggregationStrategy
from ..server.no_defence_server import NoDefenceAggregationStrategy
from ..utils.config import ExperimentConfig, AttackConfig, defenceConfig, ConfigManager, DeterministicEnvironment
from ..utils.logging_utils import ExperimentLogger
import flwr as fl

class ExperimentRunner:
    """Main experiment runner coordinating server and clients"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.experiment_config = ExperimentConfig(**self.config.get('experiment', {}))
        self.logger = ExperimentLogger(self.experiment_config.experiment_name)
        
        # Setup deterministic environment
        DeterministicEnvironment.setup_seeds(self.experiment_config.seed)
    
    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration from YAML"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_attack_configs(self) -> Dict[int, AttackConfig]:
        """Create attack configurations for specified clients"""
        attack_configs = {}
        
        if 'attacks' in self.config:
            for attack_spec in self.config['attacks']:
                attack_config = AttackConfig(**attack_spec)
                target_clients = attack_config.target_clients or []
                
                for client_id in target_clients:
                    attack_configs[client_id] = attack_config
        
        return attack_configs
    
    def create_centralized_eval_fn(self):
        """Create centralized evaluation function for server"""
        from ..models.cnn_mnist import MNISTNet
        from ..datasets.mnist_handler import MNISTDataHandler
        import torch
        import torch.nn as nn
        
        # Load clean test data
        data_handler = MNISTDataHandler(batch_size=32)
        _, test_loader = data_handler.create_client_dataloaders(num_clients=2, alpha=0.5)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNISTNet().to(device)
        criterion = nn.CrossEntropyLoss()
        
        def evaluate(server_round: int, parameters, config):
            """Evaluate global model on centralized test set"""
            # Load parameters into model
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            model.eval()
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
            
            self.logger.logger.info(
                f"Server Round {server_round} - Centralized Test Loss: {avg_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}"
            )
            
            return avg_loss, {"accuracy": accuracy}
        
        return evaluate
    
    def start_server(self) -> subprocess.Popen:
        """Start the federated learning server"""
        # Create centralized evaluation function
        evaluate_fn = self.create_centralized_eval_fn()
        
        # Create aggregation strategy based on configuration
        defence_config = defenceConfig(**self.config.get('defence', {}))
        
        if defence_config.strategy == 'cognitive_defence':
            strategy = CognitiveAggregationStrategy(
                config=self.experiment_config,
                anomaly_threshold=defence_config.anomaly_threshold,
                reputation_decay=defence_config.reputation_decay,
                history_size=defence_config.history_size,
                logger=self.logger,
                evaluate_fn=evaluate_fn,
                min_fit_clients=self.experiment_config.min_clients,
                min_evaluate_clients=self.experiment_config.min_clients,
                min_available_clients=self.experiment_config.min_available_clients,
            )
        else:
            # No defense or unknown strategy - use simple FedAvg
            strategy = NoDefenceAggregationStrategy(
                config=self.experiment_config,
                logger=self.logger,
                evaluate_fn=evaluate_fn,
                min_fit_clients=self.experiment_config.min_clients,
                min_evaluate_clients=self.experiment_config.min_clients,
                min_available_clients=self.experiment_config.min_available_clients,
            )
        
        self.logger.logger.info("Starting federated learning server with centralized evaluation")
        
        # Start server in separate process
        def run_server():
            fl.server.start_server(
                server_address=self.experiment_config.server_address,
                config=fl.server.ServerConfig(num_rounds=self.experiment_config.num_rounds),
                strategy=strategy,
            )
        
        import threading
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Give server time to start
        time.sleep(5)
        
        return server_thread
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete federated learning experiment"""
        self.logger.logger.info(f"Starting experiment: {self.experiment_config.experiment_name}")
        
        # Start server
        server_thread = self.start_server()
        
        # Create client orchestrator
        orchestrator = ClientOrchestrator(
            server_address=self.experiment_config.server_address,
            experiment_config=self.experiment_config,
            logger=self.logger,
            max_memory_mb=self.config.get('orchestration', {}).get('max_memory_mb', 6000)
        )
        
        # Get attack configurations
        attack_configs = self.create_attack_configs()
        
        # Run multi-client experiment
        num_clients = self.config.get('orchestration', {}).get('num_clients', 10)
        batch_size = self.config.get('orchestration', {}).get('batch_size', 3)
        
        experiment_results = orchestrator.run_experiment(
            num_clients=num_clients,
            attack_configs=attack_configs,
            batch_size=batch_size
        )
        
        # Save complete experiment log
        self.logger.save_experiment_log()
        
        # Save experiment results
        results_file = f"experiments/results/{self.experiment_config.experiment_name}_results.json"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        self.logger.logger.info(f"Experiment completed. Results saved to {results_file}")
        
        return experiment_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment configuration YAML')
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    results = runner.run_experiment()
    
    print(f"Experiment completed successfully!")
    print(f"Total clients: {results['total_clients']}")
    print(f"Successful clients: {results['successful_clients']}")
    print(f"Duration: {results['duration_seconds']:.2f} seconds")

if __name__ == "__main__":
    main()