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
from ..defences.cognitive_defence import Cognitivedefencestrategy
from ..utils.config import ExperimentConfig, AttackConfig, DefenseConfig, ConfigManager, DeterministicEnvironment
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
    
    def start_server(self) -> subprocess.Popen:
        """Start the federated learning server"""
        # Create defense strategy
        defense_config = DefenseConfig(**self.config.get('defense', {}))
        
        if defense_config.strategy == 'cognitive_defense':
            defense = Cognitivedefencestrategy(
                anomaly_threshold=defense_config.anomaly_threshold,
                reputation_decay=defense_config.reputation_decay,
                history_size=defense_config.history_size
            )
        else:
            # Fallback to cognitive defense
            defense = Cognitivedefencestrategy()
        
        # Create aggregation strategy
        strategy = CognitiveAggregationStrategy(
            defense=defense,
            config=self.experiment_config,
            logger=self.logger,
            min_fit_clients=self.experiment_config.min_clients,
            min_evaluate_clients=self.experiment_config.min_clients,
            min_available_clients=self.experiment_config.min_available_clients,
        )
        
        self.logger.logger.info("Starting federated learning server")
        
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