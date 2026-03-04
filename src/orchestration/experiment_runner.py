# src/orchestration/experiment_runner.py
"""Main experiment runner"""
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import subprocess
import time
import signal
import os
import sys

from .client_orchestrator import ClientOrchestrator
from ..server.cognitive_server import CognitiveAggregationStrategy
from ..server.cognitive_server_v2 import CognitiveAggregationStrategyV2
from ..server.no_defence_server import NoDefenceAggregationStrategy
from ..server.krum_server import KrumAggregationStrategy
from ..server.trimmed_mean_server import TrimmedMeanAggregationStrategy
from ..server.vert_server import VERTAggregationStrategy
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
        from ..utils.config import DeterministicEnvironment
        
        # Setup deterministic environment
        DeterministicEnvironment.setup_seeds(self.experiment_config.seed)
        
        # Load clean test data
        self.logger.logger.info("Loading centralized test dataset for server evaluation...")
        data_handler = MNISTDataHandler(batch_size=32)
        _, test_loader = data_handler.create_client_dataloaders(num_clients=2, alpha=0.5)
        
        device = DeterministicEnvironment.get_device()
        self.logger.logger.info(f"Using device for server evaluation: {device}")
        
        def evaluate(server_round: int, parameters, config):
            """Evaluate global model on centralized test set"""
            try:
                # Create fresh model instance
                model = MNISTNet().to(device)
                
                # Load parameters into model
                params_dict = zip(model.state_dict().keys(), parameters)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                model.load_state_dict(state_dict, strict=True)
                
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
                
                self.logger.logger.info(
                    f"📊 Server Round {server_round} - CENTRALIZED EVALUATION | "
                    f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} "
                    f"(tested on {total} samples)"
                )
                
                return avg_loss, {"centralized_accuracy": accuracy}
                
            except Exception as e:
                self.logger.logger.error(f"Error in centralized evaluation: {e}")
                import traceback
                self.logger.logger.error(traceback.format_exc())
                return None
        
        return evaluate
    
    def start_server(self, run_in_main_thread: bool = False) -> subprocess.Popen:
        """Start the federated learning server
        
        Args:
            run_in_main_thread: If True, run server on main thread (blocking). 
                               If False, run in daemon thread (non-blocking).
        """
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
                fraction_evaluate=1.0,  # Evaluate on all clients for distributed metrics
            )
        elif defence_config.strategy == 'cognitive_defence_v2':
            # CogDef v2: multi-signal OODA + MAPE-K adaptive defence
            defence_raw = self.config.get('defence', {})
            strategy = CognitiveAggregationStrategyV2(
                config=self.experiment_config,
                anomaly_threshold=defence_raw.get('anomaly_threshold', 0.5),
                direction_weight=defence_raw.get('direction_weight', 0.40),
                norm_weight=defence_raw.get('norm_weight', 0.15),
                cluster_weight=defence_raw.get('cluster_weight', 0.25),
                temporal_weight=defence_raw.get('temporal_weight', 0.20),
                initial_reputation=defence_raw.get('initial_reputation', 0.5),
                recovery_rate=defence_raw.get('recovery_rate', 0.03),
                penalty_severity=defence_raw.get('penalty_severity', 0.8),
                yellow_threshold=defence_raw.get('yellow_threshold', 0.3),
                orange_threshold=defence_raw.get('orange_threshold', 0.6),
                red_threshold=defence_raw.get('red_threshold', 0.8),
                clip_multiplier=defence_raw.get('clip_multiplier', 2.0),
                trim_beta=defence_raw.get('trim_beta', 0.2),
                enable_mape_k=defence_raw.get('enable_mape_k', True),
                history_size=defence_raw.get('history_size', 100),
                logger=self.logger,
                evaluate_fn=evaluate_fn,
                min_fit_clients=self.experiment_config.min_clients,
                min_evaluate_clients=self.experiment_config.min_clients,
                min_available_clients=self.experiment_config.min_available_clients,
                fraction_evaluate=1.0,
            )
        elif defence_config.strategy == 'krum':
            # Extract Krum-specific parameters
            num_byzantine = self.config.get('defence', {}).get('num_byzantine', 2)
            multi_krum = self.config.get('defence', {}).get('multi_krum', False)
            
            strategy = KrumAggregationStrategy(
                config=self.experiment_config,
                num_byzantine=num_byzantine,
                multi_krum=multi_krum,
                logger=self.logger,
                evaluate_fn=evaluate_fn,
                min_fit_clients=self.experiment_config.min_clients,
                min_evaluate_clients=self.experiment_config.min_clients,
                min_available_clients=self.experiment_config.min_available_clients,
                fraction_evaluate=1.0,
            )
        elif defence_config.strategy == 'trimmed_mean':
            # Extract Trimmed Mean-specific parameters
            beta = self.config.get('defence', {}).get('beta', 0.2)
            
            strategy = TrimmedMeanAggregationStrategy(
                config=self.experiment_config,
                beta=beta,
                logger=self.logger,
                evaluate_fn=evaluate_fn,
                min_fit_clients=self.experiment_config.min_clients,
                min_evaluate_clients=self.experiment_config.min_clients,
                min_available_clients=self.experiment_config.min_available_clients,
                fraction_evaluate=1.0,
            )
        elif defence_config.strategy == 'vert':
            # Extract VERT-specific parameters
            kappa = self.config.get('defence', {}).get('kappa', 5)
            history_size = self.config.get('defence', {}).get('history_size', 10)
            projection_dim = self.config.get('defence', {}).get('projection_dim', 100)
            learning_rate = self.config.get('defence', {}).get('learning_rate', 0.01)
            min_history_rounds = self.config.get('defence', {}).get('min_history_rounds', 3)
            
            strategy = VERTAggregationStrategy(
                config=self.experiment_config,
                kappa=kappa,
                history_size=history_size,
                projection_dim=projection_dim,
                learning_rate=learning_rate,
                min_history_rounds=min_history_rounds,
                logger=self.logger,
                evaluate_fn=evaluate_fn,
                min_fit_clients=self.experiment_config.min_clients,
                min_evaluate_clients=self.experiment_config.min_clients,
                min_available_clients=self.experiment_config.min_available_clients,
                fraction_evaluate=1.0,
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
                fraction_evaluate=1.0,  # Evaluate on all clients for distributed metrics
            )
        
        
        self.logger.logger.info("Starting federated learning server with centralized evaluation")
        
        # Create server log file
        server_log_file = f"logs/{self.experiment_config.experiment_name}_server.log"
        Path("logs").mkdir(exist_ok=True)
        
        # Use run_server_with_eval.py as separate process
        # run_server_with_eval.py only accepts --config argument, server address comes from config file
        cmd = [
            sys.executable,
            "run_server_with_eval.py",
            "--config", self.config_path
        ]
        
        with open(server_log_file, 'w') as log_file:
            server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        # Give server time to start and bind to port
        time.sleep(3)
        
        self.logger.logger.info(f"Server logs being written to: {server_log_file}")
        self.logger.logger.info(f"Server process PID: {server_process.pid}")
        
        return server_process
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete federated learning experiment"""
        self.logger.logger.info(f"Starting experiment: {self.experiment_config.experiment_name}")
        
        # Check if server_only mode is enabled
        if self.config.get('server_only', False):
            self.logger.logger.info("Server-only mode: Running server as subprocess, waiting for external clients to connect")
            server_process = self.start_server()
            
            try:
                # Wait for server process
                server_process.wait()
            except KeyboardInterrupt:
                self.logger.logger.info("Server interrupted")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    server_process.wait()
            
            # Save experiment log
            self.logger.save_experiment_log()
            
            return {
                'total_clients': 0,
                'successful_clients': 0,
                'failed_clients': 0,
                'duration_seconds': 0,
                'mode': 'server_only'
            }
        
        # Start server in subprocess mode
        server_process = self.start_server()
        
        # Verify server is alive
        if server_process.poll() is not None:
            self.logger.logger.error("❌ Server process failed to start!")
            raise RuntimeError("Server process exited immediately. Check server logs.")
        
        self.logger.logger.info(f"✅ Server started on {self.experiment_config.server_address}")
        
        try:
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
                batch_size=batch_size,
                server_process=server_process
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
            
        finally:
            # Cleanup: terminate server process
            if server_process.poll() is None:  # Still running
                self.logger.logger.info("Terminating server process...")
                server_process.terminate()
                
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.logger.warning("Server process did not terminate gracefully, killing...")
                    server_process.kill()
                    server_process.wait()
            
            self.logger.logger.info("Cleanup complete")

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