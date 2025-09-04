import argparse
import yaml
import flwr as fl
from src.defenses.cognitive_defense import CognitiveDefenseStrategy
from src.server.cognitive_server import CognitiveAggregationStrategy
from src.utils.config import ExperimentConfig, DefenseConfig, DeterministicEnvironment
from src.utils.logging_utils import ExperimentLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_config = ExperimentConfig(**config.get('experiment', {}))
    defense_config = DefenseConfig(**config.get('defense', {}))
    
    # Setup deterministic environment
    DeterministicEnvironment.setup_seeds(experiment_config.seed)
    
    # Setup logging
    logger = ExperimentLogger(f"{experiment_config.experiment_name}_server")
    logger.logger.info(f"Starting server on {args.host}:{args.port}")
    
    # Create defense
    defense = CognitiveDefenseStrategy(
        anomaly_threshold=defense_config.anomaly_threshold,
        reputation_decay=defense_config.reputation_decay,
        history_size=defense_config.history_size
    )
    
    # Create strategy
    strategy = CognitiveAggregationStrategy(
        defense=defense,
        config=experiment_config,
        logger=logger,
        min_fit_clients=experiment_config.min_clients,
        min_evaluate_clients=experiment_config.min_clients,
        min_available_clients=experiment_config.min_available_clients,
    )
    
    logger.logger.info(f"Server starting with {experiment_config.num_rounds} rounds")
    
    # Start server
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=experiment_config.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
