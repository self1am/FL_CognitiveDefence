# run_single_client.py - Run a single client manually
import argparse
import torch
import flwr as fl
from src.clients.enhanced_client import EnhancedFLClient
from src.models.cnn_mnist import MNISTNet
from src.datasets.mnist_handler import MNISTDataHandler
from src.attacks.label_flip import LabelFlipAttack
from src.attacks.gradient_noise import GradientNoiseAttack
from src.utils.config import ClientConfig, DeterministicEnvironment
from src.utils.logging_utils import ExperimentLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', type=int, required=True, help='Client ID')
    parser.add_argument('--server', type=str, required=True, help='Server address (IP:PORT)')
    parser.add_argument('--attack', type=str, choices=['none', 'label_flip', 'gradient_noise'], default='none')
    parser.add_argument('--attack-intensity', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup deterministic environment
    client_seed = args.seed + args.client_id
    DeterministicEnvironment.setup_seeds(client_seed)
    device = DeterministicEnvironment.get_device()
    
    # Setup logging
    logger = ExperimentLogger(f"manual_client_{args.client_id}")
    logger.logger.info(f"Starting client {args.client_id} connecting to {args.server}")
    
    # Create model
    model = MNISTNet()
    
    # Load data
    data_handler = MNISTDataHandler(batch_size=32)
    client_loaders, test_loader = data_handler.create_client_dataloaders(
        num_clients=10, alpha=0.5
    )
    
    # Get client's data (with fallback)
    if args.client_id < len(client_loaders):
        train_loader = client_loaders[args.client_id]
    else:
        # Create more clients if needed
        client_loaders, test_loader = data_handler.create_client_dataloaders(
            num_clients=args.client_id + 1, alpha=0.5
        )
        train_loader = client_loaders[args.client_id]
    
    # Create attack
    attack = None
    if args.attack == 'label_flip':
        attack = LabelFlipAttack(intensity=args.attack_intensity)
    elif args.attack == 'gradient_noise':
        attack = GradientNoiseAttack(intensity=args.attack_intensity)
    
    # Client config
    client_config = ClientConfig(
        batch_size=32,
        epochs=2,
        learning_rate=0.001,
        optimizer='adam'
    )
    
    # Create client
    fl_client = EnhancedFLClient(
        client_id=args.client_id,
        model=model,
        trainloader=train_loader,
        testloader=test_loader,
        config=client_config,
        attack=attack,
        device=device,
        logger=logger
    )
    
    logger.logger.info(f"Client {args.client_id} ready. Connecting to server...")
    
    # Connect to server
    try:
        fl.client.start_client(
            server_address=args.server,
            client=fl_client.to_client()
        )
        logger.logger.info(f"Client {args.client_id} completed successfully")
    except Exception as e:
        logger.logger.error(f"Client {args.client_id} failed: {e}")
        raise

if __name__ == "__main__":
    main()