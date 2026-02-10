# src/orchestration/simulation_runner.py
"""Flower simulation runner using Ray for scalable client execution"""
import argparse
import yaml
from typing import Dict, Any, Optional

import flwr as fl

from ..clients.enhanced_client import EnhancedFLClient
from ..models.cnn_mnist import MNISTNet
from ..datasets.mnist_handler import MNISTDataHandler
from ..attacks.label_flip import LabelFlipAttack
from ..attacks.gradient_noise import GradientNoiseAttack
from ..attacks.stat_opt_attack import StatOptAttack
from ..attacks.dny_opt_attack import DnyOptAttack
from ..attacks.min_max_attack import MinMaxAttack
from ..attacks.min_sum_attack import MinSumAttack
from ..server.cognitive_server import CognitiveAggregationStrategy
from ..server.no_defence_server import NoDefenceAggregationStrategy
from ..server.krum_server import KrumAggregationStrategy
from ..server.trimmed_mean_server import TrimmedMeanAggregationStrategy
from ..server.vert_server import VERTAggregationStrategy
from ..utils.config import ExperimentConfig, AttackConfig, ClientConfig, defenceConfig, DeterministicEnvironment
from ..utils.logging_utils import ExperimentLogger


def create_attack(attack_config: dict):
    """Create attack instance based on configuration"""
    if not attack_config.get('attack_enabled', False):
        return None

    attack_type = attack_config.get('attack_type', 'label_flip')
    intensity = attack_config.get('attack_intensity', 0.1)

    if attack_type == 'label_flip':
        return LabelFlipAttack(intensity=intensity)
    if attack_type == 'gradient_noise':
        return GradientNoiseAttack(intensity=intensity)
    if attack_type == 'stat_opt':
        constraint_factor = attack_config.get('constraint_factor', 1.5)
        adaptive_learning_rate = attack_config.get('adaptive_learning_rate', 0.1)
        return StatOptAttack(
            intensity=intensity,
            constraint_factor=constraint_factor,
            adaptive_learning_rate=adaptive_learning_rate,
        )
    if attack_type == 'dny_opt':
        learning_rate = attack_config.get('learning_rate', 0.1)
        exploration_rate = attack_config.get('exploration_rate', 0.1)
        discount_factor = attack_config.get('discount_factor', 0.95)
        detection_threshold = attack_config.get('detection_threshold', 0.7)
        return DnyOptAttack(
            intensity=intensity,
            learning_rate=learning_rate,
            exploration_rate=exploration_rate,
            discount_factor=discount_factor,
            detection_threshold=detection_threshold,
        )
    if attack_type == 'min_max':
        defense_models = attack_config.get('defense_models', ['krum', 'trimmed_mean'])
        optimization_steps = attack_config.get('optimization_steps', 10)
        return MinMaxAttack(
            intensity=intensity,
            defense_models=defense_models,
            optimization_steps=optimization_steps,
        )
    if attack_type == 'min_sum':
        distance_weight = attack_config.get('distance_weight', 0.7)
        optimization_lr = attack_config.get('optimization_lr', 0.01)
        max_iterations = attack_config.get('max_iterations', 100)
        convergence_threshold = attack_config.get('convergence_threshold', 1e-5)
        return MinSumAttack(
            intensity=intensity,
            distance_weight=distance_weight,
            optimization_lr=optimization_lr,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
    return None


class SimulationRunner:
    """Run Flower simulation with Ray-backed client scheduling"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.experiment_config = ExperimentConfig(**self.config.get('experiment', {}))
        self.logger = ExperimentLogger(self.experiment_config.experiment_name)

        DeterministicEnvironment.setup_seeds(self.experiment_config.seed)

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_attack_configs(self) -> Dict[int, AttackConfig]:
        attack_configs: Dict[int, AttackConfig] = {}
        for attack_spec in self.config.get('attacks', []) or []:
            attack_config = AttackConfig(**attack_spec)
            target_clients = attack_config.target_clients or []
            for client_id in target_clients:
                attack_configs[client_id] = attack_config
        return attack_configs

    def _get_client_hparams(self) -> Dict[str, Any]:
        orchestration = self.config.get('orchestration', {})
        client_config = self.config.get('client_config', {})
        return {
            "batch_size": client_config.get("batch_size", orchestration.get("batch_size", 32)),
            "epochs": client_config.get("local_epochs", 2),
            "learning_rate": client_config.get("learning_rate", 0.001),
            "optimizer": client_config.get("optimizer", "adam"),
        }

    def _create_centralized_eval_fn(self):
        from ..datasets.mnist_handler import MNISTDataHandler
        import torch
        import torch.nn as nn

        DeterministicEnvironment.setup_seeds(self.experiment_config.seed)
        self.logger.logger.info("Loading centralized test dataset for server evaluation...")
        data_handler = MNISTDataHandler(batch_size=64)
        _, test_loader = data_handler.create_client_dataloaders(num_clients=2, alpha=0.5)

        device = DeterministicEnvironment.get_device()
        self.logger.logger.info(f"Using device for server evaluation: {device}")

        def evaluate(server_round: int, parameters, config):
            try:
                model = MNISTNet().to(device)
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

    def _create_strategy(self):
        evaluate_fn = self._create_centralized_eval_fn()
        defence_config = defenceConfig(**self.config.get('defence', {}))

        if defence_config.strategy == 'cognitive_defence':
            return CognitiveAggregationStrategy(
                config=self.experiment_config,
                anomaly_threshold=defence_config.anomaly_threshold,
                reputation_decay=defence_config.reputation_decay,
                history_size=defence_config.history_size,
                logger=self.logger,
                evaluate_fn=evaluate_fn,
                min_fit_clients=self.experiment_config.min_clients,
                min_evaluate_clients=self.experiment_config.min_clients,
                min_available_clients=self.experiment_config.min_available_clients,
                fraction_evaluate=1.0,
            )
        if defence_config.strategy == 'krum':
            num_byzantine = self.config.get('defence', {}).get('num_byzantine', 2)
            multi_krum = self.config.get('defence', {}).get('multi_krum', False)
            return KrumAggregationStrategy(
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
        if defence_config.strategy == 'trimmed_mean':
            beta = self.config.get('defence', {}).get('beta', 0.2)
            return TrimmedMeanAggregationStrategy(
                config=self.experiment_config,
                beta=beta,
                logger=self.logger,
                evaluate_fn=evaluate_fn,
                min_fit_clients=self.experiment_config.min_clients,
                min_evaluate_clients=self.experiment_config.min_clients,
                min_available_clients=self.experiment_config.min_available_clients,
                fraction_evaluate=1.0,
            )
        if defence_config.strategy == 'vert':
            kappa = self.config.get('defence', {}).get('kappa', 5)
            history_size = self.config.get('defence', {}).get('history_size', 10)
            projection_dim = self.config.get('defence', {}).get('projection_dim', 100)
            learning_rate = self.config.get('defence', {}).get('learning_rate', 0.01)
            min_history_rounds = self.config.get('defence', {}).get('min_history_rounds', 3)
            return VERTAggregationStrategy(
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

        return NoDefenceAggregationStrategy(
            config=self.experiment_config,
            logger=self.logger,
            evaluate_fn=evaluate_fn,
            min_fit_clients=self.experiment_config.min_clients,
            min_evaluate_clients=self.experiment_config.min_clients,
            min_available_clients=self.experiment_config.min_available_clients,
            fraction_evaluate=1.0,
        )

    def run(self):
        orchestration = self.config.get('orchestration', {})
        simulation = self.config.get('simulation', {})
        num_clients = orchestration.get('num_clients', 10)
        hparams = self._get_client_hparams()
        attack_configs = self._create_attack_configs()

        strategy = self._create_strategy()

        def client_fn(cid: str) -> fl.client.Client:
            client_id = int(cid)
            DeterministicEnvironment.setup_seeds(self.experiment_config.seed + client_id)
            device = DeterministicEnvironment.get_device()

            logger = ExperimentLogger(f"{self.experiment_config.experiment_name}_client_{client_id}")
            logger.logger.info(
                f"Starting client {client_id} (simulation) with hparams: {hparams}"
            )

            model = MNISTNet()

            data_handler = MNISTDataHandler(batch_size=hparams["batch_size"])
            client_loaders, test_loader = data_handler.create_client_dataloaders(
                num_clients=num_clients,
                alpha=0.5,
            )
            train_loader = client_loaders[client_id]

            attack = None
            attack_config = attack_configs.get(client_id)
            if attack_config and attack_config.enabled:
                attack = create_attack({
                    "attack_enabled": True,
                    "attack_type": attack_config.attack_type,
                    "attack_intensity": attack_config.intensity,
                })

            client_config = ClientConfig(
                batch_size=hparams["batch_size"],
                epochs=hparams["epochs"],
                learning_rate=hparams["learning_rate"],
                optimizer=hparams["optimizer"],
            )

            fl_client = EnhancedFLClient(
                client_id=client_id,
                model=model,
                trainloader=train_loader,
                testloader=test_loader,
                config=client_config,
                attack=attack,
                device=device,
                logger=logger,
            )

            return fl_client.to_client()

        client_resources = simulation.get("client_resources", {"num_cpus": 1})
        ray_init_args = simulation.get("ray_init_args", {})

        self.logger.logger.info(
            f"Starting Flower simulation with {num_clients} clients, "
            f"client_resources={client_resources}"
        )

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=self.experiment_config.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
            ray_init_args=ray_init_args,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML',
    )
    args = parser.parse_args()

    SimulationRunner(args.config).run()


if __name__ == "__main__":
    main()
