# src/server/cognitive_defence_posg_server.py
"""
POSG Aggregation Strategy — Wraps the SAC/GRU defended aggregation.
"""
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from ..defences.cognitive_defence_posg import CognitiveDefencePOSG
from ..utils.logging_utils import ExperimentLogger
from ..utils.config import ExperimentConfig


class POSGAggregationStrategy(fl.server.strategy.FedAvg):
    """
    Federated learning strategy using SAC + GRU belief tracking for defence.
    
    Wraps the CognitiveDefencePOSG to integrate with the Flower framework.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        max_clients: int = 100,
        obs_dim: int = 6,
        belief_hidden_dim: int = 64,
        sac_hidden_dims: list = None,
        lr: float = 0.0003,
        gamma: float = 0.99,
        reward_alpha: float = 1.0,
        reward_beta: float = 0.3,
        reward_gamma: float = 0.2,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        device: str = "cpu",
        warmup_rounds: int = 5,
        logger: Optional[ExperimentLogger] = None,
        evaluate_fn: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.config = config
        self.logger = logger
        self._evaluate_fn = evaluate_fn
        
        # Initialize POSG defence
        sac_hidden_dims = sac_hidden_dims or [256, 256]
        self.defence = CognitiveDefencePOSG(
            max_clients=max_clients,
            obs_dim=obs_dim,
            belief_hidden_dim=belief_hidden_dim,
            sac_hidden_dims=sac_hidden_dims,
            lr=lr,
            gamma=gamma,
            reward_alpha=reward_alpha,
            reward_beta=reward_beta,
            reward_gamma=reward_gamma,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            device=device,
            history_size=200,
            warmup_rounds=warmup_rounds,
        )
        
        self._current_parameters = None
        self._last_val_acc = None

        if self.logger:
            self.logger.logger.info(
                f"Initialized server with POSG Defence Strategy (SAC + GRU)"
            )
            self.logger.logger.info(f"  {self.defence.get_defence_description()}")
            if evaluate_fn:
                self.logger.logger.info("Centralized evaluation enabled on server")

    def aggregate_fit(
        self, server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, float]]:
        """Aggregate fitted model parameters using POSG defence."""
        
        if not results:
            return None, {}

        # Extract parameters for aggregation
        client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]] = {}
        
        for client, fit_res in results:
            client_id = client.cid
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_samples = fit_res.num_examples
            metrics = fit_res.metrics or {}
            
            client_updates[client_id] = (parameters, num_samples, metrics)
        
        # Set global model for cosine-similarity feature
        if self._current_parameters is not None:
            current_params = fl.common.parameters_to_ndarrays(self._current_parameters)
            self.defence.set_global_model(current_params)
        
        # Run POSG aggregation with validation accuracy if available
        aggregated_params, decisions = self.defence.aggregate_updates(
            client_updates, 
            val_acc=self._last_val_acc,
            deterministic=False,
        )
        
        if aggregated_params is None:
            # Fallback to simple FedAvg if defence fails
            return super().aggregate_fit(server_round, results, failures)
        
        # Log decisions
        if self.logger:
            for decision in decisions:
                self.logger.log_decision(decision)
        
        # Convert back to Flower parameters
        result_params = fl.common.ndarrays_to_parameters(aggregated_params)
        self._current_parameters = result_params
        
        # Return aggregated parameters and metrics
        metrics_aggregated = {
            "defence_round": server_round,
            "active_clients": len(client_updates),
        }
        
        return result_params, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: fl.common.Parameters
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Evaluate model using centralized test set and store accuracy for reward."""
        
        if self.evaluate_fn is None:
            return None
        
        # Call evaluate_fn with NDArrays exactly as FedAvg parent does
        parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        result = self.evaluate_fn(server_round, parameters_ndarrays, {})
        
        if result is not None:
            loss, metrics = result
            # Store validation accuracy for next round's SAC reward
            if "centralized_accuracy" in metrics:
                self._last_val_acc = float(metrics["centralized_accuracy"])
            
            return loss, metrics
        
        return None
