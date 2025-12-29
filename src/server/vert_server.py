# src/server/vert_server.py
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from ..utils.logging_utils import ExperimentLogger, ExplainableDecision
from ..utils.config import ExperimentConfig
from ..defences.vert_defence import VERTDefenceStrategy


class VERTAggregationStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy with VERT defence mechanism.
    Integrates VERT vertical defence aggregation into Flower framework.
    
    VERT is a vertical defence that uses historical gradient information to predict
    client behavior and selects the top-κ clients with highest similarity between
    predicted and actual gradients.
    """
    
    def __init__(self, 
                 config: ExperimentConfig,
                 kappa: int = 5,
                 history_size: int = 10,
                 projection_dim: int = 100,
                 learning_rate: float = 0.01,
                 min_history_rounds: int = 3,
                 logger: Optional[ExperimentLogger] = None,
                 evaluate_fn: Optional[Any] = None,
                 **kwargs):
        """
        Initialize VERT aggregation strategy.
        
        Args:
            config: Experiment configuration
            kappa: Number of top clients to select
            history_size: Number of historical rounds to maintain
            projection_dim: Dimension of projected feature space
            learning_rate: Learning rate for predictor training
            min_history_rounds: Minimum rounds of history required before VERT activates
            logger: Experiment logger
            evaluate_fn: Evaluation function for centralized evaluation
        """
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.config = config
        self.logger = logger
        self.round_logs = []
        self._current_parameters = None
        self._evaluate_fn = evaluate_fn
        
        # Initialize VERT defence
        self.vert_defence = VERTDefenceStrategy(
            kappa=kappa,
            history_size=history_size,
            projection_dim=projection_dim,
            learning_rate=learning_rate,
            min_history_rounds=min_history_rounds
        )
        
        if self.logger:
            self.logger.logger.info(
                f"Initialized server with {self.vert_defence.get_defence_description()}"
            )
            if evaluate_fn:
                self.logger.logger.info("Centralized evaluation enabled on server")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using VERT defence."""
        
        if not results:
            return None, {}
        
        if self.logger:
            self.logger.logger.info(
                f"🔄 Round {server_round}: Aggregating {len(results)} client updates with VERT"
            )
        
        # Convert Flower results to defence format
        client_updates = {}
        for i, (client, fit_res) in enumerate(results):
            client_id = f"client_{i}"
            parameters = [np.array(param) for param in fl.common.parameters_to_ndarrays(fit_res.parameters)]
            num_samples = fit_res.num_examples
            metrics = fit_res.metrics if fit_res.metrics else {}
            
            client_updates[client_id] = (parameters, num_samples, metrics)
        
        # Apply VERT defence
        aggregated_params, decisions = self.vert_defence.aggregate_updates(client_updates)
        
        if aggregated_params is None:
            if self.logger:
                self.logger.logger.warning("⚠️  VERT aggregation returned None")
            return None, {}
        
        # Log decisions
        if self.logger:
            for i, decision in enumerate(decisions):
                client_id = f"client_{i}"
                status = "✅ ACCEPT" if decision.decision == "accept" else "❌ REJECT"
                self.logger.logger.info(
                    f"  {status} {client_id}: {decision.reasoning}"
                )
        
        # Store round log
        round_log = {
            'round': server_round,
            'defence_strategy': self.vert_defence.get_defence_description(),
            'n_clients': len(results),
            'decisions': [
                {
                    'client_id': f"client_{i}",
                    'decision': d.decision,
                    'confidence': d.confidence,
                    'reasoning': d.reasoning,
                    'evidence': d.evidence
                }
                for i, d in enumerate(decisions)
            ]
        }
        self.round_logs.append(round_log)
        
        # Convert back to Flower Parameters
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_params)
        self._current_parameters = aggregated_parameters
        
        # Prepare metrics
        metrics = {
            'round': server_round,
            'n_clients': len(results),
            'defence_strategy': self.vert_defence.get_defence_description()
        }
        
        return aggregated_parameters, metrics
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model using centralized test set if available."""
        if self._evaluate_fn is None:
            return None
        
        try:
            loss, metrics = self._evaluate_fn(server_round, 
                                             fl.common.parameters_to_ndarrays(parameters), 
                                             {})
            return loss, metrics
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Evaluation failed: {e}")
            return None
