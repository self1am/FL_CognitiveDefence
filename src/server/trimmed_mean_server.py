# src/server/trimmed_mean_server.py
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from ..utils.logging_utils import ExperimentLogger, ExplainableDecision
from ..utils.config import ExperimentConfig
from ..defences.trimmed_mean_defence import TrimmedMeanDefenceStrategy

class TrimmedMeanAggregationStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy with Trimmed Mean defence mechanism.
    Integrates Trimmed Mean Byzantine-robust aggregation into Flower framework.
    """
    
    def __init__(self, 
                 config: ExperimentConfig,
                 beta: float = 0.2,
                 logger: Optional[ExperimentLogger] = None,
                 evaluate_fn: Optional[Any] = None,
                 **kwargs):
        
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.config = config
        self.logger = logger
        self.round_logs = []
        self._current_parameters = None
        self._evaluate_fn = evaluate_fn
        
        # Initialize Trimmed Mean defence
        self.trimmed_mean_defence = TrimmedMeanDefenceStrategy(beta=beta)
        
        if self.logger:
            self.logger.logger.info(
                f"Initialized server with {self.trimmed_mean_defence.get_defence_description()}"
            )
            if evaluate_fn:
                self.logger.logger.info("Centralized evaluation enabled on server")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using Trimmed Mean defence."""
        
        if not results:
            return None, {}
        
        if self.logger:
            self.logger.logger.info(
                f"🔄 Round {server_round}: Aggregating {len(results)} client updates with Trimmed Mean"
            )
        
        # Convert Flower results to defence format
        client_updates = {}
        for i, (client, fit_res) in enumerate(results):
            client_id = f"client_{i}"
            parameters = [np.array(param) for param in fl.common.parameters_to_ndarrays(fit_res.parameters)]
            num_samples = fit_res.num_examples
            metrics = fit_res.metrics if fit_res.metrics else {}
            
            client_updates[client_id] = (parameters, num_samples, metrics)
        
        # Apply Trimmed Mean defence
        aggregated_params, decisions = self.trimmed_mean_defence.aggregate_updates(client_updates)
        
        if aggregated_params is None:
            if self.logger:
                self.logger.logger.warning("⚠️  Trimmed Mean aggregation returned None")
            return None, {}
        
        # Log decisions
        if self.logger:
            for i, decision in enumerate(decisions):
                client_id = f"client_{i}"
                if decision.decision == "accept":
                    status = "✅ ACCEPT"
                elif decision.decision == "partial_reject":
                    status = "⚠️  TRIMMED"
                else:
                    status = "❌ REJECT"
                self.logger.logger.info(
                    f"  {status} {client_id}: {decision.reasoning}"
                )
        
        # Store round log
        round_log = {
            'round': server_round,
            'defence_strategy': self.trimmed_mean_defence.get_defence_description(),
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
            'defence_strategy': self.trimmed_mean_defence.get_defence_description()
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
