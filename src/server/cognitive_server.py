# src/server/cognitive_server.py
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from ..defences.base_defence import Basedefence
from ..utils.logging_utils import ExperimentLogger, ExplainableDecision
from ..utils.config import ExperimentConfig

class CognitiveAggregationStrategy(fl.server.strategy.FedAvg):
    """Enhanced FedAvg with cognitive defence mechanisms"""
    
    def __init__(self, 
                 defence: Basedefence,
                 config: ExperimentConfig,
                 logger: Optional[ExperimentLogger] = None,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.defence = defence
        self.config = config
        self.logger = logger
        self.round_logs = []
        self._current_parameters = None
        
        if self.logger:
            self.logger.logger.info(f"Initialized server with {self.defence.get_defence_description()}")
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Override aggregation with cognitive defences"""
        if self.logger:
            self.logger.logger.info(
                f"Starting cognitive aggregation for round {server_round} - "
                f"{len(results)} results, {len(failures)} failures"
            )
        
        if not results:
            if self.logger:
                self.logger.logger.warning("No results received for aggregation")
            return None, {}
        
        try:
            # Convert Flower results to our internal format
            client_updates = {}
            for i, (client, fit_res) in enumerate(results):
                client_id = f"client_{i}"
                client_updates[client_id] = (
                    fl.common.parameters_to_ndarrays(fit_res.parameters),
                    fit_res.num_examples,
                    fit_res.metrics
                )
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Error converting client results: {e}")
            return super().aggregate_fit(server_round, results, failures)
        
        try:
            # Apply cognitive defence
            aggregated_params, decisions = self.defence.aggregate_updates(client_updates)
            
            # Log round information
            round_metrics = {
                'round': server_round,
                'num_clients': len(client_updates),
                'num_decisions': len(decisions),
                'avg_decision_confidence': np.mean([d.confidence for d in decisions]) if decisions else 0,
                'defence_strategy': self.defence.get_defence_description()
            }
            
            if self.logger:
                self.logger.log_round_summary(server_round, round_metrics, decisions)
            
            # Store round log
            round_log = {
                'round': server_round,
                'timestamp': datetime.now().isoformat(),
                'metrics': round_metrics,
                'decisions': [
                    {
                        'decision': d.decision,
                        'confidence': d.confidence,
                        'reasoning': d.reasoning,
                        'evidence': d.evidence
                    } for d in decisions
                ]
            }
            self.round_logs.append(round_log)
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Error in cognitive aggregation: {e}")
            return super().aggregate_fit(server_round, results, failures)
        
        if aggregated_params is not None:
            self._current_parameters = fl.common.ndarrays_to_parameters(aggregated_params)
            return self._current_parameters, {}
        else:
            if self.logger:
                self.logger.logger.warning("Cognitive aggregation failed, falling back to standard FedAvg")
            return super().aggregate_fit(server_round, results, failures)
    
    def get_round_logs(self) -> List[Dict[str, Any]]:
        """Get all round logs"""
        return self.round_logs.copy()