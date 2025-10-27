# src/server/no_defence_server.py
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Any, Dict, Callable
from datetime import datetime

from ..utils.logging_utils import ExperimentLogger, ExplainableDecision
from ..utils.config import ExperimentConfig


class NoDefenceAggregationStrategy(fl.server.strategy.FedAvg):
    """Simple FedAvg aggregation with no defense mechanisms"""
    
    def __init__(self, 
                 config: ExperimentConfig,
                 logger: Optional[ExperimentLogger] = None,
                 evaluate_fn: Optional[Callable] = None,
                 **kwargs):
        
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.config = config
        self.logger = logger
        self.round_logs = []
        self._current_parameters = None
        self._evaluate_fn = evaluate_fn
        
        if self.logger:
            self.logger.logger.info("Initialized server with No Defense (Simple FedAvg)")
            if evaluate_fn:
                self.logger.logger.info("Centralized evaluation enabled on server")
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Simple FedAvg aggregation without defense mechanisms"""
        if self.logger:
            self.logger.logger.info(
                f"Starting FedAvg aggregation for round {server_round} - "
                f"{len(results)} results, {len(failures)} failures"
            )
        
        if not results:
            if self.logger:
                self.logger.logger.warning("No results received for aggregation")
            return None, {}
        
        try:
            # Use standard FedAvg from parent class
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            
            # Create simple decision logs for consistency
            decisions = []
            for i, (client, fit_res) in enumerate(results):
                decision = ExplainableDecision(
                    decision="accept",
                    confidence=1.0,
                    reasoning="No defense applied - simple FedAvg aggregation",
                    evidence={
                        'weight': fit_res.num_examples,
                        'aggregation_method': 'fedavg'
                    }
                )
                decisions.append(decision)
            
            # Log round information
            round_metrics = {
                'round': server_round,
                'num_clients': len(results),
                'num_decisions': len(decisions),
                'avg_decision_confidence': 1.0,
                'defence_strategy': 'No Defense (Simple FedAvg)'
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
            
            return aggregated_parameters, aggregated_metrics
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Error in FedAvg aggregation: {e}")
            return None, {}
    
    def get_round_logs(self) -> List[Dict[str, Any]]:
        """Get all round logs"""
        return self.round_logs.copy()
