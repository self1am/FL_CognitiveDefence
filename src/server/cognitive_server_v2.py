# src/server/cognitive_server_v2.py
"""
CogDef v2 Server Integration — Flower FedAvg strategy wrapper.

Wires the CognitiveDefenceV2 OODA+MAPE-K loop into Flower's
aggregate_fit() pipeline, identical to how cognitive_server.py
wraps the v1 defence.
"""
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from ..defences.cognitive_defence_v2 import CognitiveDefenceV2, ThreatLevel
from ..utils.logging_utils import ExperimentLogger, ExplainableDecision
from ..utils.config import ExperimentConfig


class CognitiveAggregationStrategyV2(fl.server.strategy.FedAvg):
    """
    FedAvg strategy enhanced with CogDef v2 multi-signal cognitive defence.
    Drop-in replacement for CognitiveAggregationStrategy.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        # CogDef v2 parameters
        anomaly_threshold: float = 0.5,
        direction_weight: float = 0.40,
        norm_weight: float = 0.15,
        cluster_weight: float = 0.25,
        temporal_weight: float = 0.20,
        initial_reputation: float = 0.5,
        recovery_rate: float = 0.03,
        penalty_severity: float = 0.8,
        yellow_threshold: float = 0.3,
        orange_threshold: float = 0.6,
        red_threshold: float = 0.8,
        clip_multiplier: float = 2.0,
        trim_beta: float = 0.2,
        enable_mape_k: bool = True,
        history_size: int = 100,
        # Flower + logging
        logger: Optional[ExperimentLogger] = None,
        evaluate_fn: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.config = config
        self.logger = logger
        self.round_logs = []
        self._current_parameters = None
        self._last_accuracy = None

        # Instantiate the v2 defence
        self.defence = CognitiveDefenceV2(
            anomaly_threshold=anomaly_threshold,
            direction_weight=direction_weight,
            norm_weight=norm_weight,
            cluster_weight=cluster_weight,
            temporal_weight=temporal_weight,
            initial_reputation=initial_reputation,
            recovery_rate=recovery_rate,
            penalty_severity=penalty_severity,
            yellow_threshold=yellow_threshold,
            orange_threshold=orange_threshold,
            red_threshold=red_threshold,
            clip_multiplier=clip_multiplier,
            trim_beta=trim_beta,
            enable_mape_k=enable_mape_k,
            history_size=history_size,
        )

        if self.logger:
            self.logger.logger.info(
                f"Initialized CogDef v2 server — {self.defence.get_defence_description()}"
            )

    def aggregate_fit(self, server_round: int, results, failures):
        """Override aggregation with CogDef v2 cognitive defence."""
        if self.logger:
            self.logger.logger.info(
                f"[CogDef v2] Round {server_round}: "
                f"{len(results)} results, {len(failures)} failures, "
                f"posture={self.defence.threat_level.value}"
            )

        if not results:
            if self.logger:
                self.logger.logger.warning("No results received")
            return None, {}

        # Convert Flower results → internal format
        try:
            client_updates = {}
            for i, (client, fit_res) in enumerate(results):
                client_id = f"client_{i}"
                client_updates[client_id] = (
                    fl.common.parameters_to_ndarrays(fit_res.parameters),
                    fit_res.num_examples,
                    fit_res.metrics,
                )
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Error converting results: {e}")
            return super().aggregate_fit(server_round, results, failures)

        # Run CogDef v2 OODA + MAPE-K loop
        try:
            aggregated_params, explainable_decisions = self.defence.aggregate_updates(
                client_updates, accuracy=self._last_accuracy
            )

            # Log round summary
            round_metrics = {
                'round': server_round,
                'num_clients': len(client_updates),
                'threat_level': self.defence.threat_level.value,
                'num_flagged': sum(
                    1 for d in explainable_decisions
                    if d.decision in ('reduce_weight', 'reject')
                ),
                'detector_weights': dict(self.defence.detector_weights),
                'defence_strategy': self.defence.get_defence_description(),
            }

            if self.logger:
                self.logger.log_round_summary(
                    server_round, round_metrics, explainable_decisions
                )

            # Store log
            self.round_logs.append({
                'round': server_round,
                'timestamp': datetime.now().isoformat(),
                'metrics': round_metrics,
                'decisions': [
                    {
                        'decision': d.decision,
                        'confidence': d.confidence,
                        'reasoning': d.reasoning,
                        'evidence': d.evidence,
                    }
                    for d in explainable_decisions
                ],
            })

        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"CogDef v2 error: {e}, falling back to FedAvg")
            return super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            self._current_parameters = fl.common.ndarrays_to_parameters(aggregated_params)
            return self._current_parameters, {}
        else:
            if self.logger:
                self.logger.logger.warning("CogDef v2 returned None, falling back to FedAvg")
            return super().aggregate_fit(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        """Intercept evaluation to feed accuracy back to MAPE-K."""
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            if 'accuracy' in metrics:
                self._last_accuracy = metrics['accuracy']
        return result

    def get_round_logs(self) -> List[Dict[str, Any]]:
        return self.round_logs.copy()
