# src/defences/cognitive_defense.py
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
from datetime import datetime
from .base_defence import BaseDefense
from ..utils.logging_utils import ExplainableDecision

class Cognitivedefencestrategy(BaseDefense):
    """
    Enhanced cognitive defense implementing OODA loop and MAPE-K framework
    """
    
    def __init__(self, anomaly_threshold: float = 0.7, reputation_decay: float = 0.8, 
                 history_size: int = 100):
        super().__init__(history_size=history_size)
        self.anomaly_threshold = anomaly_threshold
        self.reputation_decay = reputation_decay
        self.historical_updates = deque(maxlen=history_size)
    
    def observe(self, client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """OODA: Observe - Collect information about client updates"""
        observations = {}
        for client_id, (parameters, num_samples, metrics) in client_updates.items():
            param_norms = [float(np.linalg.norm(param)) for param in parameters]
            observations[client_id] = {
                'param_norms': param_norms,
                'total_norm': sum(param_norms),
                'num_samples': num_samples,
                'avg_norm': sum(param_norms) / len(param_norms) if param_norms else 0,
                'update_time': datetime.now().isoformat()
            }
        return observations
    
    def orient(self, observations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """OODA: Orient - Analyze observations in context"""
        analysis = {}
        
        if len(self.historical_updates) > 2:  # Need some history for comparison
            historical_norms = [update['total_norm'] for update in self.historical_updates]
            mean_norm = np.mean(historical_norms)
            std_norm = np.std(historical_norms)
            
            for client_id, obs in observations.items():
                z_score = abs(obs['total_norm'] - mean_norm) / (std_norm + 1e-8)
                is_anomalous = z_score > 2.0
                confidence = min(z_score / 3.0, 1.0)
                
                analysis[client_id] = {
                    'z_score': float(z_score),
                    'is_anomalous': is_anomalous,
                    'confidence': float(confidence),
                    'deviation_from_mean': float(obs['total_norm'] - mean_norm),
                    'historical_context': {
                        'mean_norm': float(mean_norm),
                        'std_norm': float(std_norm),
                        'history_size': len(self.historical_updates)
                    }
                }
        else:
            # Insufficient history - minimal analysis
            for client_id in observations.keys():
                analysis[client_id] = {
                    'z_score': 0.0,
                    'is_anomalous': False,
                    'confidence': 0.0,
                    'deviation_from_mean': 0.0,
                    'historical_context': {'insufficient_history': True}
                }
        
        return analysis
    
    def decide(self, analysis: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[ExplainableDecision]]:
        """OODA: Decide - Determine actions based on analysis"""
        decisions = {}
        explainable_decisions = []
        
        for client_id, client_analysis in analysis.items():
            current_reputation = self.get_client_reputation(client_id)
            
            if client_analysis['is_anomalous']:
                # Reduce reputation and weight
                new_reputation = current_reputation * self.reputation_decay
                weight_multiplier = max(new_reputation, 0.1)
                
                decision = ExplainableDecision(
                    decision="reduce_weight",
                    confidence=client_analysis['confidence'],
                    reasoning=f"Anomalous update detected with z-score {client_analysis['z_score']:.2f}. "
                             f"Reducing client weight from {current_reputation:.2f} to {weight_multiplier:.2f}",
                    evidence={
                        'z_score': client_analysis['z_score'],
                        'previous_reputation': current_reputation,
                        'new_reputation': new_reputation,
                        'historical_context': client_analysis.get('historical_context', {})
                    }
                )
                
                decisions[client_id] = {
                    'action': 'reduce_weight',
                    'weight_multiplier': weight_multiplier,
                    'reason': decision.reasoning
                }
                
                self.update_client_reputation(client_id, new_reputation - current_reputation)
                
            else:
                # Reward good behavior
                reputation_bonus = 0.05
                new_reputation = min(current_reputation + reputation_bonus, 1.0)
                
                decision = ExplainableDecision(
                    decision="accept",
                    confidence=1.0 - client_analysis['confidence'],
                    reasoning=f"Normal update detected. Maintaining/improving reputation "
                             f"from {current_reputation:.2f} to {new_reputation:.2f}",
                    evidence={
                        'z_score': client_analysis['z_score'],
                        'previous_reputation': current_reputation,
                        'new_reputation': new_reputation
                    }
                )
                
                decisions[client_id] = {
                    'action': 'accept',
                    'weight_multiplier': 1.0,
                    'reason': decision.reasoning
                }
                
                self.update_client_reputation(client_id, reputation_bonus)
            
            explainable_decisions.append(decision)
        
        return decisions, explainable_decisions
    
    def act(self, client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]], 
            decisions: Dict[str, Dict[str, Any]]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """OODA: Act - Execute decisions and aggregate updates"""
        weighted_updates = []
        total_samples = 0
        aggregation_log = {}
        
        for client_id, (parameters, num_samples, metrics) in client_updates.items():
            if client_id in decisions:
                weight = decisions[client_id]['weight_multiplier'] * num_samples
                weighted_updates.append((parameters, weight))
                total_samples += weight
                
                aggregation_log[client_id] = {
                    'original_weight': num_samples,
                    'adjusted_weight': float(weight),
                    'action': decisions[client_id]['action'],
                    'reputation': self.get_client_reputation(client_id)
                }
        
        # Perform weighted aggregation
        if weighted_updates and total_samples > 0:
            aggregated_params = []
            num_params = len(weighted_updates[0][0])
            
            for param_idx in range(num_params):
                weighted_sum = sum(params[param_idx] * weight 
                                 for params, weight in weighted_updates)
                aggregated_params.append(weighted_sum / total_samples)
        else:
            aggregated_params = None
        
        return aggregated_params, aggregation_log
    
    def aggregate_updates(self, client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]) -> Tuple[List[np.ndarray], List[ExplainableDecision]]:
        """Main aggregation method implementing OODA loop"""
        # OODA Loop execution
        observations = self.observe(client_updates)
        analysis = self.orient(observations)
        decisions, explainable_decisions = self.decide(analysis)
        aggregated_params, aggregation_log = self.act(client_updates, decisions)
        
        # Store for historical context
        self.historical_updates.extend([obs for obs in observations.values()])
        self.increment_round()
        
        return aggregated_params, explainable_decisions
    
    def get_defense_description(self) -> str:
        return f"Cognitive Defense Strategy (OODA Loop + MAPE-K, threshold={self.anomaly_threshold})"