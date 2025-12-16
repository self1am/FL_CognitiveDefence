# src/defences/krum_defence.py
import numpy as np
from typing import Dict, List, Tuple, Any
from .base_defence import Basedefence
from ..utils.logging_utils import ExplainableDecision

class KrumDefenceStrategy(Basedefence):
    """
    Krum defence strategy for Byzantine-robust aggregation.
    
    Krum selects a single client update (or average of top-k updates) that has
    the smallest sum of squared distances to its nearest neighbors.
    This makes it robust against Byzantine attacks.
    
    Reference: Blanchard et al., "Machine Learning with Adversaries: Byzantine 
    Tolerant Gradient Descent", NeurIPS 2017
    """
    
    def __init__(self, num_byzantine: int = 2, multi_krum: bool = False, **kwargs):
        """
        Initialize Krum defence strategy.
        
        Args:
            num_byzantine: Expected number of Byzantine (malicious) clients
            multi_krum: If True, average top (n-f-2) clients. If False, use single best.
        """
        super().__init__(**kwargs)
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
    
    def _compute_pairwise_distances(self, updates: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between all client updates.
        
        Args:
            updates: List of flattened parameter updates
            
        Returns:
            Distance matrix of shape (n_clients, n_clients)
        """
        n = len(updates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Compute Euclidean distance
                dist = np.linalg.norm(updates[i] - updates[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _flatten_parameters(self, parameters: List[np.ndarray]) -> np.ndarray:
        """Flatten list of parameter arrays into single vector."""
        return np.concatenate([param.flatten() for param in parameters])
    
    def _unflatten_parameters(self, flat_params: np.ndarray, 
                            shapes: List[Tuple]) -> List[np.ndarray]:
        """Unflatten single vector back into list of parameter arrays."""
        params = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            params.append(flat_params[idx:idx + size].reshape(shape))
            idx += size
        return params
    
    def aggregate_updates(self, 
                         client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
                        ) -> Tuple[List[np.ndarray], List[ExplainableDecision]]:
        """
        Aggregate client updates using Krum algorithm.
        
        The algorithm:
        1. Flatten all client updates into vectors
        2. Compute pairwise distances between all updates
        3. For each update, compute score as sum of distances to closest n-f-2 neighbors
        4. Select update(s) with smallest score
        """
        if not client_updates:
            return None, []
        
        # Extract client data
        client_ids = list(client_updates.keys())
        n_clients = len(client_ids)
        
        # Need at least 2f+3 clients for Krum to work
        min_clients = 2 * self.num_byzantine + 3
        if n_clients < min_clients:
            # Fall back to simple averaging if not enough clients
            return self._fallback_aggregation(client_updates)
        
        # Flatten all parameter updates
        flattened_updates = []
        param_shapes = None
        
        for client_id in client_ids:
            parameters, _, _ = client_updates[client_id]
            if param_shapes is None:
                param_shapes = [param.shape for param in parameters]
            flattened_updates.append(self._flatten_parameters(parameters))
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(flattened_updates)
        
        # Compute Krum scores
        # For each client, sum distances to n-f-2 closest neighbors
        n_closest = n_clients - self.num_byzantine - 2
        scores = []
        
        for i in range(n_clients):
            # Get distances from client i to all others
            dists_i = distances[i].copy()
            dists_i[i] = np.inf  # Exclude self
            
            # Sum of distances to n_closest nearest neighbors
            closest_distances = np.sort(dists_i)[:n_closest]
            score = np.sum(closest_distances ** 2)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Select best client(s)
        decisions = []
        
        if self.multi_krum:
            # Multi-Krum: average top n-f-2 updates
            n_select = n_closest
            selected_indices = np.argsort(scores)[:n_select]
            
            # Average selected updates
            selected_updates = [flattened_updates[i] for i in selected_indices]
            aggregated_flat = np.mean(selected_updates, axis=0)
            
            # Log decisions
            for i, client_id in enumerate(client_ids):
                if i in selected_indices:
                    decision = ExplainableDecision(
                        decision="accept",
                        confidence=1.0,
                        reasoning=f"Selected by Multi-Krum (score: {scores[i]:.2f}, rank: {np.where(selected_indices == i)[0][0] + 1}/{n_select})",
                        evidence={
                            'krum_score': float(scores[i]),
                            'selected': True,
                            'n_selected': n_select,
                            'method': 'multi_krum'
                        }
                    )
                else:
                    decision = ExplainableDecision(
                        decision="reject",
                        confidence=0.8,
                        reasoning=f"Rejected by Multi-Krum (score: {scores[i]:.2f}, too high)",
                        evidence={
                            'krum_score': float(scores[i]),
                            'selected': False,
                            'method': 'multi_krum'
                        }
                    )
                decisions.append(decision)
        else:
            # Standard Krum: select single best update
            best_idx = np.argmin(scores)
            aggregated_flat = flattened_updates[best_idx]
            
            # Log decisions
            for i, client_id in enumerate(client_ids):
                if i == best_idx:
                    decision = ExplainableDecision(
                        decision="accept",
                        confidence=1.0,
                        reasoning=f"Selected by Krum (best score: {scores[i]:.2f})",
                        evidence={
                            'krum_score': float(scores[i]),
                            'selected': True,
                            'best': True,
                            'method': 'krum'
                        }
                    )
                else:
                    decision = ExplainableDecision(
                        decision="reject",
                        confidence=0.8,
                        reasoning=f"Not selected by Krum (score: {scores[i]:.2f})",
                        evidence={
                            'krum_score': float(scores[i]),
                            'selected': False,
                            'method': 'krum'
                        }
                    )
                decisions.append(decision)
        
        # Unflatten aggregated parameters
        aggregated_params = self._unflatten_parameters(aggregated_flat, param_shapes)
        
        self.increment_round()
        
        return aggregated_params, decisions
    
    def _fallback_aggregation(self, 
                             client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
                            ) -> Tuple[List[np.ndarray], List[ExplainableDecision]]:
        """Fallback to simple FedAvg when not enough clients for Krum."""
        weighted_updates = []
        total_samples = 0
        
        for client_id, (parameters, num_samples, metrics) in client_updates.items():
            weighted_updates.append((parameters, num_samples))
            total_samples += num_samples
        
        if weighted_updates and total_samples > 0:
            aggregated_params = []
            num_params = len(weighted_updates[0][0])
            
            for param_idx in range(num_params):
                weighted_sum = sum(params[param_idx] * weight 
                                 for params, weight in weighted_updates)
                aggregated_params.append(weighted_sum / total_samples)
        else:
            aggregated_params = None
        
        # Create decisions
        decisions = []
        for client_id in client_updates.keys():
            decision = ExplainableDecision(
                decision="accept",
                confidence=0.5,
                reasoning=f"Fallback to FedAvg (insufficient clients for Krum: need {2 * self.num_byzantine + 3}, have {len(client_updates)})",
                evidence={'fallback': True, 'method': 'fedavg'}
            )
            decisions.append(decision)
        
        self.increment_round()
        return aggregated_params, decisions
    
    def get_defence_description(self) -> str:
        mode = "Multi-Krum" if self.multi_krum else "Krum"
        return f"{mode} Defence Strategy (f={self.num_byzantine} Byzantine clients)"
