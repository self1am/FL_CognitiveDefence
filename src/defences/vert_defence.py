# src/defences/vert_defence.py
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from .base_defence import Basedefence
from ..utils.logging_utils import ExplainableDecision


class VERTDefenceStrategy(Basedefence):
    """
    VERT (VERtical Trusted aggregation) defence strategy for Byzantine-robust aggregation.
    
    VERT is a vertical defence that uses historical gradient information to predict
    client behavior and selects the top-κ clients with highest similarity between
    predicted and actual gradients.
    
    The algorithm uses:
    - A predictor (f_pred) to predict what gradient a client should produce
    - A projector (f_proj) to project gradients to a lower-dimensional feature space
    - Coefficient matrices A and B to combine historical user and global gradients
    - Cosine similarity to compare predicted vs actual gradients
    
    Reference: VERT: Verified and Efficient Robust Aggregation for Vertical Federated Learning
    https://arxiv.org/pdf/2411.10673
    """
    
    # Class constants for numerical stability and initialization
    DEFAULT_RANDOM_SEED = 42
    DEFAULT_COEFF_CENTER = 0.5
    COEFF_NOISE_SCALE = 0.1
    PREDICTOR_INIT_SCALE = 0.01
    NORM_EPSILON = 1e-10
    
    def __init__(self, 
                 kappa: int = 5,
                 history_size: int = 10,
                 projection_dim: int = 100,
                 learning_rate: float = 0.01,
                 min_history_rounds: int = 3,
                 random_seed: int = 42,
                 **kwargs):
        """
        Initialize VERT defence strategy.
        
        Args:
            kappa: Number of top clients to select (κ in the algorithm)
            history_size: Number of historical rounds to maintain (m in the algorithm)
            projection_dim: Dimension of projected feature space
            learning_rate: Learning rate for predictor training
            min_history_rounds: Minimum rounds of history required before VERT activates
            random_seed: Random seed for reproducibility
        """
        super().__init__(**kwargs)
        self.kappa = kappa
        self.history_size = history_size
        self.projection_dim = projection_dim
        self.learning_rate = learning_rate
        self.min_history_rounds = min_history_rounds
        self.random_seed = random_seed
        
        # Historical gradients storage
        # client_history: {client_id: deque of (round, gradient)}
        self.client_history: Dict[str, deque] = {}
        # global_history: deque of (round, gradient)
        self.global_history: deque = deque(maxlen=history_size)
        
        # Coefficient matrices (initialized lazily based on gradient dimension)
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        
        # Predictor weights (simple linear predictor, initialized lazily)
        self.predictor_weights: Optional[np.ndarray] = None
        
        # Set of clients considered optimal in previous rounds
        self.optimal_clients: set = set()
        
    def _flatten_parameters(self, parameters: List[np.ndarray]) -> np.ndarray:
        """Flatten list of parameter arrays into single vector."""
        return np.concatenate([param.flatten() for param in parameters])
    
    def _unflatten_parameters(self, flat_params: np.ndarray, 
                              shapes: List[Tuple]) -> List[np.ndarray]:
        """Unflatten single vector back into list of parameter arrays."""
        params = []
        idx = 0
        for shape in shapes:
            size = int(np.prod(shape))
            params.append(flat_params[idx:idx + size].reshape(shape))
            idx += size
        return params
    
    def _init_coefficient_matrices(self, gradient_dim: int):
        """Initialize coefficient matrices A and B."""
        # A and B are element-wise coefficient matrices
        # Initialize with small random values centered around DEFAULT_COEFF_CENTER
        np.random.seed(self.random_seed)
        self.A = self.DEFAULT_COEFF_CENTER + self.COEFF_NOISE_SCALE * np.random.randn(gradient_dim)
        self.B = self.DEFAULT_COEFF_CENTER + self.COEFF_NOISE_SCALE * np.random.randn(gradient_dim)
    
    def _init_predictor(self):
        """Initialize the predictor weights."""
        # Simple linear predictor from projection_dim to projection_dim
        np.random.seed(self.random_seed)
        self.predictor_weights = np.random.randn(self.projection_dim, self.projection_dim) * self.PREDICTOR_INIT_SCALE
    
    def _project(self, gradient: np.ndarray) -> np.ndarray:
        """
        Project gradient to lower-dimensional feature space (f_proj).
        Uses random projection for dimensionality reduction.
        """
        gradient_dim = len(gradient)
        
        if gradient_dim <= self.projection_dim:
            # If gradient is already smaller than projection dim, just pad
            projected = np.zeros(self.projection_dim)
            projected[:gradient_dim] = gradient
            return projected
        
        # Use deterministic random projection matrix
        np.random.seed(self.random_seed)
        projection_matrix = np.random.randn(self.projection_dim, gradient_dim) / np.sqrt(gradient_dim)
        return projection_matrix @ gradient
    
    def _predict(self, projected_input: np.ndarray) -> np.ndarray:
        """
        Predict the next gradient using the predictor (f_pred).
        Simple linear predictor.
        """
        if self.predictor_weights is None:
            self._init_predictor()
        return self.predictor_weights @ projected_input
    
    def _compute_input(self, client_gradient: np.ndarray, global_gradient: np.ndarray) -> np.ndarray:
        """
        Compute input for prediction: g_input = A ⊙ g_k + B ⊙ g
        """
        return self.A * client_gradient + self.B * global_gradient
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < self.NORM_EPSILON or norm_b < self.NORM_EPSILON:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _get_client_history_gradient(self, client_id: str, target_round: int, 
                                     global_gradient: np.ndarray) -> np.ndarray:
        """
        Get historical gradient for a client at a specific round.
        If client wasn't selected in that round, use global gradient (exception handling).
        """
        if client_id not in self.client_history:
            return global_gradient
        
        for round_num, gradient in self.client_history[client_id]:
            if round_num == target_round:
                return gradient
        
        # Client wasn't in C_opt for that round, use global gradient
        return global_gradient
    
    def _train_predictor(self, client_id: str, flattened_gradient: np.ndarray):
        """
        Train predictor and coefficient matrices using historical data.
        Uses simple gradient descent optimization.
        """
        if len(self.global_history) < 2:
            return
        
        # Get historical rounds
        global_rounds = list(self.global_history)
        
        for i in range(len(global_rounds) - 1):
            round_tau, global_grad_tau = global_rounds[i]
            round_tau_plus_1, global_grad_tau_plus_1 = global_rounds[i + 1]
            
            # Get client gradient at round tau (with exception handling)
            client_grad_tau = self._get_client_history_gradient(
                client_id, round_tau, global_grad_tau
            )
            
            # Get client gradient at round tau+1 for target
            client_grad_tau_plus_1 = self._get_client_history_gradient(
                client_id, round_tau_plus_1, global_grad_tau_plus_1
            )
            
            # Compute input: g_input = A ⊙ g_k + B ⊙ g
            g_input = self._compute_input(client_grad_tau, global_grad_tau)
            
            # Project to feature space
            p_input = self._project(g_input)
            
            # Predict
            predicted = self._predict(p_input)
            
            # Target: projected next gradient
            target = self._project(client_grad_tau_plus_1)
            
            # Simple gradient descent update for predictor
            error = predicted - target
            grad_W = np.outer(error, p_input)
            
            # Add gradient clipping to prevent exploding gradients/NaNs
            grad_norm = np.linalg.norm(grad_W)
            if grad_norm > 1.0:
                grad_W = grad_W * (1.0 / grad_norm)
                
            self.predictor_weights -= self.learning_rate * grad_W
    
    def _compute_client_similarity(self, client_id: str, 
                                   current_gradient: np.ndarray) -> float:
        """
        Compute similarity between predicted and actual gradient for a client.
        """
        if len(self.global_history) < 1:
            return 1.0  # Default to full trust if no history
        
        # Get previous round's global gradient
        prev_round, prev_global_gradient = self.global_history[-1]
        
        # Get previous round's client gradient
        prev_client_gradient = self._get_client_history_gradient(
            client_id, prev_round, prev_global_gradient
        )
        
        # Compute input from previous round
        g_input = self._compute_input(prev_client_gradient, prev_global_gradient)
        
        # Project and predict
        p_input = self._project(g_input)
        predicted = self._predict(p_input)
        
        # Project actual current gradient
        actual_projected = self._project(current_gradient)
        
        # Compute cosine similarity
        similarity = self._cosine_similarity(predicted, actual_projected)
        
        return similarity
    
    def _update_history(self, selected_clients: Dict[str, np.ndarray], 
                        global_gradient: np.ndarray):
        """Update historical gradient storage."""
        current_round = self.round_number
        
        # Update client history
        for client_id, gradient in selected_clients.items():
            if client_id not in self.client_history:
                self.client_history[client_id] = deque(maxlen=self.history_size)
            self.client_history[client_id].append((current_round, gradient))
        
        # Update global history
        self.global_history.append((current_round, global_gradient))
    
    def aggregate_updates(self, 
                         client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
                        ) -> Tuple[List[np.ndarray], List[ExplainableDecision]]:
        """
        Aggregate client updates using VERT algorithm.
        
        The algorithm:
        1. For each client, train predictor using historical data
        2. Predict what gradient the client should produce
        3. Compute cosine similarity between predicted and actual
        4. Select top-κ clients with highest similarity
        5. Average their gradients
        """
        if not client_updates:
            return None, []
        
        client_ids = list(client_updates.keys())
        n_clients = len(client_ids)
        
        # Extract and flatten gradients
        flattened_gradients = {}
        param_shapes = None
        
        for client_id in client_ids:
            parameters, _, _ = client_updates[client_id]
            if param_shapes is None:
                param_shapes = [param.shape for param in parameters]
            flattened_gradients[client_id] = self._flatten_parameters(parameters)
        
        gradient_dim = len(next(iter(flattened_gradients.values())))
        
        # Initialize coefficient matrices if needed
        if self.A is None or len(self.A) != gradient_dim:
            self._init_coefficient_matrices(gradient_dim)
        
        # Check if we have enough history to use VERT
        if len(self.global_history) < self.min_history_rounds:
            # Fall back to simple averaging until we have enough history
            return self._fallback_aggregation(client_updates, 
                                              "Insufficient history for VERT, using FedAvg")
        
        # Train predictor for each client and compute similarities
        similarities = {}
        
        for client_id in client_ids:
            # Train predictor using historical data
            self._train_predictor(client_id, flattened_gradients[client_id])
            
            # Compute similarity
            similarity = self._compute_client_similarity(
                client_id, flattened_gradients[client_id]
            )
            similarities[client_id] = similarity
        
        # Select top-κ clients with highest similarity
        kappa = min(self.kappa, n_clients)  # Can't select more than available
        sorted_clients = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        selected_clients = [client_id for client_id, _ in sorted_clients[:kappa]]
        
        # Update optimal clients set
        self.optimal_clients = set(selected_clients)
        
        # Average selected gradients
        selected_gradients = [flattened_gradients[client_id] for client_id in selected_clients]
        aggregated_flat = np.mean(selected_gradients, axis=0)
        
        # Update history with selected clients' gradients
        selected_gradients_dict = {
            client_id: flattened_gradients[client_id] 
            for client_id in selected_clients
        }
        self._update_history(selected_gradients_dict, aggregated_flat)
        
        # Create explainable decisions
        decisions = []
        
        for client_id in client_ids:
            similarity = similarities[client_id]
            is_selected = client_id in selected_clients
            rank = next((i + 1 for i, (cid, _) in enumerate(sorted_clients) if cid == client_id), 0)
            
            if is_selected:
                decision = ExplainableDecision(
                    decision="accept",
                    confidence=similarity,
                    reasoning=f"Selected by VERT (similarity: {similarity:.4f}, rank: {rank}/{n_clients})",
                    evidence={
                        'similarity': float(similarity),
                        'rank': rank,
                        'selected': True,
                        'kappa': kappa,
                        'method': 'vert'
                    }
                )
            else:
                decision = ExplainableDecision(
                    decision="reject",
                    confidence=1.0 - similarity,
                    reasoning=f"Rejected by VERT (similarity: {similarity:.4f}, rank: {rank}/{n_clients}, below top-{kappa})",
                    evidence={
                        'similarity': float(similarity),
                        'rank': rank,
                        'selected': False,
                        'kappa': kappa,
                        'method': 'vert'
                    }
                )
            
            decisions.append(decision)
        
        # Unflatten aggregated parameters
        aggregated_params = self._unflatten_parameters(aggregated_flat, param_shapes)
        
        self.increment_round()
        
        return aggregated_params, decisions
    
    def _fallback_aggregation(self, 
                             client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]],
                             reason: str
                            ) -> Tuple[List[np.ndarray], List[ExplainableDecision]]:
        """Fallback to simple FedAvg when VERT conditions aren't met."""
        # Flatten all gradients for history storage
        flattened_gradients = {}
        param_shapes = None
        weighted_updates = []
        total_samples = 0
        
        for client_id, (parameters, num_samples, _) in client_updates.items():
            if param_shapes is None:
                param_shapes = [param.shape for param in parameters]
            flattened_gradients[client_id] = self._flatten_parameters(parameters)
            weighted_updates.append((parameters, num_samples))
            total_samples += num_samples
        
        if weighted_updates and total_samples > 0:
            aggregated_params = []
            num_params = len(weighted_updates[0][0])
            
            for param_idx in range(num_params):
                weighted_sum = sum(params[param_idx] * weight 
                                 for params, weight in weighted_updates)
                aggregated_params.append(weighted_sum / total_samples)
            
            # Compute and store aggregated gradient for history
            aggregated_flat = self._flatten_parameters(aggregated_params)
        else:
            aggregated_params = None
            aggregated_flat = None
        
        # Update history even during fallback
        if aggregated_flat is not None:
            self._update_history(flattened_gradients, aggregated_flat)
        
        # Create decisions
        decisions = []
        for client_id in client_updates.keys():
            decision = ExplainableDecision(
                decision="accept",
                confidence=0.5,
                reasoning=f"Fallback to FedAvg ({reason}). History size: {len(self.global_history)}/{self.min_history_rounds}",
                evidence={
                    'fallback': True, 
                    'reason': reason,
                    'history_size': len(self.global_history),
                    'min_required': self.min_history_rounds,
                    'method': 'fedavg_fallback'
                }
            )
            decisions.append(decision)
        
        self.increment_round()
        return aggregated_params, decisions
    
    def get_defence_description(self) -> str:
        return f"VERT Defence Strategy (κ={self.kappa}, history_size={self.history_size})"
