# src/defences/cognitive_defence_posg.py
"""
CogDef-POSG: Deep Reinforcement Learning–based Partially Observable
             Stochastic Game (POSG) Defender for Federated Learning.

Replaces the heuristic OODA loop in ``cognitive_defence.py`` with:
  1. **Rich feature engineering** – cosine similarity to the previous global
     model plus Fisher-Information Trace of each client update.
  2. **GRU-based belief tracking** – per-client hidden states capture temporal
     gradient signatures (solves the "boiling-frog" problem).
  3. **Soft Actor-Critic (SAC) policy** – outputs continuous aggregation
     weights a ∈ [0,1]^N to surgically down-weight adversarial clients.
  4. **Long-horizon reward** –
         R = α · ΔValAcc  −  β · H(b)  −  γ · Ω
     penalises uncertainty and anomalous model drift.

References
----------
*  Palit (2025) – baseline DQN defence (myopic; fails against stealth).
*  Xie et al. (2025) – multi-round consistency attacks.
*  Haarnoja et al. (2018) – Soft Actor-Critic.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base_defence import Basedefence
from .client_tracker import ClientTracker
from .sac_agent import SACAgent
from ..utils.logging_utils import ExplainableDecision

logger = logging.getLogger(__name__)


# ======================================================================
# Feature-engineering helpers
# ======================================================================

def _flatten(params: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of parameter arrays into a single 1-D vector."""
    return np.concatenate([p.ravel() for p in params])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(dot / norm)


def _fisher_information_trace(update: List[np.ndarray]) -> float:
    """
    Approximate the *trace* of the Fisher Information Matrix of an update.

    For an update Δθ the empirical Fisher trace is  Tr(F) ≈ Σ_j (Δθ_j)².
    This equals the squared L2 norm of the flattened update – computationally
    free and a strong signal for large-gradient poisoning attacks.
    """
    return float(sum(np.sum(p ** 2) for p in update))


# ======================================================================
# Online Welford normalizer
# ======================================================================

class _WelfordNormalizer:
    """
    Incremental mean/variance tracker (Welford 1962) for observation normalization.

    Maintains per-feature running statistics so that every feature fed to the
    GRU has roughly zero mean and unit variance – crucial for stable gradient
    flow when feature scales span orders of magnitude (e.g. Fisher-trace vs
    normalized sample-count).
    """

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.ones(dim, dtype=np.float64)   # initialise to 1 so std≥1 before any data

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return z-scored x; safe (1e-6 floor on std)."""
        if self.n < 2:
            return x.astype(np.float32)
        std = np.sqrt(self.M2 / self.n) + 1e-6
        return ((x - self.mean) / std).astype(np.float32)


# ======================================================================
# Reward helpers
# ======================================================================

def compute_reward(
    val_acc_before: float,
    val_acc_after: float,
    belief_entropy: float,
    model_divergence: float,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> float:
    """
    Multi-objective long-horizon reward.

    R = α · ΔValAcc  −  β · H(b)  −  γ · Ω

    Parameters
    ----------
    val_acc_before, val_acc_after : float
        Validation accuracy before and after aggregation.
    belief_entropy : float
        Average Shannon entropy of belief states (high → uncertain).
    model_divergence : float
        L2 distance between global model before and after the round.
    alpha, beta, gamma : float
        Coefficients weighting accuracy gain, uncertainty, and divergence.
    """
    delta_acc = val_acc_after - val_acc_before
    return alpha * delta_acc - beta * belief_entropy - gamma * model_divergence


# ======================================================================
# Main Defence
# ======================================================================

class CognitiveDefencePOSG(Basedefence):
    """
    POSG-based cognitive defence implementing:
        Observe  → rich feature extraction (norms, cosine-sim, Fisher trace)
        Orient   → GRU belief-state update per client
        Decide   → SAC policy over concatenated belief states
        Act      → weighted federated aggregation

    The defender learns over many rounds to *isolate* clients whose belief
    trajectory indicates sustained adversarial behaviour.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_clients: int = 20,
        obs_dim: int = 6,
        belief_hidden_dim: int = 64,
        sac_hidden_dims: list[int] | None = None,
        lr: float = 1e-3,  # Increased from 3e-4 for faster convergence
        gamma: float = 0.95,  # Reduced from 0.99 for medium-horizon rewards
        reward_alpha: float = 10.0,  # Increased from 1.0 to emphasize accuracy
        reward_beta: float = 0.05,  # Reduced from 0.3 to decrease uncertainty penalty
        reward_gamma: float = 0.2,
        buffer_capacity: int = 1000,  # Reduced from 50k to match realistic sample count
        batch_size: int = 16,  # Reduced from 64 to allow updates with small buffer
        device: str = "cpu",
        history_size: int = 200,
        warmup_rounds: int = 10,  # Extended from 5 for better SAC initialization
    ):
        """
        Parameters
        ----------
        max_clients : int
            Maximum number of simultaneous clients.  Determines the SAC
            action dimension (one weight per client slot).
        obs_dim : int
            Dimensionality of the per-client observation vector:
              [total_norm, avg_norm, max_norm, cosine_sim, fisher_trace, num_samples]
        belief_hidden_dim : int
            GRU hidden-state width.
        sac_hidden_dims : list[int]
            Hidden layers for actor/critic MLPs.
        reward_alpha, reward_beta, reward_gamma : float
            Coefficients for the long-horizon reward function.
        """
        super().__init__(history_size=history_size)

        self.max_clients = max_clients
        self.obs_dim = obs_dim
        self.device = device
        self._belief_hidden_dim = belief_hidden_dim

        # Reward coefficients
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma

        # ---- Online observation normalizer (Welford) ----
        # Normalizes the 6-dim observation before GRU input, ensuring stable
        # gradient flow regardless of model/dataset-dependent feature scales.
        self._obs_norm = _WelfordNormalizer(obs_dim)

        # ---- Belief tracker (GRU) ----
        self.tracker = ClientTracker(obs_dim=obs_dim, hidden_dim=belief_hidden_dim)

        # ---- SAC agent ----
        # Compact state: [mean_belief || std_belief] over active clients only.
        # Dimension = 2 * hidden_dim, independent of max_clients.
        # This replaces the naive max_clients*hidden_dim concatenation which
        # was 6400-dimensional and almost entirely zeros — killing learning.
        state_dim = 2 * belief_hidden_dim
        action_dim = max_clients
        sac_hidden_dims = sac_hidden_dims or [256, 256]

        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=sac_hidden_dims,
            lr_actor=lr,
            lr_critic=lr,
            lr_alpha=lr,
            gamma=gamma,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            device=device,
        )

        # Warm-up: use heuristic weights for first N rounds while buffer fills.
        # The GRU and replay buffer are still updated during warm-up.
        self.warmup_rounds = warmup_rounds

        # ---- Internal bookkeeping ----
        self._global_model_flat: Optional[np.ndarray] = None  # previous global model
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[np.ndarray] = None
        self._prev_val_acc: Optional[float] = None
        # Reward stabilization: exponential moving average of validation accuracy
        self._acc_ema: float = 0.0
        self._prev_acc_ema: float = 0.0  # Previous EMA for delta calculation
        self._acc_ema_alpha: float = 0.3  # Smoothing factor (0.3 = fast adaptation)
        self._active_client_ids: List[str] = []
        self._client_slot_map: Dict[str, int] = {}  # client_id → slot index
        self._round_diagnostics: deque = deque(maxlen=history_size)
        # Cache of flattened updates for the current round (used by warmup heuristic)
        self._current_flattened_updates: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Slot management (maps variable client IDs → fixed-size vectors)
    # ------------------------------------------------------------------

    def _ensure_slot(self, client_id: str) -> int:
        if client_id not in self._client_slot_map:
            if len(self._client_slot_map) >= self.max_clients:
                # Evict the oldest slot
                oldest = next(iter(self._client_slot_map))
                del self._client_slot_map[oldest]
                self.tracker.reset_client(oldest)
            self._client_slot_map[client_id] = len(self._client_slot_map)
        return self._client_slot_map[client_id]

    # ------------------------------------------------------------------
    # OODA: Observe
    # ------------------------------------------------------------------

    def observe(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]],
    ) -> Dict[str, np.ndarray]:
        """
        Extract a rich observation vector for each client.

        Features (per client):
            0  total_norm       – L2 norm of full update
            1  avg_norm         – mean per-layer norm
            2  max_norm         – max per-layer norm
            3  cosine_sim       – cosine similarity to previous global model
            4  fisher_trace     – Tr(F) ≈ ||Δθ||² (squared gradient magnitude)
            5  num_samples      – local dataset size (normalised)
        """
        observations: Dict[str, np.ndarray] = {}
        self._current_flattened_updates = {}  # reset each round

        # Normalisation reference for num_samples
        sample_counts = [ns for _, (_, ns, _) in client_updates.items()]
        max_samples = max(sample_counts) if sample_counts else 1.0

        for client_id, (parameters, num_samples, _metrics) in client_updates.items():
            param_norms = [float(np.linalg.norm(p)) for p in parameters]
            total_norm = sum(param_norms)
            avg_norm = total_norm / len(param_norms) if param_norms else 0.0
            max_norm = max(param_norms) if param_norms else 0.0

            flat = _flatten(parameters)

            # Cosine similarity against previous global model
            if self._global_model_flat is not None and len(self._global_model_flat) == len(flat):
                cos_sim = _cosine_similarity(flat, self._global_model_flat)
            else:
                cos_sim = 0.0  # first round – no reference

            fisher = _fisher_information_trace(parameters)

            raw_obs = np.array(
                [total_norm, avg_norm, max_norm, cos_sim, fisher, num_samples / max_samples],
                dtype=np.float64,
            )
            # Update running statistics and normalize
            self._obs_norm.update(raw_obs)
            obs = self._obs_norm.normalize(raw_obs)
            observations[client_id] = obs
            self._current_flattened_updates[client_id] = flat

        return observations

    # ------------------------------------------------------------------
    # OODA: Orient  (GRU belief update)
    # ------------------------------------------------------------------

    def orient(
        self, observations: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """
        Feed observations into the GRU tracker and build the SAC state.

        Returns
        -------
        beliefs : dict  client_id → belief tensor  (hidden_dim,)
        state   : ndarray  (max_clients * hidden_dim,)
        """
        beliefs: Dict[str, torch.Tensor] = {}
        self._active_client_ids = list(observations.keys())

        for cid, obs_vec in observations.items():
            self._ensure_slot(cid)
            obs_t = torch.from_numpy(obs_vec).float()
            belief = self.tracker.update(cid, obs_t)
            beliefs[cid] = belief

        # ---- Compact state: [mean_belief || std_belief] over active clients ----
        # Motivation: the naive max_clients×hidden_dim concatenation produces a
        # 6400-dim vector almost entirely zeros for sparse participation, which
        # drowns the gradient signal.  The sufficient statistic of the belief
        # distribution — its first two moments — is only 2×hidden_dim = 128-dim
        # and is always dense regardless of how many clients are active.
        b_stack = torch.stack(
            [beliefs[cid] for cid in self._active_client_ids], dim=0
        )  # (n_active, hidden_dim)
        mean_b = b_stack.mean(dim=0)            # (hidden_dim,)
        std_b = b_stack.std(dim=0) + 1e-6       # (hidden_dim,)
        state = torch.cat([mean_b, std_b], dim=-1).detach().cpu().numpy()  # (2*hidden_dim,)
        return beliefs, state

    # ------------------------------------------------------------------
    # OODA: Decide  (SAC policy)
    # ------------------------------------------------------------------

    def _heuristic_weights(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Multi-Krum scoring for Byzantine-robust client selection.

        **Why we replaced FLTrust-cosine:**
        - FLTrust fails when Byzantine fraction ≥ 30% (corrupts the majority)
        - Cosine similarity is vulnerable to coordinated direction attacks
        - Distance-based methods (Krum) are provably robust up to f < n/2

        **Multi-Krum Algorithm (Blanchard et al., 2017):**
        1. Compute pairwise L2 distances between all client updates
        2. For each client i, sum distances to m nearest neighbors (m = n-f-2)
        3. Select clients with smallest scores (closest to majority cluster)
        4. Assign high weight to selected, low weight to isolated

        **Theoretical guarantee**: Robust to f < n/2 Byzantine clients.
        With 40% Byzantine (f=0.4n < 0.5n), this should correctly isolate.
        """
        cids = list(observations.keys())
        n = len(cids)
        flats = [self._current_flattened_updates.get(c) for c in cids]

        if not all(f is not None for f in flats) or n < 3:
            return {c: 1.0 for c in cids}

        # ── Step 1: Estimate Byzantine fraction (conservative: 40%) ──────────
        f_est = max(1, int(np.ceil(0.40 * n)))
        m = max(1, n - f_est - 2)  # Number of nearest neighbors to consider

        # ── Step 2: Compute pairwise distance matrix ──────────────────────────
        # D[i,j] = ||update_i - update_j||_2^2
        mat = np.vstack([flats[i].astype(np.float64) for i in range(n)])
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = float(np.sum((mat[i] - mat[j]) ** 2))
                D[i, j] = dist_sq
                D[j, i] = dist_sq

        # ── Step 3: Krum score = sum of distances to m nearest neighbors ─────
        scores = np.zeros(n)
        for i in range(n):
            distances = np.delete(D[i], i)  # Remove self-distance (0)
            distances_sorted = np.sort(distances)
            scores[i] = np.sum(distances_sorted[:m])

        # ── Step 4: Select n-f-2 clients with lowest scores (most consensus) ──
        n_select = max(1, n - f_est - 2)
        selected_indices = np.argsort(scores)[:n_select]

        # ── Step 5: Assign weights ────────────────────────────────────────────
        weights: Dict[str, float] = {}
        num_isolated = 0
        for idx, cid in enumerate(cids):
            if idx in selected_indices:
                weights[cid] = 1.0  # Trusted (in majority cluster)
            else:
                weights[cid] = 0.1  # Isolated (likely Byzantine)
                num_isolated += 1

        # Log selection stats for debugging
        logger.debug(
            f"Multi-Krum: {n_select}/{n} selected, {num_isolated} isolated "
            f"(f_est={f_est}, m={m})"
        )

        return weights

    def decide(
        self,
        state: np.ndarray,
        beliefs: Dict[str, torch.Tensor],
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[ExplainableDecision], np.ndarray]:
        """
        Query the SAC agent for per-client aggregation weights.

        During ``warmup_rounds`` a norm-based heuristic is used instead of
        the untrained SAC policy, preventing the cold-start accuracy collapse.

        Returns
        -------
        decisions : dict  client_id → {action, weight_multiplier, reason}
        explanations : list[ExplainableDecision]
        raw_action : ndarray  (max_clients,) – full action vector for replay
        """
        in_warmup = self.round_number < self.warmup_rounds

        if in_warmup:
            # Heuristic weights – but still build a valid raw_action for the
            # replay buffer so transitions are stored from round 1.
            heuristic = self._heuristic_weights(observations)
            raw_action = self.agent.select_action(state, deterministic=False)
            # Override SAC weights with heuristic for actual aggregation
            for cid in self._active_client_ids:
                slot = self._client_slot_map[cid]
                raw_action[slot] = heuristic.get(cid, 1.0)
        else:
            raw_action = self.agent.select_action(state, deterministic=deterministic)

        decisions: Dict[str, Dict[str, Any]] = {}
        explanations: List[ExplainableDecision] = []

        for cid in self._active_client_ids:
            slot = self._client_slot_map[cid]
            weight = float(np.clip(raw_action[slot], 0.0, 1.0))

            # Interpret weight for logging
            phase = "warm-up heuristic" if in_warmup else "SAC policy"
            if weight < 0.2:
                label = "isolate"
                reasoning = (
                    f"{phase} assigned weight {weight:.3f} (< 0.2) — "
                    f"client is effectively isolated based on adverse belief trajectory."
                )
            elif weight < 0.5:
                label = "reduce_weight"
                reasoning = (
                    f"{phase} assigned weight {weight:.3f} — "
                    f"partial trust; monitoring for further adversarial signals."
                )
            else:
                label = "accept"
                reasoning = (
                    f"{phase} assigned weight {weight:.3f} (≥ 0.5) — "
                    f"belief state indicates benign behaviour."
                )

            belief_norm = float(beliefs[cid].detach().norm()) if cid in beliefs else 0.0

            decisions[cid] = {
                "action": label,
                "weight_multiplier": weight,
                "reason": reasoning,
            }

            explanations.append(
                ExplainableDecision(
                    decision=label,
                    confidence=weight,
                    reasoning=reasoning,
                    evidence={
                        "sac_weight": weight,
                        "belief_norm": belief_norm,
                        "slot_index": slot,
                        "round": self.round_number,
                    },
                )
            )

        return decisions, explanations, raw_action

    # ------------------------------------------------------------------
    # OODA: Act  (weighted aggregation)
    # ------------------------------------------------------------------

    def act(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]],
        decisions: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """
        Aggregate client updates using SAC-assigned weights.

        Returns
        -------
        aggregated_params : list[ndarray] or None
        aggregation_log : dict
        """
        weighted_updates: List[Tuple[List[np.ndarray], float]] = []
        total_weight = 0.0
        aggregation_log: Dict[str, Any] = {}

        for cid, (parameters, num_samples, _) in client_updates.items():
            if cid not in decisions:
                continue
            w = decisions[cid]["weight_multiplier"] * num_samples
            weighted_updates.append((parameters, w))
            total_weight += w

            aggregation_log[cid] = {
                "original_samples": num_samples,
                "sac_weight": decisions[cid]["weight_multiplier"],
                "effective_weight": float(w),
                "action_label": decisions[cid]["action"],
                "reputation": self.get_client_reputation(cid),
            }

        if weighted_updates and total_weight > 0:
            # ── Median-norm clipping ─────────────────────────────────────────
            # Cap each update's L2 norm to the per-round median norm.  This
            # bounds Byzantine amplification independent of detection accuracy.
            update_norms = [
                float(np.linalg.norm(_flatten(params)))
                for params, _ in weighted_updates
            ]
            clip_norm = float(np.median(update_norms))
            clipped_updates: List[Tuple[List[np.ndarray], float]] = []
            for params, w in weighted_updates:
                flat = _flatten(params)
                n = float(np.linalg.norm(flat))
                if n > clip_norm + 1e-8:
                    scale = clip_norm / n
                    shapes = [p.shape for p in params]
                    flat_c = flat * scale
                    reconstructed: List[np.ndarray] = []
                    offset = 0
                    for shape in shapes:
                        size = int(np.prod(shape))
                        reconstructed.append(flat_c[offset: offset + size].reshape(shape))
                        offset += size
                    clipped_updates.append((reconstructed, w))
                else:
                    clipped_updates.append((params, w))
            # ─────────────────────────────────────────────────────────────────
            num_params = len(clipped_updates[0][0])
            aggregated_params: List[np.ndarray] = []
            for idx in range(num_params):
                weighted_sum = sum(p[idx] * w for p, w in clipped_updates)
                aggregated_params.append(weighted_sum / total_weight)
        else:
            aggregated_params = None

        return aggregated_params, aggregation_log

    # ------------------------------------------------------------------
    # Main entry point – replaces heuristic OODA loop
    # ------------------------------------------------------------------

    def aggregate_updates(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]],
        val_acc: Optional[float] = None,
        deterministic: bool = False,
    ) -> Tuple[Optional[List[np.ndarray]], List[ExplainableDecision]]:
        """
        Full OODA loop backed by the POSG/SAC pipeline.

        Parameters
        ----------
        client_updates : dict
            Standard FL client updates mapping.
        val_acc : float, optional
            Current validation accuracy (used for reward computation).
            If ``None`` the learning step is skipped for this round.
        deterministic : bool
            If True the SAC policy uses its mode (evaluation).

        Returns
        -------
        aggregated_params : list[ndarray] or None
        explainable_decisions : list[ExplainableDecision]
        """
        # 1. Observe ---------------------------------------------------------
        observations = self.observe(client_updates)

        # 2. Orient  (GRU belief update → SAC state) -------------------------
        beliefs, state = self.orient(observations)

        # 3. Decide  (SAC policy query) --------------------------------------
        decisions, explanations, raw_action = self.decide(
            state, beliefs, observations, deterministic=deterministic
        )

        # 4. Act  (weighted aggregation) -------------------------------------
        aggregated_params, agg_log = self.act(client_updates, decisions)

        # 5. RL Learning step -------------------------------------------------
        if val_acc is not None and self._prev_state is not None:
            # ── Update exponential moving average of accuracy ────────────────────
            if self.round_number == 1:
                self._acc_ema = val_acc
                self._prev_acc_ema = val_acc
            else:
                self._prev_acc_ema = self._acc_ema
                self._acc_ema = (
                    self._acc_ema_alpha * val_acc +
                    (1.0 - self._acc_ema_alpha) * self._acc_ema
                )

            # Compute smoothed accuracy change (reduces noise in reward signal)
            delta_acc_smoothed = self._acc_ema - self._prev_acc_ema

            # Compute model divergence
            if aggregated_params is not None and self._global_model_flat is not None:
                new_flat = _flatten(aggregated_params)
                if len(new_flat) == len(self._global_model_flat):
                    divergence = float(np.linalg.norm(new_flat - self._global_model_flat))
                else:
                    divergence = 0.0
            else:
                divergence = 0.0

            belief_ent = float(
                self.tracker.belief_entropy(self._active_client_ids).item()
            )

            # Compute reward with smoothed accuracy and adjusted coefficients
            # Using delta_acc_smoothed instead of raw accuracy difference
            reward = (
                self.reward_alpha * delta_acc_smoothed -  # 10.0 * Δacc_smooth
                self.reward_beta * belief_ent -           # 0.05 * H(belief)
                self.reward_gamma * divergence            # 0.2 * ||Δθ||
            )

            self.agent.store_transition(
                state=self._prev_state,
                action=self._prev_action,
                reward=reward,
                next_state=state,
                done=False,
            )

            update_info = self.agent.update()
            if update_info is not None:
                # Clip GRU gradients to prevent explosion from noisy observations
                torch.nn.utils.clip_grad_norm_(self.tracker.parameters(), max_norm=1.0)

                logger.debug(
                    "SAC update – critic=%.4f actor=%.4f α=%.4f (reward=%.4f, Δacc_smooth=%.4f)",
                    update_info["critic_loss"],
                    update_info["actor_loss"],
                    update_info["alpha"],
                    reward,
                    delta_acc_smoothed,
                )

        # 6. Book-keeping for next round --------------------------------------
        self._prev_state = state
        self._prev_action = raw_action
        self._prev_val_acc = val_acc

        if aggregated_params is not None:
            self._global_model_flat = _flatten(aggregated_params)

        # Update reputation based on SAC weights
        for cid in self._active_client_ids:
            w = decisions[cid]["weight_multiplier"]
            current_rep = self.get_client_reputation(cid)
            # Soft exponential-moving-average reputation update
            new_rep = 0.9 * current_rep + 0.1 * w
            self.update_client_reputation(cid, new_rep - current_rep)

        self._round_diagnostics.append({
            "round": self.round_number,
            "active_clients": len(self._active_client_ids),
            "mean_weight": float(np.mean([
                decisions[c]["weight_multiplier"] for c in self._active_client_ids
            ])) if self._active_client_ids else 0.0,
            "val_acc": val_acc,
            "timestamp": datetime.now().isoformat(),
        })

        self.increment_round()
        return aggregated_params, explanations

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def set_global_model(self, global_params: List[np.ndarray]) -> None:
        """
        Provide the current global model so that cosine-similarity features
        can be computed.  Call this before each ``aggregate_updates``.
        """
        self._global_model_flat = _flatten(global_params)

    def get_defence_description(self) -> str:
        return (
            f"CogDef-POSG (SAC + GRU belief tracker, "
            f"max_clients={self.max_clients}, "
            f"obs_dim={self.obs_dim}, γ={self.agent.gamma})"
        )

    def save_checkpoint(self, path: str) -> None:
        """Persist the SAC agent and tracker weights."""
        import os, torch as _torch

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _torch.save(
            {
                "tracker": self.tracker.state_dict(),
                "sac": {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "critic_target": self.agent.critic_target.state_dict(),
                    "log_alpha": self.agent.log_alpha,
                },
                "round_number": self.round_number,
                "client_reputation": dict(self.client_reputation),
                "client_slot_map": dict(self._client_slot_map),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Restore from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.tracker.load_state_dict(ckpt["tracker"])
        self.agent.actor.load_state_dict(ckpt["sac"]["actor"])
        self.agent.critic.load_state_dict(ckpt["sac"]["critic"])
        self.agent.critic_target.load_state_dict(ckpt["sac"]["critic_target"])
        self.agent.log_alpha = ckpt["sac"]["log_alpha"]
        self.round_number = ckpt["round_number"]
        self.client_reputation = ckpt["client_reputation"]
        self._client_slot_map = ckpt["client_slot_map"]
        logger.info("Checkpoint loaded from %s (round %d)", path, self.round_number)
