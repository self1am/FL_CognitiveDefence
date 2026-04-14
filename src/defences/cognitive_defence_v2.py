# src/defences/cognitive_defence_v2.py
"""
CogDef v2: Cognitive-Inspired Multi-Signal Defence Framework

Enhanced OODA loop with multi-signal detection, adaptive threat response,
and MAPE-K self-tuning feedback loop.

This is the Sprint 1-5 implementation skeleton with Sprint 1 fully implemented.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from .base_defence import Basedefence
from ..utils.logging_utils import ExplainableDecision
from .client_tracker import ClientTracker


# =============================================================================
# Data structures
# =============================================================================

class ThreatLevel(Enum):
    """Global threat posture levels."""
    GREEN = "green"    # Normal operation — weighted FedAvg
    YELLOW = "yellow"  # Elevated — weighted FedAvg + clipping
    ORANGE = "orange"  # High — exclude suspects + trimmed mean
    RED = "red"        # Critical — Krum on trusted clients only


@dataclass
class ClientProfile:
    """Per-client tracking state across rounds."""
    reputation: float = 0.5       # Start neutral, earn trust
    norm_history: deque = field(default_factory=lambda: deque(maxlen=50))
    direction_history: deque = field(default_factory=lambda: deque(maxlen=50))
    anomaly_history: deque = field(default_factory=lambda: deque(maxlen=50))
    rounds_seen: int = 0
    consecutive_flags: int = 0
    consecutive_clean: int = 0
    last_anomaly_score: float = 0.0


@dataclass
class RoundDiagnostics:
    """Diagnostics collected during a single round for MAPE-K feedback."""
    round_number: int
    threat_level: ThreatLevel
    num_clients: int
    num_flagged: int
    num_rejected: int
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    detector_scores: Dict[str, List[float]] = field(default_factory=dict)
    fusion_weights: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Main Defence Class
# =============================================================================

class CognitiveDefenceV2(Basedefence):
    """
    Enhanced cognitive defence implementing:
    - Multi-signal OODA loop (Observe-Orient-Decide-Act)
    - MAPE-K self-tuning feedback
    - Adaptive threat posture with escalating response
    """

    def __init__(
        self,
        # Detection thresholds
        anomaly_threshold: float = 0.5,
        direction_weight: float = 0.40,
        norm_weight: float = 0.15,
        cluster_weight: float = 0.25,
        temporal_weight: float = 0.20,
        # Reputation parameters
        initial_reputation: float = 0.5,
        recovery_rate: float = 0.03,
        penalty_severity: float = 0.8,
        # Threat level thresholds
        yellow_threshold: float = 0.3,
        orange_threshold: float = 0.6,
        red_threshold: float = 0.8,
        # Posture escalation
        attack_fraction_trigger: float = 0.30,
        posture_cooldown_rounds: int = 5,
        # History
        history_size: int = 100,
        # Clipping (for YELLOW mode)
        clip_multiplier: float = 2.0,
        # Trimmed Mean (for ORANGE mode)
        trim_beta: float = 0.2,
        # Krum (for RED mode)
        krum_byzantine_fraction: float = 0.4,
        # MAPE-K
        enable_mape_k: bool = True,
        mape_k_window: int = 10,
    ):
        super().__init__(history_size=history_size)

        # Detection weights (must sum to ~1.0)
        self.detector_weights = {
            'norm': norm_weight,
            'direction': direction_weight,
            'cluster': cluster_weight,
            'temporal': temporal_weight,
        }

        # Thresholds
        self.anomaly_threshold = anomaly_threshold
        self.yellow_threshold = yellow_threshold
        self.orange_threshold = orange_threshold
        self.red_threshold = red_threshold

        # Reputation
        self.initial_reputation = initial_reputation
        self.recovery_rate = recovery_rate
        self.penalty_severity = penalty_severity

        # Per-client profiles
        self.client_profiles: Dict[str, ClientProfile] = {}

        # Global threat posture
        self.threat_level = ThreatLevel.GREEN
        self.attack_fraction_trigger = attack_fraction_trigger
        self.posture_cooldown_rounds = posture_cooldown_rounds
        self._rounds_since_escalation = 0
        self._flagged_fraction_history = deque(maxlen=20)

        # Global update history (for orientation baselines)
        self.global_mean_direction: Optional[np.ndarray] = None
        self.global_norm_history = deque(maxlen=history_size)
        self.round_diagnostics: List[RoundDiagnostics] = []

        # Aggregation mode params
        self.clip_multiplier = clip_multiplier
        self.trim_beta = trim_beta
        self.krum_byzantine_fraction = krum_byzantine_fraction

        # MAPE-K
        self.enable_mape_k = enable_mape_k
        self.mape_k_window = mape_k_window

        # GRU temporal belief tracker (POSG belief state)
        # obs_dim=3: [norm_score, direction_score, cluster_score] each round
        # hidden_dim=32: compact belief representation per client
        self.client_tracker = ClientTracker(obs_dim=3, hidden_dim=32)
        # Previous observation vectors — used to compute inter-round signal change
        self._prev_observations: Dict[str, torch.Tensor] = {}

    # -----------------------------------------------------------------
    # Client profile management
    # -----------------------------------------------------------------

    def _get_profile(self, client_id: str) -> ClientProfile:
        if client_id not in self.client_profiles:
            self.client_profiles[client_id] = ClientProfile(
                reputation=self.initial_reputation
            )
        return self.client_profiles[client_id]

    # =================================================================
    # OBSERVE — Multi-Signal Feature Extraction
    # =================================================================

    def observe(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract multiple signals from each client's update.

        Returns per-client observation dict with:
          - param_norms: per-layer L2 norms
          - total_norm: sum of layer norms
          - flattened: flattened parameter vector (for direction analysis)
          - per_layer_norms: list of per-layer norms
        """
        observations = {}
        all_flattened = []

        for client_id, (parameters, num_samples, metrics) in client_updates.items():
            flat = np.concatenate([p.flatten() for p in parameters])
            per_layer_norms = [float(np.linalg.norm(p)) for p in parameters]
            total_norm = float(np.linalg.norm(flat))

            observations[client_id] = {
                'flattened': flat,
                'per_layer_norms': per_layer_norms,
                'total_norm': total_norm,
                'num_samples': num_samples,
            }
            all_flattened.append(flat)

        # Compute per-round deltas: each client's deviation from the round mean.
        #
        # Why deltas and not full parameters:
        #   Flower sends full model weights, not gradients.  In early rounds the
        #   base model dominates all client vectors → cosine similarity ≈ 0.9999
        #   for every client (honest and attacker alike), completely masking the
        #   attack signal.  Subtracting the round mean removes the shared base
        #   model component and leaves only the client-specific update direction.
        #   Label-flip attackers then appear as a cluster pointing systematically
        #   away from the honest mean, which both the direction detector and the
        #   cluster detector can resolve.
        if all_flattened:
            mean_flat = np.mean(all_flattened, axis=0)
            for client_id in observations:
                observations[client_id]['delta'] = observations[client_id]['flattened'] - mean_flat

            # Compute consensus direction on deltas (geometric median is
            # still robust — a large minority of attackers cannot shift it
            # more than their fraction allows)
            all_deltas = np.stack([observations[cid]['delta'] for cid in observations])
            consensus = self._geometric_median(all_deltas)
            for client_id in observations:
                observations[client_id]['consensus_direction'] = consensus

        return observations

    def _geometric_median(self, points: np.ndarray, max_iter: int = 30, tol: float = 1e-6) -> np.ndarray:
        """
        Weiszfeld algorithm for geometric median.
        More robust than arithmetic mean — a single outlier can't shift it much.
        """
        y = np.median(points, axis=0)  # Initialize with coordinate-wise median
        for _ in range(max_iter):
            dists = np.linalg.norm(points - y, axis=1, keepdims=True)
            dists = np.maximum(dists, 1e-10)  # Avoid division by zero
            weights = 1.0 / dists
            y_new = np.sum(points * weights, axis=0) / np.sum(weights)
            if np.linalg.norm(y_new - y) < tol:
                break
            y = y_new
        return y

    def _majority_consensus(
        self,
        observations: Dict[str, Dict[str, Any]],
        cluster_scores: Dict[str, float],
    ) -> Optional[np.ndarray]:
        """
        Return the geometric median of the majority (honest) cluster's updates.

        The cluster detector splits clients into a minority (attackers, score > 0)
        and a majority (honest, score == 0).  When the split is detected, using
        only the majority updates as the direction reference prevents attacker
        gradients from corrupting the consensus — the root cause of the direction
        detector's failure against label-flip attacks at high attack fractions.

        Returns None when the cluster detector did not fire (all scores == 0),
        so the caller keeps the existing all-clients geometric median unchanged.
        """
        if not any(s > 0.0 for s in cluster_scores.values()):
            return None  # No cluster split detected — keep global consensus

        majority_ids = [cid for cid, s in cluster_scores.items() if s == 0.0]
        if len(majority_ids) < 3:
            return None  # Too few clients to estimate a reliable direction

        majority_flat = np.stack([observations[cid].get('delta', observations[cid]['flattened'])
                                  for cid in majority_ids])
        return self._geometric_median(majority_flat)

    # =================================================================
    # ORIENT — Multi-Detector Fusion
    # =================================================================

    def orient(
        self,
        observations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all detectors and fuse into per-client anomaly scores.
        """
        analysis = {}

        # Cluster detector runs at population level before the per-client loop
        cluster_scores = self._detect_cluster_outliers(observations)

        # If the cluster detector fired, recompute the consensus direction from
        # the majority (honest) cluster only.  Without this, _detect_direction_anomaly
        # uses a geometric median that is pulled ~40 % toward attacker updates,
        # which suppresses the direction signal against label-flip style attacks.
        majority_consensus = self._majority_consensus(observations, cluster_scores)
        if majority_consensus is not None:
            for obs_dict in observations.values():
                obs_dict['consensus_direction'] = majority_consensus

        for client_id, obs in observations.items():
            profile = self._get_profile(client_id)
            scores = {}

            # --- Detector 1: Norm Anomaly ---
            scores['norm'] = self._detect_norm_anomaly(obs, profile)

            # --- Detector 2: Direction Anomaly (CRITICAL NEW SIGNAL) ---
            scores['direction'] = self._detect_direction_anomaly(obs)

            # --- Detector 3: Cluster Outlier ---
            scores['cluster'] = cluster_scores.get(client_id, 0.0)

            # --- Detector 4: GRU Temporal Belief (POSG belief state) ---
            # Feed [norm, direction, cluster] into the GRU and measure how
            # rapidly this client's belief state is changing across rounds.
            scores['temporal'] = self._update_gru_belief(
                client_id,
                scores['norm'],
                scores['direction'],
                scores['cluster'],
            )

            # --- Fused anomaly score ---
            fused_score = sum(
                self.detector_weights[k] * scores[k]
                for k in self.detector_weights
            )

            # Classify threat level for this client
            if fused_score >= self.red_threshold:
                client_threat = ThreatLevel.RED
            elif fused_score >= self.orange_threshold:
                client_threat = ThreatLevel.ORANGE
            elif fused_score >= self.yellow_threshold:
                client_threat = ThreatLevel.YELLOW
            else:
                client_threat = ThreatLevel.GREEN

            analysis[client_id] = {
                'detector_scores': scores,
                'fused_score': float(fused_score),
                'client_threat': client_threat,
            }

            # Update profile history
            profile.norm_history.append(obs['total_norm'])
            profile.anomaly_history.append(fused_score)
            profile.rounds_seen += 1

        return analysis

    def _detect_norm_anomaly(self, obs: Dict, profile: ClientProfile) -> float:
        """
        Per-layer norm z-score anomaly detection (improved from v1).
        Uses per-layer analysis instead of just total norm.
        """
        if len(self.global_norm_history) < 3:
            return 0.0

        # Compare total norm to global distribution
        hist_norms = list(self.global_norm_history)
        mean_norm = np.mean(hist_norms)
        std_norm = np.std(hist_norms) + 1e-8

        z_score = abs(obs['total_norm'] - mean_norm) / std_norm

        # Normalize to [0, 1] using sigmoid-like mapping
        # z=2 → ~0.5, z=3 → ~0.73, z=4 → ~0.88
        score = 1.0 - 1.0 / (1.0 + np.exp(z_score - 2.5))
        return float(np.clip(score, 0.0, 1.0))

    def _detect_direction_anomaly(self, obs: Dict) -> float:
        """
        Cosine similarity to consensus direction.
        THIS IS THE KEY NEW SIGNAL that catches label-flip attacks.

        Returns score in [0, 1] where:
          0.0 = perfectly aligned with consensus (safe)
          1.0 = pointing opposite to consensus (malicious)
        """
        consensus = obs.get('consensus_direction')
        if consensus is None:
            return 0.0

        # Use delta (deviation from round mean) so the base model does not
        # dominate the direction comparison.  Fall back to full params only
        # if delta is not available (shouldn't happen after observe()).
        flat = obs.get('delta', obs['flattened'])
        norm_flat = np.linalg.norm(flat)
        norm_consensus = np.linalg.norm(consensus)

        if norm_flat < 1e-10 or norm_consensus < 1e-10:
            return 0.0

        cos_sim = np.dot(flat, consensus) / (norm_flat * norm_consensus)

        # Convert: cos_sim=1.0 → score=0.0, cos_sim=-1.0 → score=1.0
        score = (1.0 - cos_sim) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def _detect_temporal_anomaly(self, obs: Dict, profile: ClientProfile) -> float:
        """
        Track how consistent this client's behavior is over time.
        High variance in direction → suspicious.
        """
        if len(profile.direction_history) < 3:
            # Store current direction for future comparison
            flat = obs['flattened']
            norm = np.linalg.norm(flat)
            if norm > 1e-10:
                profile.direction_history.append(flat / norm)
            return 0.0

        # Compute current direction
        flat = obs['flattened']
        norm = np.linalg.norm(flat)
        if norm < 1e-10:
            return 0.0
        current_dir = flat / norm

        # Cosine similarities to recent directions
        recent_sims = []
        for past_dir in list(profile.direction_history)[-5:]:
            sim = np.dot(current_dir, past_dir)
            recent_sims.append(sim)

        profile.direction_history.append(current_dir)

        # High variance in similarities → erratic behavior
        if len(recent_sims) < 2:
            return 0.0

        variance = np.var(recent_sims)
        mean_sim = np.mean(recent_sims)

        # Low mean similarity + high variance = very suspicious
        score = (1.0 - mean_sim) * 0.5 + min(variance * 5.0, 0.5)
        return float(np.clip(score, 0.0, 1.0))

    def _update_gru_belief(
        self,
        client_id: str,
        norm_score: float,
        direction_score: float,
        cluster_score: float,
    ) -> float:
        """
        Update the GRU belief state for *client_id* and return a temporal
        anomaly score derived from the rate of change of that belief state.

        The GRU ingests a 3-dim observation vector
            x = [norm_score, direction_score, cluster_score]
        and produces a 32-dim hidden state h that summarises all past
        observations for this client.  This is the POSG belief state:
            b_i^t = GRU(x_i^t, b_i^{t-1})

        Temporal score = ||h_new - h_old|| / (||h_new|| + ||h_old|| + ε)

        Interpretation:
          - Honest clients have stable behaviour → smooth belief evolution → low score
          - Adaptive attackers switch strategy across rounds → volatile belief → high score
          - Round 1 always returns 0 (no prior state to compare against)

        No training is required.  The GRU's recurrent gating acts as a
        stateful low-pass filter: persistent anomalous signals accumulate in
        h while transient noise is suppressed.
        """
        obs_vec = torch.tensor(
            [norm_score, direction_score, cluster_score], dtype=torch.float32
        )

        # Update the GRU — this accumulates the POSG belief state h_i^t
        # even though we derive the score from the observation layer below.
        self.client_tracker.update(client_id, obs_vec)

        # Temporal score = inter-round change in the detection signals themselves.
        #
        # Why observation-level rather than hidden-state-level:
        #   A randomly-initialised GRU converges quickly for ALL clients,
        #   so ||Δh|| is uninformative without domain training.
        #   The observation vector [norm, direction, cluster] already encodes
        #   what we care about — an adaptive attacker who switches strategy
        #   produces large swings in these signals round-to-round; an honest
        #   client with stable behaviour does not.
        #
        # The GRU hidden state h_i^t is still the canonical belief
        # representation used in the POSG formulation (and available for
        # downstream use / future training); the score here is a
        # training-free proxy derived from the GRU's inputs.
        prev_obs = self._prev_observations.get(client_id)
        self._prev_observations[client_id] = obs_vec.detach()

        if prev_obs is None:
            return 0.0

        # L2 distance between consecutive observation vectors,
        # normalised by the theoretical maximum change (all signals 0↔1 → √3)
        obs_delta = torch.norm(obs_vec - prev_obs).item()
        return float(np.clip(obs_delta / (3.0 ** 0.5), 0.0, 1.0))

    def _detect_cluster_outliers(
        self, observations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Population-level bimodal cluster detection via PCA + gap statistic.

        Coordinated attackers each submit updates that may individually look
        plausible, but together they form a distinct sub-cluster in the update
        space.  We find this by projecting onto the top PCA axes (which
        maximise inter-group variance) and looking for a significant gap in
        the sorted projections — the signature of two clusters.

        Gap criterion: a split is only declared when the largest gap between
        consecutive sorted projections exceeds 20 % of the total data range.
        Below that threshold, the distribution is treated as unimodal and all
        scores return 0.

        Returns per-client score in [0, 1]:
          0   = clearly in the majority (honest) cluster
          1   = at the far end of the minority (attacker) cluster
        """
        from sklearn.decomposition import PCA

        client_ids = list(observations.keys())
        n = len(client_ids)

        if n < 6:
            return {cid: 0.0 for cid in client_ids}

        # Use deltas for clustering — same reason as direction detector:
        # full parameters are dominated by the shared base model and produce
        # near-identical vectors for all clients.  Deltas isolate the client-
        # specific update direction, making the attacker cluster visible.
        X = np.stack([observations[cid].get('delta', observations[cid]['flattened'])
                      for cid in client_ids])
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(row_norms, 1e-10)

        # Project onto top-3 PCs (captures the three most discriminative directions)
        n_components = min(3, n - 1, X.shape[1])
        if n_components < 1:
            return {cid: 0.0 for cid in client_ids}

        try:
            pca = PCA(n_components=n_components, svd_solver='randomized', random_state=0)
            X_reduced = pca.fit_transform(X)   # (n, n_components)
        except Exception:
            return {cid: 0.0 for cid in client_ids}

        # Check each PC for a significant bimodal gap; keep the most pronounced one
        best_scores = np.zeros(n)
        best_gap_ratio = 0.0

        for k in range(n_components):
            proj = X_reduced[:, k]
            sorted_vals = np.sort(proj)
            gaps = np.diff(sorted_vals)

            if len(gaps) == 0:
                continue

            max_gap_idx = int(np.argmax(gaps))
            max_gap = gaps[max_gap_idx]
            data_range = sorted_vals[-1] - sorted_vals[0]

            if data_range < 1e-10:
                continue

            gap_ratio = max_gap / data_range

            # Only act on clear bimodal splits (≥ 20 % of range) and only if
            # this PC is more discriminative than any we've seen so far
            if gap_ratio < 0.20 or gap_ratio <= best_gap_ratio:
                continue

            best_gap_ratio = gap_ratio
            split_val = (sorted_vals[max_gap_idx] + sorted_vals[max_gap_idx + 1]) / 2.0

            n_below = int(np.sum(proj <= split_val))
            n_above = n - n_below

            # Minority cluster is the smaller group; score by distance from split
            if n_below <= n_above:
                below_vals = proj[proj <= split_val]
                scale = split_val - (float(below_vals.min()) if len(below_vals) else split_val) + 1e-10
                best_scores = np.where(proj <= split_val, (split_val - proj) / scale, 0.0)
            else:
                above_vals = proj[proj > split_val]
                scale = (float(above_vals.max()) if len(above_vals) else split_val) - split_val + 1e-10
                best_scores = np.where(proj > split_val, (proj - split_val) / scale, 0.0)

        return {client_ids[i]: float(np.clip(best_scores[i], 0.0, 1.0)) for i in range(n)}

    # =================================================================
    # DECIDE — Adaptive Threat Assessment
    # =================================================================

    def decide(
        self,
        analysis: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[ExplainableDecision]]:
        """
        Make per-client decisions and update global threat posture.
        """
        decisions = {}
        explainable = []

        # Count flagged clients for posture assessment.
        # Include YELLOW+ so that attacks detected at the lower threat tier
        # (e.g. label-flip, which produces YELLOW scores) still trigger
        # posture escalation.  Without this, the posture stays GREEN and
        # peace-mode FedAvg is used regardless of how many clients are flagged.
        num_flagged = sum(
            1 for a in analysis.values()
            if a['client_threat'] in (ThreatLevel.YELLOW, ThreatLevel.ORANGE, ThreatLevel.RED)
        )
        flagged_fraction = num_flagged / max(len(analysis), 1)
        self._flagged_fraction_history.append(flagged_fraction)

        # Update global threat posture
        self._update_threat_posture()

        for client_id, client_analysis in analysis.items():
            profile = self._get_profile(client_id)
            fused_score = client_analysis['fused_score']
            client_threat = client_analysis['client_threat']

            # Update reputation
            if client_threat in (ThreatLevel.ORANGE, ThreatLevel.RED):
                # Full penalty: severe and immediate
                penalty = self.penalty_severity * fused_score
                profile.reputation = max(0.0, profile.reputation * (1.0 - penalty))
                profile.consecutive_flags += 1
                profile.consecutive_clean = 0
            elif client_threat == ThreatLevel.YELLOW:
                # Light penalty: 30 % of full severity.
                # Prevents reputation from growing while a client is persistently
                # suspicious — without this, label-flip attackers at YELLOW get
                # rewarded each round and their influence increases over time.
                light_penalty = self.penalty_severity * 0.3 * fused_score
                profile.reputation = max(0.0, profile.reputation * (1.0 - light_penalty))
                profile.consecutive_flags += 1
                profile.consecutive_clean = 0
            else:
                # GREEN: reward good behaviour
                bonus = self.recovery_rate * (1.0 - profile.reputation)
                profile.reputation = min(1.0, profile.reputation + bonus)
                profile.consecutive_clean += 1
                profile.consecutive_flags = 0

            profile.last_anomaly_score = fused_score

            # Determine action based on global posture + client threat
            action, weight = self._determine_action(
                client_threat, profile, self.threat_level
            )

            decisions[client_id] = {
                'action': action,
                'weight_multiplier': weight,
                'client_threat': client_threat.value,
                'fused_score': fused_score,
                'reputation': profile.reputation,
            }

            # Build explainable decision
            decision = ExplainableDecision(
                decision=action,
                confidence=fused_score if action != 'accept' else 1.0 - fused_score,
                reasoning=(
                    f"Client threat={client_threat.value}, "
                    f"fused_score={fused_score:.3f}, "
                    f"reputation={profile.reputation:.3f}, "
                    f"action={action}, weight={weight:.3f}, "
                    f"global_posture={self.threat_level.value}"
                ),
                evidence={
                    'detector_scores': client_analysis['detector_scores'],
                    'fused_score': fused_score,
                    'reputation': profile.reputation,
                    'threat_level': client_threat.value,
                    'global_posture': self.threat_level.value,
                    'consecutive_flags': profile.consecutive_flags,
                }
            )
            explainable.append(decision)

        return decisions, explainable

    def _determine_action(
        self,
        client_threat: ThreatLevel,
        profile: ClientProfile,
        global_posture: ThreatLevel
    ) -> Tuple[str, float]:
        """
        Determine action and weight based on client threat + global posture.

        Returns (action_string, weight_multiplier).
        """
        # RED clients are always rejected in ORANGE+ posture
        if client_threat == ThreatLevel.RED:
            if global_posture in (ThreatLevel.ORANGE, ThreatLevel.RED):
                return 'reject', 0.0
            else:
                return 'reduce_weight', max(profile.reputation * 0.1, 0.01)

        # ORANGE clients
        if client_threat == ThreatLevel.ORANGE:
            if global_posture == ThreatLevel.RED:
                return 'reject', 0.0
            elif global_posture == ThreatLevel.ORANGE:
                return 'reduce_weight', max(profile.reputation * 0.3, 0.01)
            else:
                return 'reduce_weight', max(profile.reputation * 0.5, 0.05)

        # YELLOW clients
        if client_threat == ThreatLevel.YELLOW:
            return 'reduce_weight', max(profile.reputation * 0.7, 0.1)

        # GREEN clients
        return 'accept', min(profile.reputation, 1.0)

    def _update_threat_posture(self):
        """
        Update global threat posture based on recent flagged fractions.
        Implements hysteresis (easier to escalate, harder to de-escalate).
        """
        if len(self._flagged_fraction_history) < 3:
            return

        recent_avg = np.mean(list(self._flagged_fraction_history)[-5:])
        self._rounds_since_escalation += 1

        # Escalation (fast)
        if recent_avg >= 0.5:
            self.threat_level = ThreatLevel.RED
            self._rounds_since_escalation = 0
        elif recent_avg >= self.attack_fraction_trigger:
            if self.threat_level.value < ThreatLevel.ORANGE.value:
                self.threat_level = ThreatLevel.ORANGE
                self._rounds_since_escalation = 0
        elif recent_avg >= 0.15:
            if self.threat_level == ThreatLevel.GREEN:
                self.threat_level = ThreatLevel.YELLOW
                self._rounds_since_escalation = 0

        # De-escalation (slow — requires cooldown)
        if self._rounds_since_escalation >= self.posture_cooldown_rounds:
            if recent_avg < 0.10 and self.threat_level != ThreatLevel.GREEN:
                # Step down one level
                levels = [ThreatLevel.GREEN, ThreatLevel.YELLOW,
                          ThreatLevel.ORANGE, ThreatLevel.RED]
                current_idx = levels.index(self.threat_level)
                if current_idx > 0:
                    self.threat_level = levels[current_idx - 1]
                    self._rounds_since_escalation = 0

    # =================================================================
    # ACT — Threat-Proportional Aggregation
    # =================================================================

    def act(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]],
        decisions: Dict[str, Dict[str, Any]],
        observations: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """
        Aggregate updates using the aggregation mode appropriate for
        the current global threat posture.
        """
        # Filter out rejected clients
        active_updates = {}
        for client_id, (params, n_samples, metrics) in client_updates.items():
            if decisions.get(client_id, {}).get('action') != 'reject':
                weight = decisions.get(client_id, {}).get('weight_multiplier', 1.0)
                active_updates[client_id] = (params, n_samples, metrics, weight)

        if not active_updates:
            # All clients rejected — extreme case, use geometric median of all
            return self._geometric_median_aggregate(client_updates), {
                'mode': 'emergency_geometric_median',
                'reason': 'all clients rejected'
            }

        # Select aggregation mode based on threat posture
        if self.threat_level == ThreatLevel.GREEN:
            result = self._aggregate_peace(active_updates)
            mode = 'peace_weighted_fedavg'
        elif self.threat_level == ThreatLevel.YELLOW:
            result = self._aggregate_vigilant(active_updates, observations)
            mode = 'vigilant_clipped_fedavg'
        elif self.threat_level == ThreatLevel.ORANGE:
            result = self._aggregate_defensive(active_updates)
            mode = 'defensive_trimmed_mean'
        else:  # RED
            result = self._aggregate_lockdown(active_updates)
            mode = 'lockdown_krum'

        agg_log = {
            'mode': mode,
            'threat_level': self.threat_level.value,
            'num_active': len(active_updates),
            'num_rejected': len(client_updates) - len(active_updates),
        }

        return result, agg_log

    def _aggregate_peace(
        self,
        active_updates: Dict[str, Tuple[List[np.ndarray], int, Any, float]]
    ) -> Optional[List[np.ndarray]]:
        """Mode 1: Reputation-weighted FedAvg."""
        weighted = []
        total_weight = 0.0

        for client_id, (params, n_samples, _, rep_weight) in active_updates.items():
            w = rep_weight * n_samples
            weighted.append((params, w))
            total_weight += w

        if total_weight == 0:
            return None

        num_params = len(weighted[0][0])
        aggregated = []
        for idx in range(num_params):
            wsum = sum(p[idx] * w for p, w in weighted)
            aggregated.append(wsum / total_weight)
        return aggregated

    def _aggregate_vigilant(
        self,
        active_updates: Dict[str, Tuple[List[np.ndarray], int, Any, float]],
        observations: Dict[str, Dict[str, Any]],
    ) -> Optional[List[np.ndarray]]:
        """Mode 2: Reputation-weighted FedAvg with norm clipping."""
        # Compute median norm for clipping threshold
        norms = [obs['total_norm'] for obs in observations.values()]
        median_norm = float(np.median(norms))
        clip_threshold = median_norm * self.clip_multiplier

        weighted = []
        total_weight = 0.0

        for client_id, (params, n_samples, _, rep_weight) in active_updates.items():
            # Clip update to threshold
            flat = np.concatenate([p.flatten() for p in params])
            update_norm = np.linalg.norm(flat)
            if update_norm > clip_threshold:
                scale = clip_threshold / update_norm
                params = [p * scale for p in params]

            w = rep_weight * n_samples
            weighted.append((params, w))
            total_weight += w

        if total_weight == 0:
            return None

        num_params = len(weighted[0][0])
        aggregated = []
        for idx in range(num_params):
            wsum = sum(p[idx] * w for p, w in weighted)
            aggregated.append(wsum / total_weight)
        return aggregated

    def _aggregate_defensive(
        self,
        active_updates: Dict[str, Tuple[List[np.ndarray], int, Any, float]]
    ) -> Optional[List[np.ndarray]]:
        """Mode 3: Trimmed mean on active (non-rejected) clients."""
        all_params = []
        for client_id, (params, _, _, _) in active_updates.items():
            all_params.append(params)

        if not all_params:
            return None

        n = len(all_params)
        n_trim = int(np.floor(n * self.trim_beta))
        num_params = len(all_params[0])

        aggregated = []
        for idx in range(num_params):
            stacked = np.stack([p[idx] for p in all_params], axis=0)
            sorted_params = np.sort(stacked, axis=0)
            if n_trim > 0 and n - 2 * n_trim > 0:
                trimmed = sorted_params[n_trim:-n_trim]
            else:
                trimmed = sorted_params
            aggregated.append(np.mean(trimmed, axis=0))
        return aggregated

    def _aggregate_lockdown(
        self,
        active_updates: Dict[str, Tuple[List[np.ndarray], int, Any, float]]
    ) -> Optional[List[np.ndarray]]:
        """Mode 4: Multi-Krum on active clients."""
        client_ids = list(active_updates.keys())
        n = len(client_ids)

        if n < 4:
            # Too few clients for Krum, use geometric median
            return self._aggregate_defensive(active_updates)

        # Flatten all updates
        flattened = []
        param_shapes = None
        for cid in client_ids:
            params = active_updates[cid][0]
            if param_shapes is None:
                param_shapes = [p.shape for p in params]
            flattened.append(np.concatenate([p.flatten() for p in params]))

        # Pairwise distances
        n_byz = max(1, int(n * self.krum_byzantine_fraction))
        n_closest = max(1, n - n_byz - 2)

        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(flattened[i] - flattened[j])
                dists[i, j] = d
                dists[j, i] = d

        # Krum scores
        scores = []
        for i in range(n):
            d = dists[i].copy()
            d[i] = np.inf
            closest = np.sort(d)[:n_closest]
            scores.append(np.sum(closest ** 2))

        # Select top n_closest
        selected = np.argsort(scores)[:n_closest]
        selected_flat = [flattened[i] for i in selected]
        avg_flat = np.mean(selected_flat, axis=0)

        # Unflatten
        result = []
        idx = 0
        for shape in param_shapes:
            size = int(np.prod(shape))
            result.append(avg_flat[idx:idx + size].reshape(shape))
            idx += size
        return result

    def _geometric_median_aggregate(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> Optional[List[np.ndarray]]:
        """Emergency fallback: geometric median of all updates."""
        all_flat = []
        param_shapes = None
        for cid, (params, _, _) in client_updates.items():
            if param_shapes is None:
                param_shapes = [p.shape for p in params]
            all_flat.append(np.concatenate([p.flatten() for p in params]))

        if not all_flat:
            return None

        median_flat = self._geometric_median(np.array(all_flat))

        result = []
        idx = 0
        for shape in param_shapes:
            size = int(np.prod(shape))
            result.append(median_flat[idx:idx + size].reshape(shape))
            idx += size
        return result

    # =================================================================
    # MAPE-K Feedback Loop
    # =================================================================

    def _mape_k_adjust(self):
        """
        Monitor-Analyze-Plan-Execute: adjust detector weights and thresholds
        based on recent accuracy trends and detection outcomes.

        Called at the end of each round.
        """
        if not self.enable_mape_k or len(self.round_diagnostics) < self.mape_k_window:
            return

        recent = self.round_diagnostics[-self.mape_k_window:]

        # Monitor: check if accuracy has been declining
        accuracies = [
            r.accuracy_after for r in recent
            if r.accuracy_after is not None
        ]

        if len(accuracies) < 3:
            return

        # Analyze: is accuracy trending down?
        trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
        avg_flagged = np.mean([r.num_flagged / max(r.num_clients, 1) for r in recent])

        # Plan: adjust weights
        if trend < -0.005 and avg_flagged < 0.2:
            # Accuracy declining AND fewer than 20% of clients flagged
            # → detectors are too lenient, missing real attackers
            # Increase direction weight (most discriminative signal)
            self.detector_weights['direction'] = min(
                0.6, self.detector_weights['direction'] + 0.02
            )
            self.anomaly_threshold = max(0.3, self.anomaly_threshold - 0.02)

        elif trend > 0.005 and avg_flagged > 0.6:
            # Accuracy rising AND more than 60% of clients flagged
            # → likely false positives inflating the count; relax slightly.
            # Threshold raised from 0.4 → 0.6: with legitimate 40% attack
            # fractions, avg_flagged sits at ~0.40 while accuracy is healthy
            # (defence working correctly) — we must not confuse that with FPs.
            self.anomaly_threshold = min(0.7, self.anomaly_threshold + 0.01)

        # Renormalize weights
        total = sum(self.detector_weights.values())
        for k in self.detector_weights:
            self.detector_weights[k] /= total

    # =================================================================
    # Main entry point
    # =================================================================

    def aggregate_updates(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int, Dict[str, Any]]],
        accuracy: Optional[float] = None,
    ) -> Tuple[Optional[List[np.ndarray]], List[ExplainableDecision]]:
        """
        Main aggregation method implementing the full OODA + MAPE-K loop.
        """
        # OODA Loop
        observations = self.observe(client_updates)
        analysis = self.orient(observations)
        decisions, explainable = self.decide(analysis)
        aggregated, agg_log = self.act(client_updates, decisions, observations)

        # Update global norm history
        for obs in observations.values():
            self.global_norm_history.append(obs['total_norm'])

        # Update global mean direction
        all_flat = [obs['flattened'] for obs in observations.values()]
        if all_flat:
            self.global_mean_direction = np.mean(all_flat, axis=0)

        # Store diagnostics for MAPE-K
        diag = RoundDiagnostics(
            round_number=self.round_number,
            threat_level=self.threat_level,
            num_clients=len(client_updates),
            num_flagged=sum(
                1 for a in analysis.values()
                if a['client_threat'] in (ThreatLevel.YELLOW, ThreatLevel.ORANGE, ThreatLevel.RED)
            ),
            num_rejected=sum(
                1 for d in decisions.values()
                if d.get('action') == 'reject'
            ),
            accuracy_after=accuracy,
            detector_scores={
                k: [a['detector_scores'].get(k, 0) for a in analysis.values()]
                for k in self.detector_weights
            },
            fusion_weights=dict(self.detector_weights),
        )
        self.round_diagnostics.append(diag)

        # MAPE-K feedback
        self._mape_k_adjust()

        self.increment_round()
        return aggregated, explainable

    def get_defence_description(self) -> str:
        return (
            f"CogDef v2 (OODA+MAPE-K, multi-signal, "
            f"posture={self.threat_level.value}, "
            f"weights={{dir={self.detector_weights['direction']:.2f}, "
            f"norm={self.detector_weights['norm']:.2f}, "
            f"clust={self.detector_weights['cluster']:.2f}, "
            f"temp={self.detector_weights['temporal']:.2f}}})"
        )
