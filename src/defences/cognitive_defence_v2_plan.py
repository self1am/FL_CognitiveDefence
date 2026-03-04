# =============================================================================
# COGNITIVE DEFENCE v2: STATE-OF-THE-ART ENHANCEMENT PLAN
# =============================================================================
#
# This file defines the enhanced cognitive defence architecture.
# It serves as both documentation AND the implementation skeleton.
#
# THESIS TITLE (proposed):
#   "CogDef: A Cognitive-Inspired Multi-Signal Defence Framework
#    for Byzantine-Robust Federated Learning"
#
# NOVEL CONTRIBUTION (what makes this publishable):
#   Existing defences use ONE signal (norms, distances, or statistics).
#   CogDef uses an OODA cognitive loop that FUSES multiple detection
#   signals, maintains temporal reputation memory, and ADAPTS its
#   defence posture in real-time — switching between soft reweighting,
#   hard rejection, and robust aggregation based on threat level.
#
# =============================================================================

"""
ARCHITECTURE OVERVIEW
=====================

Current v1 (broken):
  Observe → L2 norms only
  Orient  → z-score vs global history (can't detect label flips)
  Decide  → blanket reputation decay (penalises everyone equally)
  Act     → weighted FedAvg (still includes malicious updates)

Enhanced v2 (this plan):
  Observe → Multi-signal extraction (norms, directions, layers, cross-client)
  Orient  → Multi-detector fusion with per-client profiling
  Decide  → Adaptive threat assessment with escalating response
  Act     → Threat-proportional aggregation (reweight → reject → robust)

KEY INSIGHT: Label-flip attacks produce normal-magnitude gradients pointing
in the WRONG DIRECTION. The current norm-only detection is fundamentally
blind to this. We need directional analysis.
"""


# =============================================================================
# PHASE 1: OBSERVE — Multi-Signal Feature Extraction
# =============================================================================
#
# Current: Only extracts L2 norms per parameter layer
# Problem: Norms can't detect direction-based attacks (label flip, sign flip)
#
# Enhanced signals to extract per client update:
#
# Signal 1: L2 Norm Profile (existing)
#   - Per-layer norms, total norm, avg norm
#   - Useful for: gradient scaling attacks, noise injection
#
# Signal 2: Cosine Similarity to Global Model Direction [NEW]
#   - Compare each client's update direction to the running mean direction
#   - cos_sim = dot(client_update, mean_update) / (||client|| * ||mean||)
#   - Useful for: label flip, sign flip, targeted attacks
#   - THIS IS THE #1 MISSING SIGNAL
#
# Signal 3: Per-Layer Anomaly Decomposition [NEW]
#   - Compute z-scores per layer, not just globally
#   - Attack may affect output layer heavily but leave early layers normal
#   - Useful for: targeted backdoor attacks, partial poisoning
#
# Signal 4: Cross-Client Clustering [NEW]
#   - Cluster all client updates using cosine distance
#   - Honest clients form a tight cluster; attackers form outlier cluster(s)
#   - Use robust clustering (HDBSCAN or spectral) — NOT k-means
#   - Useful for: coordinated attacks, sybil detection
#
# Signal 5: Temporal Consistency [NEW]
#   - Track per-client update direction over rounds
#   - Honest clients converge; attackers flip direction round-to-round
#   - Compute autocorrelation of each client's cosine similarity over time
#   - Useful for: adaptive attacks that alternate strategies
#
# Signal 6: Update Magnitude Relative to Loss [NEW]
#   - If a client's update is large but loss isn't decreasing, suspicious
#   - Ratio: ||update|| / reported_loss_improvement
#   - Useful for: gradient amplification attacks


# =============================================================================
# PHASE 2: ORIENT — Multi-Detector Fusion with Per-Client Profiling
# =============================================================================
#
# Current: Single z-score > 2.0 threshold on total norm
# Problem: One-dimensional detection, no per-client memory, no fusion
#
# Enhanced detection:
#
# Detector 1: Norm Anomaly Score (improved from current)
#   - Per-layer z-scores against per-layer historical distribution
#   - Weighted sum of per-layer anomaly scores
#   - Score: 0.0 (normal) to 1.0 (anomalous)
#
# Detector 2: Direction Anomaly Score [NEW — CRITICAL]
#   - Cosine similarity to geometric median of honest updates
#   - Use iterative Weiszfeld algorithm for geometric median
#   - Score: 1.0 - cos_sim (high = pointing away from consensus)
#   - This ALONE would fix the label-flip detection failure
#
# Detector 3: Cluster Outlier Score [NEW]
#   - Run DBSCAN/HDBSCAN on flattened + PCA-reduced updates
#   - Outlier clients get score 1.0, core cluster members get 0.0
#   - Transition zone for borderline clients
#
# Detector 4: Temporal Inconsistency Score [NEW]
#   - Compare current update direction to client's own recent history
#   - High variance in direction across rounds → suspicious
#   - Score: rolling std of per-round cosine similarities
#
# FUSION: Weighted combination of all detector scores
#   anomaly_score_i = w1*norm_score + w2*direction_score +
#                     w3*cluster_score + w4*temporal_score
#
#   Default weights: w1=0.15, w2=0.40, w3=0.25, w4=0.20
#   (Direction gets highest weight — it catches what v1 misses)
#
#   The MAPE-K Monitor component tracks which detectors are most
#   effective at catching confirmed attackers and adjusts weights
#   over time (meta-learning loop).


# =============================================================================
# PHASE 3: DECIDE — Adaptive Threat Assessment
# =============================================================================
#
# Current: Binary decision (anomalous → decay reputation by 0.8, else +0.05)
# Problem: Treats all anomalies identically, no escalation, floor too high (0.1)
#
# Enhanced decision framework:
#
# 3a. Per-Client Reputation Model (improved)
#   -  Reputation r_i ∈ [0, 1], starts at 0.5 (not 1.0 — earn trust)
#   - Good behavior: r_i += alpha * (1 - r_i)  [diminishing returns]
#   - Bad behavior:  r_i *= (1 - anomaly_score * severity)
#   - Configurable: recovery_rate, penalty_severity, initial_reputation
#
# 3b. Threat Level Classification [NEW]
#   - GREEN  (anomaly_score < 0.3): Normal operation
#   - YELLOW (0.3 ≤ score < 0.6):  Soft reweighting
#   - ORANGE (0.6 ≤ score < 0.8):  Heavy down-weighting
#   - RED    (score ≥ 0.8):        Hard rejection (exclude from aggregation)
#
# 3c. Global Threat Posture [NEW — MAPE-K Analyze+Plan]
#   - Track what fraction of clients are YELLOW+ in recent rounds
#   - If > 30% clients flagged → system is under attack
#   - Escalate global posture: tighten thresholds, lower trust floor
#   - If attack subsides → gradually relax back to normal
#   - This is the MAPE-K feedback loop that makes it truly adaptive


# =============================================================================
# PHASE 4: ACT — Threat-Proportional Aggregation
# =============================================================================
#
# Current: Weighted FedAvg with reputation-scaled weights (min 0.1)
# Problem: Even with weight 0.1, 40 malicious clients contribute 4.0 total
#          weight vs 60 honest clients at ~1.0 each = 60.0. That's still
#          6.25% malicious influence — enough to degrade accuracy significantly.
#
# Enhanced aggregation modes (selected based on global threat posture):
#
# Mode 1: PEACE (Green posture)
#   - Reputation-weighted FedAvg
#   - All clients included, weights proportional to reputation
#   - Most efficient, no computational overhead
#
# Mode 2: VIGILANT (Yellow posture)  [NEW]
#   - Reputation-weighted FedAvg with clipping
#   - Clip each client's update to median norm * multiplier
#   - Removes gradient scaling attacks while preserving direction
#
# Mode 3: DEFENSIVE (Orange posture)  [NEW]
#   - Exclude RED clients entirely
#   - Apply trimmed-mean on remaining clients (trim YELLOW+ clients)
#   - Combine with reputation weighting
#
# Mode 4: LOCKDOWN (Red posture)  [NEW]
#   - Run Multi-Krum on GREEN clients only
#   - If too few GREEN clients, use geometric median of all updates
#   - Maximum robustness, some convergence cost


# =============================================================================
# PHASE 5: MAPE-K FEEDBACK LOOP — Self-Improving Detection
# =============================================================================
#
# The MAPE-K (Monitor-Analyze-Plan-Execute) framework wraps the OODA loop:
#
# Monitor:
#   - Track accuracy trajectory round-over-round
#   - Track detection rates per detector (which detector flagged which client)
#   - Track false-positive signal (clients flagged but accuracy still rose)
#
# Analyze:
#   - If accuracy dropped AND many clients were flagged → attack ongoing
#   - If accuracy dropped AND few clients flagged → detectors insufficient
#   - If accuracy stable AND many clients flagged → possible false positives
#
# Plan:
#   - Adjust detector fusion weights based on which detectors correlate
#     with actual accuracy degradation
#   - Adjust anomaly thresholds: tighten if missing attacks, loosen if
#     too many false positives
#   - Adjust threat posture escalation/de-escalation speed
#
# Execute:
#   - Apply planned adjustments for the next round
#   - Log all adjustments for explainability


# =============================================================================
# IMPLEMENTATION PRIORITY ORDER
# =============================================================================
#
# Sprint 1 (Week 1-2): Direction Detection — BIGGEST IMPACT
#   □ Add cosine similarity to geometric median in Observe
#   □ Add Direction Anomaly Score in Orient
#   □ Fuse direction score with existing norm score (even 50/50 is better)
#   □ Re-run baseline 01 (label_flip) — expect massive improvement
#
# Sprint 2 (Week 2-3): Reputation + Threat Levels
#   □ Fix reputation model (start at 0.5, better decay curve)
#   □ Implement 4-level threat classification (GREEN/YELLOW/ORANGE/RED)
#   □ Implement hard rejection for RED clients
#   □ Re-run baselines 01-05 — expect improvement across all attacks
#
# Sprint 3 (Week 3-4): Adaptive Aggregation Modes
#   □ Implement global threat posture tracking
#   □ Implement Mode 2 (clipping) and Mode 3 (exclude + trimmed mean)
#   □ Implement Mode 4 (lockdown with Krum fallback)
#   □ Re-run baselines — expect competitive with Krum on static attacks
#
# Sprint 4 (Week 4-5): Clustering + Temporal Analysis
#   □ Add cross-client clustering (HDBSCAN)
#   □ Add temporal consistency scoring
#   □ Full multi-signal fusion with learned weights
#   □ Re-run baselines — expect SOTA on adaptive attacks
#
# Sprint 5 (Week 5-6): MAPE-K Self-Tuning
#   □ Implement accuracy-feedback weight adjustment
#   □ Implement threshold auto-tuning
#   □ Implement posture escalation/de-escalation
#   □ Run extended experiments (50+ rounds) to show adaptation
#
# Sprint 6 (Week 6-8): Publication-Ready Evaluation
#   □ CIFAR-10 experiments (non-IID distribution)
#   □ Multiple seeds (123, 456, 789) for statistical significance
#   □ Ablation study: each signal's contribution
#   □ Convergence analysis plots
#   □ Comparison tables vs SOTA (FLTrust, FLAME, RFA)


# =============================================================================
# EVALUATION FRAMEWORK FOR THESIS
# =============================================================================
#
# METRICS TO REPORT (for each experiment):
#   1. Final accuracy after R rounds
#   2. Peak accuracy achieved
#   3. Attack Success Rate (ASR): how much accuracy degraded vs clean baseline
#   4. Defence Effectiveness: 1 - (accuracy_drop / max_possible_drop)
#   5. Convergence speed: rounds to reach 90% of clean baseline accuracy
#   6. Detection precision: correctly flagged / total flagged
#   7. Detection recall: correctly flagged / total malicious
#   8. F1 score of detection
#   9. Computational overhead: time per round vs FedAvg
#
# EXPERIMENTS MATRIX (minimum for publication):
#   Dataset:  MNIST (IID) + CIFAR-10 (IID + non-IID)
#   Attacks:  label_flip, dny_opt, stat_opt, min_max, min_sum
#   Defences: No Defence, CogDef-v2, Krum, Trimmed Mean, VERT
#             + FLTrust, FLAME (if time permits)
#   Scales:   100 clients (40% malicious), 100 clients (20% malicious)
#   Rounds:   50 minimum
#   Seeds:    3 seeds minimum, report mean ± std
#
# ABLATION STUDY:
#   CogDef-v2 full → remove direction → remove clustering →
#   remove temporal → remove MAPE-K → pure norm-only (= v1)
#   Shows each component's marginal contribution
#
# PLOTS FOR THESIS:
#   1. Accuracy vs Round curves (all defences, per attack) — 5 plots
#   2. Detection precision/recall over rounds — shows adaptation
#   3. Threat posture transitions over rounds — shows cognitive awareness
#   4. Reputation distribution heatmap (clients × rounds)
#   5. Ablation bar chart (each signal's contribution to final accuracy)
#   6. Radar chart: accuracy, robustness, speed, overhead (per defence)


# =============================================================================
# WHY THIS IS THESIS-WORTHY
# =============================================================================
#
# 1. NOVEL FRAMEWORK: No existing defence uses a cognitive OODA + MAPE-K
#    architecture. This is a genuinely new framing bridging cognitive science
#    and Byzantine fault tolerance.
#
# 2. MULTI-SIGNAL FUSION: Most defences are single-signal (Krum = distances,
#    Trimmed Mean = magnitudes, VERT = historical prediction). CogDef fuses
#    multiple orthogonal signals — this is the key to handling diverse attacks.
#
# 3. ADAPTIVE RESPONSE: Most defences use a fixed strategy regardless of
#    threat level. CogDef escalates/de-escalates its response posture,
#    which means it's efficient when there's no attack (Mode 1 = FedAvg)
#    and maximally robust under attack (Mode 4 = Krum-like).
#
# 4. EXPLAINABILITY: Every decision is logged with evidence and reasoning.
#    This is a differentiator vs black-box defences — important for
#    trustworthy AI / responsible ML narrative.
#
# 5. SELF-IMPROVING: The MAPE-K loop means the defence gets better over
#    time by learning which signals matter most against the current attack.
#    This directly counters adaptive attacks that evolve their strategy.
#
# POSITIONING vs SOTA:
#   - vs Krum:         CogDef is more efficient (doesn't always reject)
#   - vs Trimmed Mean: CogDef detects directional attacks (label flip)
#   - vs VERT:         CogDef doesn't need prediction warm-up rounds
#   - vs FLTrust:      CogDef doesn't need a trusted clean dataset
#   - vs FLAME:        CogDef has temporal adaptation (FLAME is stateless)
"""
