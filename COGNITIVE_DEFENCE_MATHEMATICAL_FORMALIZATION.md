# Mathematical Formalization and Optimization of Cognitive Defense (POSG-SAC)

**Date**: March 12, 2026  
**Status**: Critical Analysis & Optimization Roadmap

---

## Executive Summary

The CogDef-POSG framework shows **proof of concept** (95-97% accuracy achieved in Rounds 3-5 of adaptive attacks), but suffers from **catastrophic forgetting** and **unstable learning**. This document provides:

1. **Formal mathematical framework** of the current approach
2. **Root cause analysis** of performance collapse
3. **Theoretical soundness evaluation**
4. **Concrete optimization strategies**

**Key Finding**: The defense is theoretically sound but suffers from implementation issues in reward design, state representation, and training stability.

---

## 1. Theoretical Framework

### 1.1 Problem Formulation: Federated Learning as a POSG

We model Byzantine-robust FL as a **Partially Observable Stochastic Game (POSG)** between:

- **Defender (Server)**: Aggregates client updates while maximizing global model accuracy
- **Adversary (Byzantine clients)**: Injects poisoned updates to degrade model performance

#### State Space

**True State** $s_t \in \mathcal{S}$ (unknown to defender):
$$
s_t = \{(\theta_t^{(i)}, \beta_i, \tau_i)\}_{i=1}^{N}
$$

where:
- $\theta_t^{(i)}$: Client $i$'s model update at round $t$
- $\beta_i \in \{0, 1\}$: Byzantine indicator (1 = malicious, 0 = benign)
- $\tau_i$: Attack strategy/intensity of client $i$

**Observation** $o_t^{(i)} \in \mathbb{R}^6$ (what defender sees):
$$
o_t^{(i)} = \begin{bmatrix}
\|\Delta\theta_t^{(i)}\|_2 \\
\frac{1}{L}\sum_{\ell=1}^{L} \|\Delta\theta_t^{(i,\ell)}\|_2 \\
\max_{\ell} \|\Delta\theta_t^{(i,\ell)}\|_2 \\
\cos(\Delta\theta_t^{(i)}, \theta_{t-1}^{(g)}) \\
\text{Tr}(F_i) \approx \|\Delta\theta_t^{(i)}\|_2^2 \\
\frac{n_i}{\max_j n_j}
\end{bmatrix}
$$

**Features**:
1. Total L2 norm of update
2. Average per-layer norm
3. Maximum per-layer norm
4. Cosine similarity to previous global model
5. Fisher Information trace (squared gradient magnitude)
6. Normalized sample count

---

### 1.2 Belief State Tracking (GRU-based Orient)

Since $s_t$ is partially observable, we maintain a **belief state** $b_t^{(i)} \in \mathbb{R}^{d_h}$ for each client using a GRU:

$$
b_t^{(i)} = \text{GRU}(\phi(o_t^{(i)}), b_{t-1}^{(i)})
$$

where:
- $\phi: \mathbb{R}^6 \to \mathbb{R}^{d_h}$ is a learned projection (linear → LayerNorm → ReLU)
- $d_h = 64$ (hidden dimension)
- $b_0^{(i)} = \mathbf{0}$ (zero initialization)

**Key Property**: GRU captures **temporal dynamics** of client behavior, solving the "boiling frog" problem where attacks slowly drift to avoid detection.

---

### 1.3 Compact State Representation

**Critical Optimization** (implemented in latest version):

Instead of concatenating all belief states $[b_t^{(1)}, \ldots, b_t^{(N)}] \in \mathbb{R}^{N \cdot d_h}$ (which creates a 6400-dim sparse vector for N=100, d_h=64), we use **sufficient statistics**:

$$
\tilde{s}_t = \begin{bmatrix}
\mathbb{E}_{i \in \mathcal{A}_t}[b_t^{(i)}] \\
\text{std}_{i \in \mathcal{A}_t}(b_t^{(i)})
\end{bmatrix} \in \mathbb{R}^{2d_h}
$$

where $\mathcal{A}_t$ is the set of active clients in round $t$.

**Rationale**:
- Dimension: $2 \times 64 = 128$ (dense, regardless of participation rate)
- Captures first two moments of belief distribution
- Invariant to client permutation (desirable inductive bias)
- Dramatically improves gradient signal-to-noise ratio

---

### 1.4 Policy: Soft Actor-Critic (SAC)

The defender learns a stochastic policy $\pi_\phi: \mathcal{S} \to \mathcal{A}$ that outputs **continuous aggregation weights**:

$$
\mathbf{a}_t = [w_1, \ldots, w_N] \in [0, 1]^N
$$

where $w_i$ controls the influence of client $i$ in aggregation.

#### Policy Parameterization: Beta Distribution

$$
w_i \sim \text{Beta}(\alpha_i(\tilde{s}_t), \beta_i(\tilde{s}_t))
$$

where $\alpha_i, \beta_i > 1$ are learned via neural networks with softplus activation.

**Benefits over Gaussian/Tanh squashing**:
- Natural support on $(0, 1)$ (no boundary artifacts)
- Smooth density (better for SAC's entropy regularization)
- Mode $= \frac{\alpha-1}{\alpha+\beta-2}$ allows deterministic evaluation

---

### 1.5 Aggregation with Weights

Given weights $\mathbf{w} = [w_1, \ldots, w_N]$ and updates $\{\Delta\theta_t^{(i)}\}$:

1. **Weight by contribution**:
$$
\tilde{w}_i = w_i \cdot n_i
$$

2. **Median-norm clipping** (defense against magnitude attacks):
$$
\Delta\theta_t^{(i)} \gets \Delta\theta_t^{(i)} \cdot \min\left(1, \frac{\text{median}_j \|\Delta\theta_t^{(j)}\|}{\|\Delta\theta_t^{(i)}\|}\right)
$$

3. **Weighted aggregation**:
$$
\theta_{t}^{(g)} \gets \theta_{t-1}^{(g)} + \frac{\sum_{i=1}^{N} \tilde{w}_i \Delta\theta_t^{(i)}}{\sum_{i=1}^{N} \tilde{w}_i}
$$

---

### 1.6 Reward Function

$$
R_t = \alpha \cdot \Delta \text{Acc}_{\text{val}} - \beta \cdot H(b_t) - \gamma \cdot \Omega_t
$$

**Terms**:
1. **$\Delta \text{Acc}_{\text{val}}$**: Change in validation accuracy (primary objective)
2. **$H(b_t) = \frac{1}{|\mathcal{A}_t|} \sum_{i \in \mathcal{A}_t} H(|b_t^{(i)}|)$**: Belief entropy (penalizes uncertainty)
3. **$\Omega_t = \|\theta_t^{(g)} - \theta_{t-1}^{(g)}\|_2$**: Model divergence (prevents drastic changes)

**Current coefficients**: $\alpha=1.0$, $\beta=0.3$, $\gamma=0.2$

---

## 2. Current Implementation Issues (Root Cause Analysis)

### 2.1 Reward Instability

**Problem**: $\Delta \text{Acc}_{\text{val}}$ is:
- **Noisy**: Validation accuracy fluctuates ±2-5% even with identical model
- **Delayed**: Multi-round effect not captured in single-step reward
- **Sparse**: Early rounds see ~0.01-0.02 changes, drowning in noise

**Evidence from logs**:
```
Round 0: Acc = 0.1145
Round 1: Acc = 0.1135  →  Δ = -0.001  (negative reward despite correct isolation!)
Round 2: Acc = ???     →  SAC learns to avoid isolation
```

**Consequence**: SAC learns to "play it safe" and assign weight ≈1.0 to everyone to avoid negative rewards.

---

### 2.2 Warm-up Heuristic Fragility

**Current heuristic** (FLTrust-inspired pairwise cosine):

1. L2-normalize all updates to unit vectors
2. Compute pairwise cosine similarity matrix $G_{ij}$
3. Score client $i$ as mean of top-$k$ cosines (leave-one-out)
4. Weight $w_i = \max(0, \text{score}_i) / \max_j(\text{score}_j)$

**Problems**:
- **Coordinate attacks**: With 40% Byzantine (40/100), the median/majority is corrupted
- **Adaptive stealth**: DynOpt with intensity=0.05 produces updates with $\cos \approx 0.95$ to global model (high similarity)
- **Binary threshold**: Weight < 0.2 → isolate. Binary decision loses nuance.

**Evidence**: Static label-flip logs show ~93% clients get weight=1.0, only ~7% isolated (should isolate 40%).

---

### 2.3 SAC Training Instability

**Issue 1: Cold start**
- Buffer is empty for first 5 rounds (warmup)
- When SAC takes over at round 6, it has ~500 transitions from heuristic policy
- These transitions are **off-policy** and encode the heuristic's mistakes

**Issue 2: Catastrophic collapse** (adaptive_dny_opt_cognitive_defence.log):
```
Round 3: 95.28%  ←  Defense working!
Round 4: 97.20%  ←  Peak performance
Round 5: 95.89%
Round 6: 10.35%  ←  COLLAPSE (SAC takes over)
```

**Hypothesis**: SAC's entropy regularization + noisy reward → exploration leads to accepting Byzantine clients → immediate accuracy drop → negative reward → agent learns to "do nothing" (w≈1.0 for all).

**Issue 3: Replay buffer contamination**
- Buffer contains both warmup (heuristic-guided) and post-warmup (SAC) transitions
- High-reward warmup transitions are never reproduced by SAC (distribution shift)
- Agent chases phantom strategies that don't generalize

---

### 2.4 Observation Normalization Issues

**Current**: Welford (running mean/std) normalizer

**Problem**:
- First few rounds: normalization statistics are unstable (n < 10)
- Attack-heavy rounds shift the distribution (outliers become "normal")
- GRU sees non-stationary input distribution → unstable hidden states

---

### 2.5 Belief Entropy Calculation

**Current**: 
$$
H(b_t^{(i)}) = -\sum_j p_j \log p_j, \quad p_j = \frac{|b_t^{(i)}_j|}{\sum_k |b_t^{(i)}_k|}
$$

**Problem**:
- Absolute value + L1 normalization is an arbitrary "pseudo-probability"
- No theoretical grounding (belief states are not probabilities)
- High entropy could mean "client is uncertain" OR "belief state is rich/informative"

---

## 3. Theoretical Soundness Assessment

### 3.1 Strengths ✓

1. **POSG Formulation**: Sound game-theoretic foundation
   - Captures partial observability inherent to Byzantine FL
   - Belief states are appropriate for decision-making under uncertainty

2. **Temporal Modeling**: GRU is a principled choice
   - Proven effective for sequence modeling
   - Shared weights across clients = efficient + generalizable
   - Solves the "slow drift" attack problem

3. **Continuous Actions**: Better than binary accept/reject
   - Allows "soft" decisions (e.g., 50% weight = partial trust)
   - Differentiable policy enables gradient-based RL

4. **Feature Engineering**: Observation vector is comprehensive
   - Norms capture magnitude-based attacks
   - Cosine similarity captures direction-based attacks
   - Fisher trace captures gradient poisoning

### 3.2 Weaknesses ✗

1. **Reward Design**: Theoretically flawed
   - Single-round $\Delta\text{Acc}$ violates Markov assumption (true reward is multi-round cumulative)
   - Entropy penalty lacks theoretical justification
   - Magnitude coefficients ($\alpha, \beta, \gamma$) are arbitrary

2. **Sample Efficiency**: RL is data-hungry
   - One transition per round → after 30 rounds, buffer has ~30 samples
   - SAC needs ~10k-100k transitions to converge (offline RL literature)
   - Current setup is **severely under-sampled**

3. **Non-Stationarity**: Adversary adapts
   - SAC assumes stationary MDP
   - Byzantine clients can observe defense behavior and adapt (adversarial RL)
   - No opponent modeling or game-theoretic equilibrium analysis

4. **Exploration-Exploitation**: Mismatch
   - SAC explores via entropy maximization
   - In adversarial setting, exploration = accepting Byzantine clients → catastrophic forgetting
   - No "safe exploration" mechanism

---

## 4. Optimization Strategies (Prioritized)

### Priority 1: Stabilize Reward Signal (CRITICAL)

#### Option A: Multi-Round Discounted Reward (Recommended)

Instead of immediate $\Delta\text{Acc}$, use **n-step TD target**:

$$
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \Delta\text{Acc}_{t+k}
$$

**Implementation**:
- Buffer stores $(s_t, a_t, [r_t, r_{t+1}, \ldots, r_{t+n}], s_{t+n})$
- Only update SAC every $n$ rounds when full trajectory is available
- Use $n=3$ (aligns with evidence: rounds 3-5 showed cumulative success)

#### Option B: Validation-Ensemble Smoothing

Replace single $\text{Acc}_{\text{val}}$ with **moving average**:

$$
\bar{A}_t = 0.7 \cdot \bar{A}_{t-1} + 0.3 \cdot \text{Acc}_t
$$

Reward: $R_t = \alpha \cdot (\bar{A}_t - \bar{A}_{t-1})$

**Pro**: Stable, easy to implement  
**Con**: Delayed feedback (slower learning)

#### Option C: Auxiliary Reward Shaping

Add **Byzantine detection accuracy** as auxiliary reward:

$$
R_t^{\text{aux}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{decision}_i = \beta_i^{\text{true}}]
$$

**Requires**: Ground-truth labels (simulation only)  
**Benefit**: Direct supervision on *what* the agent should learn

---

### Priority 2: Improve Warm-up Heuristic

#### Option A: Multi-Krum Defense (Proven Byzantine-resilient)

Instead of pairwise cosine, use **Multi-Krum scoring**:

1. For each client $i$, compute sum of distances to $m$ nearest neighbors:
$$
S_i = \sum_{j \in \mathcal{N}_m(i)} \|\Delta\theta_t^{(i)} - \Delta\theta_t^{(j)}\|_2^2
$$

2. Select $n-f-2$ clients with smallest scores (where $f$ = expected Byzantine count)
3. Assign weight 1.0 to selected, 0.1 to others

**Theoretical guarantee**: Robust to $f < n/2$ Byzantine clients (Blanchard et al., 2017)

**Implementation**: Replace `_heuristic_weights()` function

#### Option B: RFA (Robust Federated Aggregation)

Use **geometric median** instead of weighted average:

$$
\theta_t^{(g)} = \arg\min_{\theta} \sum_{i=1}^{N} w_i \|\theta - \theta_{t-1}^{(g)} - \Delta\theta_t^{(i)}\|_2
$$

**Robust property**: Breakdown point = 50% (best possible)  
**Downside**: Computationally expensive (iterative Weiszfeld algorithm)

---

### Priority 3: SAC Hyperparameter Tuning

#### Entropy Temperature

**Current**: Auto-tuned with target entropy = $-\dim(\mathcal{A})$ (default SAC)

**Problem**: High-dim action space ($N=100$) → high target entropy → excessive exploration

**Fix**: Reduce target entropy to focus exploitation:

$$
H_{\text{target}} = -0.1 \cdot \dim(\mathcal{A}) = -10
$$

**Code change** (in `sac_agent.py`):
```python
self.target_entropy = -0.1 * action_dim  # Instead of -action_dim
```

#### Discount Factor

**Current**: $\gamma = 0.99$ (long-horizon planning)

**Problem**: FL rounds are not infinitely discounted; round $t+30$ is as important as $t+1$

**Fix**: Use $\gamma = 0.95$ for medium-horizon reward accumulation

#### Learning Rate

**Current**: $\alpha_{\text{actor}} = \alpha_{\text{critic}} = 3 \times 10^{-4}$

**Problem**: Standard for 1M-timestep environments; we have ~30 rounds

**Fix**: **Increase** learning rate to accelerate convergence:
```python
lr_actor=1e-3, lr_critic=1e-3
```

#### Batch Size vs Buffer Size

**Current**: Buffer = 50k, Batch = 64

**Problem**: After 30 rounds, buffer has ~30 samples → batch size > buffer size → error

**Fix**: 
- Reduce buffer size to 1000 (still 30x larger than available data)
- Reduce batch size to 16 (allow updates when buffer has 16+ samples)

---

### Priority 4: Stabilize GRU Belief Tracker

#### Gradient Clipping

**Issue**: GRU gradients can explode with noisy observations

**Fix**: Add gradient clipping to GRU module:
```python
torch.nn.utils.clip_grad_norm_(self.tracker.parameters(), max_norm=1.0)
```

#### Observation Normalization Strategy

**Current**: Welford (online mean/std)

**Proposal**: **Robust Z-score** using median absolute deviation (MAD):

$$
o_{\text{norm}} = \frac{o - \text{median}(o)}{\text{MAD}(o) + \epsilon}
$$

**Benefit**: Resistant to outliers (Byzantine clients can't shift normalization stats)

#### Belief Entropy Replacement

**Current**: Arbitrary pseudo-probability entropy

**Proposal**: Use **L2 norm** as uncertainty proxy:

$$
U_t = \frac{1}{|\mathcal{A}_t|} \sum_{i \in \mathcal{A}_t} \|b_t^{(i)}\|_2
$$

**Rationale**: 
- Low norm → GRU is "uncertain" (weak hidden state)
- High norm → GRU has "strong opinion" (confident belief)

---

### Priority 5: Curriculum Learning for SAC

#### Idea: Gradually increase attack difficulty

**Phase 1 (Rounds 1-10)**: 
- Train with 20% Byzantine (easy)
- Build buffer with successful isolation strategies

**Phase 2 (Rounds 11-20)**:
- Increase to 30% Byzantine

**Phase 3 (Rounds 21-30)**:
- Full 40% Byzantine (target difficulty)

**Benefit**: SAC learns "how to defend" before facing full adversarial strength

---

## 5. Implementation Roadmap

### Step 1: Immediate Fixes (2-4 hours)

1. **Stabilize reward**:
   - Switch to moving-average validation accuracy
   - Increase $\alpha$ to 10.0 (make accuracy dominant signal)
   - Remove or reduce $\beta$ (entropy penalty) to 0.05

2. **Fix SAC hyperparameters**:
   - Reduce target entropy: `target_entropy = -0.1 * action_dim`
   - Increase learning rates: `lr=1e-3`
   - Adjust buffer/batch: `buffer_capacity=1000, batch_size=16`
   - Reduce $\gamma$ to 0.95

3. **Add gradient clipping**:
   - Clip GRU gradients to max_norm=1.0

### Step 2: Heuristic Replacement (4-6 hours)

- Implement Multi-Krum warm-up heuristic
- Extend warm-up period to 10 rounds (from 5)

### Step 3: Observation Normalization (2 hours)

- Replace Welford with MAD-based robust normalization
- Add outlier detection: cap normalized values at ±5

### Step 4: Reward Redesign (4-6 hours)

- Implement n-step TD reward (n=3)
- Add auxiliary Byzantine detection reward (simulation only)

### Step 5: Advanced Optimizations (8-12 hours)

- Implement curriculum learning pipeline
- Add opponent modeling (estimate attacker strategy from observations)
- Explore meta-learning for fast adaptation to new attack types

---

## 6. Expected Outcomes

### Baseline (No optimization):
- **Static attacks**: 11-15% accuracy (catastrophic failure)
- **Adaptive attacks**: Brief success (95%+) followed by collapse

### After Step 1-2 (Immediate fixes):
- **Static attacks**: 75-85% accuracy (functional but suboptimal)
- **Adaptive attacks**: 60-70% accuracy (reduced collapse)

### After Step 3-4 (Reward + normalization):
- **Static attacks**: 85-92% accuracy (near-optimal)
- **Adaptive attacks**: 75-85% accuracy (stable defense)

### After Step 5 (Advanced):
- **Static attacks**: 92-95% accuracy (≈ no-attack baseline)
- **Adaptive attacks**: 85-90% accuracy (robust to evolving adversaries)

---

## 7. Theoretical Guarantees

### What CAN we prove?

1. **Convergence**: If reward is Lipschitz-continuous in state-action space and policy is smooth (Beta distribution is), SAC converges to a *local* optimum (Haarnoja et al., 2018)

2. **Byzantine tolerance**: Multi-Krum+weighted aggregation is robust to $f < n/2$ Byzantine clients *under IID data* (Blanchard et al., 2017)

3. **Temporal detection**: GRU with sufficient hidden dimension can approximate any sequence-to-sequence mapping (universal approximation for RNNs, Hammerstrom 1993)

### What CANNOT we prove?

1. **Global optimality**: SAC guarantees local convergence; no global optimum guarantee in non-convex neural policy space

2. **Adversarial robustness**: No formal guarantee against *adaptive* adversaries (game-theoretic equilibrium analysis needed)

3. **Sample complexity**: No finite-sample bound (RL theory is asymptotic; actual sample efficiency depends on problem structure)

---

## 8. Alternative Approaches (If RL continues to fail)

### Plan B: Supervised Learning Replace SAC with **supervised classifier**:

- **Features**: Same observation + belief state
- **Labels**: Ground-truth Byzantine indicators (simulation) or pseudo-labels from Multi-Krum
- **Model**: Gradient Boosted Trees (XGBoost) or simple MLP
- **Advantage**: 100x better sample efficiency than RL

### Plan C: Ensemble Defense

Combine multiple defenses via **majority voting**:
- Multi-Krum (Byzantine-robust)
- FLTrust (cosine-based)
- Trimmed Mean (magnitude-robust)

Weight: $w_i = \frac{1}{3}\sum_{d=1}^{3} \mathbb{1}[\text{defense}_d \text{ accepts } i]$

### Plan D: Game-Theoretic Stackelberg Defense

- Model as **Stackelberg game**: Defender moves first (announces defense), attacker responds
- Solve for defender's optimal commitment strategy via **linear programming** (Korzhyk et al., 2011)
- **Advantage**: Computationally tractable, provable guarantee

---

## 9. Conclusion

The CogDef-POSG framework is **theoretically sound** and shows **proof-of-concept empirical success**, but suffers from:

1. **Reward instability** (noisy, delayed, sparse signal)
2. **Sample inefficiency** (RL needs 1000x more data than available)
3. **Training instability** (catastrophic forgetting, exploration in adversarial setting)

**Recommended Path**:
1. Implement **Priority 1-2 optimizations** (stabilize reward, improve heuristic)
2. Run ablation studies to isolate failure modes
3. If SAC still fails, pivot to **Plan B (supervised learning)** or **Plan C (ensemble)**

The framework has strong foundations; with proper reward engineering and training stabilization, it can achieve robust performance against both static and adaptive attacks.

---

## References

- Blanchard et al. (2017): *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent*
- Haarnoja et al. (2018): *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*
- Wu et al. (2022): *FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping*, NDSS
- Korzhyk et al. (2011): *Stackelberg vs. Nash in Security Games*
- Xie et al. (2021): *Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation*

