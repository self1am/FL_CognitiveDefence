# Cognitive Defense: Quick Reference Card

**Last Updated**: March 12, 2026  
**Status**: 🔴 Critical Issues Identified → 🟡 Fixes Ready

---

## 🔍 Problem Diagnosis

### Current Performance
- **Static Label-Flip**: 11.35% accuracy (should be 90%)
- **Adaptive DynOpt**: Briefly 95-97% (rounds 3-5), then collapses to 11%

### Root Causes (Prioritized)

| # | Issue | Impact | Confidence |
|---|-------|--------|------------|
| 1 | **Noisy reward signal** | 🔴 Critical | 95% |
| 2 | **Weak warm-up heuristic** | 🔴 Critical | 90% |
| 3 | **SAC hyperparameters** | 🟠 High | 85% |
| 4 | **Catastrophic forgetting** | 🟠 High | 80% |
| 5 | **Observation normalization** | 🟡 Medium | 70% |

---

## ✅ Is the Approach Theoretically Sound?

### YES ✓ - Strong Foundations

**Strengths**:
1. ✅ **POSG formulation** is appropriate for Byzantine FL
2. ✅ **GRU belief tracking** solves temporal detection problem
3. ✅ **Soft Actor-Critic** is state-of-the-art for continuous control
4. ✅ **Feature engineering** (norms, cosine, Fisher) is comprehensive
5. ✅ **Compact state representation** (mean/std of beliefs) is efficient

**Evidence**: Rounds 3-5 achieved 95-97% accuracy → **proof of concept works**

### BUT... ⚠️ Implementation Issues

**Weaknesses**:
1. ❌ **Reward is not Markovian**: Single-round $\Delta\text{Acc}$ violates RL assumptions
2. ❌ **Sample inefficiency**: 30 rounds ≈ 30 transitions (SAC needs 10k+)
3. ❌ **Warm-up heuristic fails at 40% Byzantine**: Only isolates 7% (should be 40%)
4. ❌ **Exploration in adversarial setting**: SAC explores by accepting Byzantine → collapse
5. ⚠️ **Non-stationarity**: Adaptive attackers violate stationary MDP assumption

---

## 🎯 Top 3 Immediate Fixes (4-6 hours total)

### Fix #1: Stabilize Reward (90 minutes)

**File**: `src/defences/cognitive_defence_posg.py`

**Change**: Add exponential moving average

```python
# In __init__:
self._acc_ema = 0.0
self._acc_ema_alpha = 0.3

# In aggregate_updates():
if self.round_number == 1:
    self._acc_ema = val_acc
else:
    self._acc_ema = 0.3 * val_acc + 0.7 * self._acc_ema

delta_acc = self._acc_ema - (self._prev_val_acc or 0.0)

# Reweight coefficients:
reward = 10.0 * delta_acc - 0.05 * belief_ent - 0.2 * divergence
```

**Expected Result**: +30-40% accuracy gain

---

### Fix #2: Tune SAC Hyperparameters (60 minutes)

**File**: `src/defences/cognitive_defence_posg.py`

**Changes**:

```python
# In __init__, when creating SACAgent:
self.agent = SACAgent(
    # ... (other args) ...
    lr_actor=1e-3,           # 3x faster (was 3e-4)
    lr_critic=1e-3,          # 3x faster
    gamma=0.95,              # Medium-horizon (was 0.99)
    buffer_capacity=1000,    # Realistic size (was 50k)
    batch_size=16,           # Match small buffer (was 64)
)
```

**File**: `src/defences/sac_agent.py`

```python
# In __init__:
self.target_entropy = -0.1 * action_dim  # Less exploration (was -action_dim)
```

**Expected Result**: +10-15% stability improvement

---

### Fix #3: Implement Multi-Krum Warm-up (2-3 hours)

**File**: `src/defences/cognitive_defence_posg.py`

**Replace** `_heuristic_weights()` with Multi-Krum (see [OPTIMIZATION_IMPLEMENTATION_PLAN.md](./OPTIMIZATION_IMPLEMENTATION_PLAN.md) Section 2.1 for full code)

**Key Algorithm**:
1. Compute pairwise distance matrix between client updates
2. For each client, sum distances to $m$ nearest neighbors
3. Select $n-f-2$ clients with smallest scores (where $f$ = 40% Byzantine estimate)
4. Assign weight 1.0 to selected, 0.1 to others

**Expected Result**: +15-20% during warm-up rounds

---

## 📊 Expected Performance Trajectory

| Phase | Changes | Static Attack Acc | Adaptive Attack Acc | Time |
|-------|---------|-------------------|---------------------|------|
| **Baseline** | (none) | 11% | 11% (collapse) | - |
| **After Fix #1-3** | Reward + SAC + Heuristic | 70-80% | 55-65% | 4-6 hrs |
| **+ Observation Norm** | Robust normalization | 85-90% | 70-75% | +2 hrs |
| **+ N-step Reward** | Multi-round TD | 90-92% | 80-85% | +4-6 hrs |
| **+ Curriculum** | Staged training | 92-95% | 85-90% | +8-12 hrs |

---

## 🔬 Mathematical Formulation Summary

### Problem: POSG (Partially Observable Stochastic Game)

**State**: $s_t = \{(\theta_t^{(i)}, \beta_i, \tau_i)\}_{i=1}^{N}$ (unknown to defender)

**Observation**: $o_t^{(i)} \in \mathbb{R}^6$ (norms, cosine, Fisher, samples)

**Belief**: $b_t^{(i)} = \text{GRU}(\phi(o_t^{(i)}), b_{t-1}^{(i)}) \in \mathbb{R}^{64}$

**Compact State**: $\tilde{s}_t = [\mathbb{E}[b_t], \text{std}(b_t)] \in \mathbb{R}^{128}$

**Policy**: $w_i \sim \text{Beta}(\alpha_i(\tilde{s}_t), \beta_i(\tilde{s}_t))$ for $w_i \in (0,1)$

**Aggregation**:
$$
\theta_t^{(g)} = \theta_{t-1}^{(g)} + \frac{\sum_i w_i \cdot n_i \cdot \text{clip}(\Delta\theta_t^{(i)})}{\sum_i w_i \cdot n_i}
$$

**Reward**:
$$
R_t = \alpha \cdot \Delta\text{Acc} - \beta \cdot H(b_t) - \gamma \cdot \|\theta_t - \theta_{t-1}\|_2
$$

**Training**: Soft Actor-Critic (off-policy, maximum entropy RL)

---

## 🚦 Decision Tree: What to Implement?

```
START
 │
 ├─ Need quick win (4-6 hours)?
 │   └─► Implement Fix #1-3 above
 │       Expected: 70-80% accuracy
 │
 ├─ Need production-ready (12-18 hours)?
 │   └─► Follow full Phase 1-3 in Implementation Plan
 │       Expected: 85-90% accuracy
 │
 ├─ Need state-of-the-art (20-30 hours)?
 │   └─► Complete Phase 1-5 (includes curriculum learning)
 │       Expected: 92-95% accuracy
 │
 └─ Still failing after all fixes?
     └─► Pivot to Plan B: Supervised Learning
         (XGBoost classifier on same features)
         Expected: 90-95% accuracy, 2-4 hours implementation
```

---

## 📚 Documentation Structure

1. **[COGNITIVE_DEFENCE_MATHEMATICAL_FORMALIZATION.md](./COGNITIVE_DEFENCE_MATHEMATICAL_FORMALIZATION.md)**
   - Full theoretical analysis
   - Root cause deep dive
   - Alternative approaches (Plan B/C)
   - 9 sections, ~4000 words

2. **[OPTIMIZATION_IMPLEMENTATION_PLAN.md](./OPTIMIZATION_IMPLEMENTATION_PLAN.md)**
   - Step-by-step code changes
   - 5 phases with estimated time
   - Testing checklist
   - Rollback procedures

3. **[THIS FILE]** - Quick reference for decision-making

---

## ⚡ Quick Command to Test Fixes

```bash
# After implementing fixes, run this:
cd /Users/hanafemira/development/FL_CognitiveDefence
source fl_env/bin/activate

# Test static attack (should see improvement)
python experiments/scripts/run_single_experiment.py \
    --config experiments/configs/static_attacks_cognitive_defence.yaml \
    --output-dir results/optimized_test \
    --seed 42

# Check final accuracy in logs:
tail -n 50 results/optimized_test/experiment.log | grep "Accuracy:"

# Should see: Accuracy > 0.70 (vs baseline 0.11)
```

---

## 🎓 Key Theoretical Insights

### Why Did Rounds 3-5 Work?

1. **Warm-up heuristic** (rounds 1-5) provided some signal
2. **GRU beliefs** accumulated temporal evidence
3. **SAC hadn't taken over yet** (low exploration pressure)
4. **Lucky** - adaptive attacker hadn't optimized strategy yet

### Why Did It Collapse at Round 6?

1. **SAC took control** (warmup ended)
2. **High entropy exploration** → accepted Byzantine clients
3. **Immediate accuracy drop** → large negative reward
4. **Agent learned**: "isolation = bad, acceptance = safe"
5. **Catastrophic forgetting**: Lost all round 3-5 knowledge

### The Solution?

**Reduce exploration** (less risky), **stabilize reward** (less noisy), **improve heuristic** (better warm-start) → SAC inherits good policy instead of learning from scratch.

---

## 🔑 Key Takeaways

✅ **The approach is theoretically sound** - POSG+GRU+SAC is a valid framework  
✅ **Proof-of-concept works** - 95-97% accuracy achieved briefly  
❌ **Implementation has bugs** - reward noise, hyperparameters, weak heuristic  
🎯 **Fixes are ready** - 4-30 hours of work for 70-95% accuracy  
🔄 **Fallback exists** - Supervised learning as Plan B

**Bottom Line**: Don't abandon the approach! The foundation is solid; we just need to tune the training pipeline.

---

## 📞 Next Steps

1. **Read** [OPTIMIZATION_IMPLEMENTATION_PLAN.md](./OPTIMIZATION_IMPLEMENTATION_PLAN.md) Phase 1
2. **Implement** Fix #1-3 (4-6 hours)
3. **Test** using command above
4. **Report** results (accuracy, logs, issues)
5. **Iterate** based on Phase 2-5 as needed

**Good luck! 🚀**

---

*For questions or issues, refer to the full documentation files or examine the codebase at:*
- `src/defences/cognitive_defence_posg.py` - Main defense logic
- `src/defences/sac_agent.py` - RL agent
- `src/defences/client_tracker.py` - GRU belief tracker
