# Adaptive Attacks in Federated Learning

## Overview

Adaptive attacks are sophisticated adversarial strategies that modify their behavior based on the defense mechanism's responses. Unlike static attacks (e.g., label flipping, gradient noise), adaptive attacks learn from feedback and optimize their strategy to evade detection while maximizing impact on the global model.

This document describes four key adaptive attack strategies implemented in this framework:

1. **stat-opt** (Statistical Optimization Attack)
2. **dny-opt** (Dynamic Optimization Attack)  
3. **min-max** (Minimax Attack)
4. **min-sum** (Minimum Sum Attack)

---

## 1. Statistical Optimization Attack (stat-opt)

### Description
The Statistical Optimization Attack (stat-opt) crafts malicious updates that statistically mimic benign updates to evade statistical defenses like trimmed mean, median, and Krum. The attack optimizes updates to stay within the statistical bounds of honest clients.

### Methodology

**Goal**: Minimize statistical distance from benign updates while maximizing attack impact

**Strategy**:
1. **Statistical Analysis**: Compute mean (μ) and standard deviation (σ) of benign client updates
2. **Constraint Optimization**: Craft malicious update m such that:
   - `||m - μ|| ≤ k·σ` where k is a constraint factor (typically 1-2)
   - Maximize damage within the statistical constraint
3. **Adaptive Adjustment**: If detected (update rejected), reduce k and retry

**Algorithm**:
```
Input: Target model parameters θ*, benign updates {u₁, ..., uₙ}
Output: Crafted malicious update m

1. Compute statistics:
   μ = mean({u₁, ..., uₙ})
   σ = std({u₁, ..., uₙ})

2. Generate base malicious update:
   m₀ = attack_objective(θ*)  # E.g., flip gradients

3. Project to statistical bounds:
   direction = normalize(m₀ - μ)
   magnitude = min(||m₀ - μ||, k·σ)
   m = μ + direction * magnitude

4. Return m
```

### Defense Evasion
- **Trimmed Mean**: Stays within trimming bounds
- **Krum**: Appears close to cluster of benign updates
- **Median**: Aligns with median statistics

### Parameters
- `intensity`: Base attack strength (0.0-1.0)
- `constraint_factor`: Multiplier for standard deviation bound (default: 1.5)
- `adaptive_learning_rate`: Rate of constraint adjustment (default: 0.1)

### References
- Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning" (USENIX Security 2020)
- Baruch et al., "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)

---

## 2. Dynamic Optimization Attack (dny-opt)

### Description
Dynamic Optimization Attack (dny-opt) continuously adapts attack parameters based on real-time feedback from the defense mechanism. It tracks which updates are accepted/rejected and dynamically adjusts intensity, direction, and strategy.

### Methodology

**Goal**: Maximize cumulative attack impact over multiple rounds through adaptive learning

**Strategy**:
1. **Feedback Collection**: Track which updates were accepted vs. rejected
2. **Strategy Learning**: Use reinforcement learning to adjust attack parameters
3. **Multi-Armed Bandit**: Treat different attack intensities as arms, select based on success rate
4. **Temporal Adaptation**: Increase stealth when detection rate is high

**Algorithm**:
```
State: S = {detection_rate, acceptance_rate, round_number}
Actions: A = {intensity levels, noise types, target selection}

1. Initialize Q-table for state-action pairs
2. For each round t:
   a. Observe current state s_t
   b. Select action a_t using ε-greedy policy
   c. Execute attack with selected parameters
   d. Observe reward r_t (1 if accepted, -1 if detected, bonus for impact)
   e. Update Q(s_t, a_t) ← Q(s_t, a_t) + α[r_t + γ·max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
   f. Update state s_{t+1}
```

### Adaptation Mechanisms
1. **Intensity Modulation**: Reduce when detection rate > threshold
2. **Technique Switching**: Alternate between gradient noise, scaling, sign flip
3. **Target Rotation**: Change targeted parameters to avoid pattern detection
4. **Timing Variation**: Skip rounds to reduce detection correlation

### Defense Evasion
- **Cognitive Defense**: Learns reputation decay patterns
- **Adaptive Defenses**: Counters with counter-adaptation
- **History-based**: Varies patterns to avoid historical profiling

### Parameters
- `learning_rate`: Q-learning update rate (default: 0.1)
- `exploration_rate`: ε for ε-greedy policy (default: 0.1)
- `discount_factor`: γ for future reward discounting (default: 0.95)
- `intensity_levels`: Discrete set of attack intensities to choose from
- `detection_threshold`: Threshold to trigger defensive mode (default: 0.7)

### References
- Shejwalkar & Houmansadr, "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning" (NDSS 2021)

---

## 3. Minimax Attack (min-max)

### Description
The Minimax Attack (min-max) formulates the attack as a game-theoretic problem, finding the optimal attack that minimizes the best-case performance of any defense strategy. It assumes the defender will respond optimally and prepares accordingly.

### Methodology

**Goal**: Find attack that guarantees maximum damage under optimal defense

**Game Formulation**:
- **Players**: Attacker vs. Defense aggregation rule
- **Attacker Strategy**: Choose malicious update m
- **Defender Strategy**: Choose aggregation function f(m, {benign updates})
- **Payoff**: Model accuracy drop (attacker wants to maximize, defender wants to minimize)

**Strategy**:
```
Objective: max_m min_f [Impact(f(m, U_benign))]

Where:
- m = malicious update
- f = defense aggregation function
- U_benign = set of benign updates
- Impact = negative effect on model accuracy
```

**Algorithm**:
```
Input: Benign updates U = {u₁, ..., uₙ}, model θ
Output: Minimax optimal attack m*

1. Initialize attack candidates M = {}
2. For each defense strategy f in {trimmed_mean, krum, median, ...}:
   a. For each attack intensity λ:
      i. Compute m_λ = λ·malicious_direction
      ii. Evaluate worst-case: v_λ,f = min_f Impact(f(m_λ, U))
   b. Select m_f = argmax_λ v_λ,f
   c. Add m_f to M

3. Select m* = argmax_{m ∈ M} min_{f} Impact(f(m, U))
4. Return m*
```

### Defense Evasion
The minimax approach explicitly considers the defense mechanism's optimal response:

- **Trimmed Mean**: Crafts updates just inside the trimming threshold
- **Krum**: Positions within k-nearest neighbors of benign cluster
- **Median**: Shifts median without being outlier
- **Cognitive Defense**: Balances immediate impact vs. reputation damage

### Computational Approach
Since enumerating all defenses is intractable, we use a **threat model** with likely defenses:

```python
defense_ensemble = {
    'trimmed_mean': weight=0.3,
    'krum': weight=0.25,
    'median': weight=0.2,
    'cognitive': weight=0.25
}
```

### Parameters
- `intensity`: Base attack strength
- `defense_models`: List of defense strategies to consider
- `optimization_steps`: Iterations for finding minimax solution (default: 10)
- `threat_model_weights`: Prior over likely defense strategies

### References
- Bhagoji et al., "Analyzing Federated Learning through an Adversarial Lens" (ICML 2019)
- Cao et al., "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping" (NDSS 2021)

---

## 4. Minimum Sum Attack (min-sum)

### Description
The Minimum Sum Attack (min-sum) crafts malicious updates that minimize the sum of distances to all benign updates while still achieving attack objectives. This makes the attack appear as a "centrist" update, highly trusted by distance-based defenses.

### Methodology

**Goal**: Minimize total distance to benign updates while maximizing attack impact

**Optimization Problem**:
```
minimize: Σᵢ ||m - uᵢ||²  (distance to benign updates)
subject to: Impact(m) ≥ τ   (maintain attack effectiveness)
```

**Strategy**:
1. **Centroid Calculation**: Compute geometric center of benign updates
2. **Direction Selection**: Choose attack direction toward target objective
3. **Magnitude Optimization**: Find maximum attack magnitude that keeps sum of distances minimal
4. **Iterative Refinement**: Use gradient descent to fine-tune the malicious update

**Algorithm**:
```
Input: Benign updates U = {u₁, ..., uₙ}, attack objective θ_target
Output: Min-sum optimal attack m*

1. Compute benign centroid:
   c = (1/n)·Σᵢ uᵢ

2. Define attack direction:
   d = normalize(θ_target - c)

3. Optimize magnitude α:
   minimize_{α} Σᵢ ||c + α·d - uᵢ||²
   subject to: ||α·d|| ≥ attack_threshold

4. Return m* = c + α*·d
```

### Geometric Interpretation
The min-sum attack positions itself at the weighted centroid of benign updates, then nudges in the attack direction:

```
    u₁     u₂
      \   /
       \ /
        m* ← positioned near centroid
       / \
      /   \
    u₃     u₄
```

This makes `m*` appear as a "consensus" update.

### Defense Evasion
- **Krum**: Minimizes sum of distances, appears as the most "central" update
- **Multi-Krum**: Gets selected in the top-k set
- **Geometric Median**: Naturally aligns with geometric median
- **Reputation Systems**: Builds trust by appearing consistent

### Parameters
- `intensity`: Attack strength (magnitude in attack direction)
- `distance_weight`: Balance between minimizing distance vs. maximizing impact (default: 0.7)
- `optimization_lr`: Learning rate for gradient descent optimization (default: 0.01)
- `max_iterations`: Maximum optimization steps (default: 100)
- `convergence_threshold`: Stopping criterion for optimization (default: 1e-5)

### References
- Baruch et al., "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
- Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates" (ICML 2018)

---

## Implementation Considerations

### Feedback Mechanism
All adaptive attacks require feedback from the server about aggregation results. We implement this through:

```python
class AdaptiveAttack(BaseAttack):
    def update_feedback(self, round_num: int, was_accepted: bool, 
                       global_accuracy: float, anomaly_score: float):
        """Called by client after receiving server response"""
        self.feedback_history.append({
            'round': round_num,
            'accepted': was_accepted,
            'accuracy': global_accuracy,
            'anomaly_score': anomaly_score
        })
        self.adapt_strategy()
```

### Attack Metrics
Track effectiveness with:
- **Stealth Score**: Fraction of updates accepted
- **Impact Score**: Drop in global model accuracy
- **Efficiency**: Impact per unit of detection risk
- **Adaptive Gain**: Improvement over non-adaptive baseline

### Ethical Considerations
These attacks are implemented for **defense research purposes only**:
1. Test robustness of defense mechanisms
2. Develop better Byzantine-robust aggregation
3. Understand federated learning security
4. Never deploy against real-world systems without authorization

---

## Usage Example

```python
from src.attacks.adaptive import StatOptAttack, DnyOptAttack, MinMaxAttack, MinSumAttack

# Statistical Optimization Attack
stat_attack = StatOptAttack(
    intensity=0.2,
    constraint_factor=1.5,
    target_clients=[0, 1, 2]
)

# Dynamic Optimization Attack  
dny_attack = DnyOptAttack(
    intensity=0.15,
    learning_rate=0.1,
    exploration_rate=0.1,
    target_clients=[3, 4]
)

# Minimax Attack
minmax_attack = MinMaxAttack(
    intensity=0.2,
    defense_models=['krum', 'trimmed_mean', 'cognitive'],
    optimization_steps=10,
    target_clients=[5, 6]
)

# Minimum Sum Attack
minsum_attack = MinSumAttack(
    intensity=0.2,
    distance_weight=0.7,
    optimization_lr=0.01,
    target_clients=[7, 8]
)
```

## Configuration Example

```yaml
attacks:
  - enabled: true
    attack_type: "stat_opt"
    intensity: 0.2
    constraint_factor: 1.5
    target_clients: [0, 1, 2]
  
  - enabled: true
    attack_type: "dny_opt"
    intensity: 0.15
    learning_rate: 0.1
    target_clients: [3, 4]
    
  - enabled: true
    attack_type: "min_max"
    intensity: 0.2
    defense_models: ["krum", "trimmed_mean"]
    target_clients: [5, 6]
    
  - enabled: true
    attack_type: "min_sum"
    intensity: 0.2
    distance_weight: 0.7
    target_clients: [7, 8]
```

---

## References

1. Fang, M., Cao, X., Jia, J., & Gong, N. (2020). Local model poisoning attacks to Byzantine-robust federated learning. *USENIX Security Symposium*.

2. Baruch, M., Baruch, G., & Goldberg, Y. (2019). A little is enough: Circumventing defenses for distributed learning. *NeurIPS*.

3. Shejwalkar, V., & Houmansadr, A. (2021). Manipulating the Byzantine: Optimizing model poisoning attacks and defenses for federated learning. *NDSS*.

4. Bhagoji, A. N., Chakraborty, S., Mittal, P., & Calo, S. (2019). Analyzing federated learning through an adversarial lens. *ICML*.

5. Cao, X., Fang, M., Liu, J., & Gong, N. (2021). FLTrust: Byzantine-robust federated learning via trust bootstrapping. *NDSS*.

6. Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. *ICML*.

7. Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. *NeurIPS*.
