# How Anomaly Scoring Works in FL_CognitiveDefence

## Overview

Your system uses **statistical anomaly detection based on Z-scores** to identify malicious client updates in federated learning. The scoring happens in the **Orient** phase of the OODA loop.

---

## Step-by-Step Anomaly Scoring Process

### Step 1: **Observe** - Collect Raw Metrics

First, the system observes each client's update and calculates the **L2 norm** of their parameters:

```python
def observe(self, client_updates):
    observations = {}
    for client_id, (parameters, num_samples, metrics) in client_updates.items():
        # Calculate L2 norm for each parameter layer
        param_norms = [float(np.linalg.norm(param)) for param in parameters]
        
        observations[client_id] = {
            'param_norms': param_norms,        # Norm per layer
            'total_norm': sum(param_norms),    # Sum of all norms
            'num_samples': num_samples,
            'avg_norm': sum(param_norms) / len(param_norms),
            'update_time': datetime.now().isoformat()
        }
    return observations
```

**What's measured:**
- `total_norm`: The magnitude of the client's parameter update (how "big" the update is)
- This captures whether a client is pushing extreme weight changes

---

### Step 2: **Orient** - Calculate Z-Score (The Anomaly Score)

The system compares the current update against **historical behavior** using statistical analysis:

```python
def orient(self, observations):
    analysis = {}
    
    if len(self.historical_updates) > 2:  # Need history for comparison
        # Get historical norms from past rounds
        historical_norms = [update['total_norm'] for update in self.historical_updates]
        mean_norm = np.mean(historical_norms)
        std_norm = np.std(historical_norms)
        
        for client_id, obs in observations.items():
            # Calculate Z-score: how many standard deviations from mean?
            z_score = abs(obs['total_norm'] - mean_norm) / (std_norm + 1e-8)
            
            # Is it anomalous?
            is_anomalous = z_score > 2.0  # 2 standard deviations
            
            # Normalize confidence to 0-1 range
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
        # Insufficient history - trust everyone initially
        for client_id in observations.keys():
            analysis[client_id] = {
                'z_score': 0.0,
                'is_anomalous': False,
                'confidence': 0.0,
                'deviation_from_mean': 0.0,
                'historical_context': {'insufficient_history': True}
            }
    
    return analysis
```

---

## The Anomaly Score: Z-Score

### What is Z-Score?

The **Z-score** measures how many standard deviations away from the mean a value is:

```
z_score = |current_norm - mean_norm| / std_norm
```

**Interpretation:**
- `z_score = 0`: Perfectly normal (at the mean)
- `z_score = 1`: 1 standard deviation from mean (still normal)
- `z_score = 2`: 2 standard deviations (suspicious)
- `z_score > 2`: **Anomalous** (flag as malicious)
- `z_score = 3+`: Highly anomalous

### Visual Example

```
Normal Distribution of Update Norms:

                    ┌────────┐
                    │ Normal │
             ┌──────┴────────┴──────┐
         ┌───┴───┐            ┌───┴───┐
    ┌────┴───────┴────────────┴───────┴────┐
    │                                        │
────┴────────────────────────────────────────┴────
   -3σ    -2σ    -1σ    μ    +1σ   +2σ   +3σ
                               ↑           ↑
                          Threshold    Attack!
                          (z=2.0)
```

---

## Step 3: **Decide** - Apply Threshold and Take Action

The system uses the anomaly determination to make decisions:

```python
def decide(self, analysis):
    decisions = {}
    
    for client_id, client_analysis in analysis.items():
        current_reputation = self.get_client_reputation(client_id)
        
        if client_analysis['is_anomalous']:  # z_score > 2.0
            # PUNISH: Reduce reputation and weight
            new_reputation = current_reputation * self.reputation_decay  # 0.8
            weight_multiplier = max(new_reputation, 0.1)  # Floor at 10%
            
            decisions[client_id] = {
                'action': 'reduce_weight',
                'weight_multiplier': weight_multiplier,
                'reason': f"Anomalous update (z-score: {z_score:.2f})"
            }
            
            self.update_client_reputation(client_id, new_reputation - current_reputation)
            
        else:
            # REWARD: Good behavior
            reputation_bonus = 0.05
            new_reputation = min(current_reputation + reputation_bonus, 1.0)
            
            decisions[client_id] = {
                'action': 'accept',
                'weight_multiplier': 1.0,  # Full weight
                'reason': f"Normal update (z-score: {z_score:.2f})"
            }
            
            self.update_client_reputation(client_id, reputation_bonus)
    
    return decisions
```

---

## Key Parameters Explained

### 1. **Anomaly Threshold** (Not directly used in Z-score, but conceptual)

```python
self.anomaly_threshold = 0.7  # Config parameter (not actively used)
```

Currently, your system uses a **hardcoded Z-score threshold of 2.0**:
```python
is_anomalous = z_score > 2.0
```

**You could modify this to use the configurable threshold:**
```python
is_anomalous = confidence > self.anomaly_threshold
# where confidence = min(z_score / 3.0, 1.0)
```

### 2. **Reputation Decay**

```python
self.reputation_decay = 0.8  # 20% penalty per anomalous round
```

When anomaly detected:
```python
new_reputation = current_reputation * 0.8
```

**Example progression:**
- Start: `1.0` (trusted)
- After 1st attack: `0.8`
- After 2nd attack: `0.64`
- After 3rd attack: `0.512`
- After 4th attack: `0.410`

The client's weight in aggregation decreases proportionally.

### 3. **Event Buffer Size (History Size)**

```python
self.historical_updates = deque(maxlen=history_size)  # default: 100
```

This stores the last 100 update observations to calculate the mean and standard deviation. 

**Trade-offs:**
- **Larger buffer (100+)**: More stable statistics, less sensitive to recent changes
- **Smaller buffer (20-50)**: More adaptive, faster detection of new attack patterns

---

## Complete Flow Example

### Scenario: Client sends poisoned update in Round 5

**Round 1-4:** Normal training
- Client A norm: `[10.2, 10.5, 10.3, 10.4]`
- Mean: `10.35`, Std Dev: `0.13`

**Round 5:** Attack!
- Client A sends poisoned update with norm: `25.0`

**Anomaly Detection:**
```python
z_score = abs(25.0 - 10.35) / 0.13 = 112.7 (!!)
is_anomalous = 112.7 > 2.0  # TRUE
confidence = min(112.7 / 3.0, 1.0) = 1.0  # Maximum confidence
```

**Decision:**
```python
current_reputation = 1.0
new_reputation = 1.0 * 0.8 = 0.8
weight_multiplier = max(0.8, 0.1) = 0.8

Action: reduce_weight
Reason: "Anomalous update detected with z-score 112.70. 
         Reducing client weight from 1.00 to 0.80"
```

**Aggregation:**
- Client A's update is down-weighted to 80% of its original contribution
- If it attacks again, it drops to 64%, then 51%, etc.

---

## Why This Works

1. **Adaptive**: Uses historical behavior, not fixed rules
2. **Statistically sound**: Z-scores are robust for outlier detection
3. **Graceful degradation**: Reputation system allows recovery
4. **Explainable**: Clear reasoning for each decision

---

## Potential Improvements

### 1. Make threshold configurable
```python
is_anomalous = (z_score / 3.0) > self.anomaly_threshold
```

### 2. Add multiple scoring dimensions
```python
# Current: Only total_norm
# Proposed: Also check gradient direction, layer-wise norms, etc.
```

### 3. Adaptive threshold per client
```python
# Different thresholds for different clients based on their history
```

### 4. Temporal patterns
```python
# Detect attacks spread across multiple rounds
```

---

## Summary

**Your system scores anomalies using:**
1. **Metric**: L2 norm of parameter updates
2. **Method**: Z-score (standard deviations from historical mean)
3. **Threshold**: Z-score > 2.0 = anomalous
4. **Confidence**: Normalized as `min(z_score / 3.0, 1.0)`
5. **Action**: Multiply client weight by decayed reputation (0.8^n)

This is a **statistical distance-based method** that's computationally efficient and works well for detecting model poisoning attacks!
