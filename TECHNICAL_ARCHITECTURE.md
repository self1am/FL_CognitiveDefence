# Technical Architecture: Federated Learning with Cognitive Defence

## Overview

This document provides a comprehensive technical overview of our Federated Learning system with Cognitive Defence mechanisms. The architecture implements an adaptive Byzantine-robust aggregation strategy based on the OODA (Observe-Orient-Decide-Act) loop and MAPE-K (Monitor-Analyze-Plan-Execute over a Knowledge base) frameworks, designed to defend against adversarial attacks in federated learning environments.

## 1. System Architecture

### 1.1 Core Components

The system consists of four primary architectural layers:

```
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                        │
│  (ExperimentRunner, ClientOrchestrator)                     │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼─────────────┐    ┌─────▼──────────────┐
│  Server Layer   │    │   Client Layer     │
│  (Aggregation)  │◄───┤  (Local Training)  │
└───┬─────────────┘    └──────┬─────────────┘
    │                         │
┌───▼─────────────┐    ┌─────▼──────────────┐
│ Defence Layer   │    │  Attack Layer      │
│ (OODA/MAPE-K)   │    │ (Adversarial Sim)  │
└─────────────────┘    └────────────────────┘
```

**Orchestration Layer**: Manages the entire experimental lifecycle, including server initialization, client spawning, resource monitoring, and results aggregation.

**Server Layer**: Implements the central aggregator using the Flower framework, coordinating federated rounds and applying defence strategies during model aggregation.

**Client Layer**: Executes local model training on distributed data partitions, with optional adversarial behavior simulation.

**Defence Layer**: Contains pluggable defence strategies that process client updates before aggregation.

**Attack Layer**: Simulates various adversarial behaviors (label flipping, gradient noise, model replacement) for defence evaluation.

## 2. Data Distribution Strategy

### 2.1 Non-IID Data Partitioning

The system implements a **non-IID (non-Independent and Identically Distributed)** data distribution strategy using the **Dirichlet distribution** to simulate realistic federated learning scenarios where clients have heterogeneous data.

#### Implementation Details

**Algorithm**: For a dataset with C classes and N clients, we use Dir(α) where α controls the degree of heterogeneity:

```python
proportions = Dirichlet(α × [1, 1, ..., 1]_C) for each client
```

**Parameters**:
- α (alpha): Controls data heterogeneity
  - α → 0: Highly non-IID (clients get few classes)
  - α → ∞: Approaches IID (uniform distribution)
  - Typical value: α = 0.5 (moderate heterogeneity)

**Process**:
1. Sample class proportions for each client from Dir(α)
2. Allocate data samples to clients based on these proportions
3. Each client receives a different class distribution

**Example Distribution** (α = 0.5, 10 clients, MNIST):
- Client 0: 60% digit '7', 30% digit '2', 10% others
- Client 1: 45% digit '3', 40% digit '8', 15% others
- Client 2: 70% digit '1', 20% digit '5', 10% others
- ... (varying distributions)

**Rationale**: This reflects real-world federated scenarios where:
- Mobile devices have user-specific data patterns
- Hospitals have region-specific disease distributions
- IoT sensors collect environment-specific measurements

### 2.2 Dataset Handling

**Dataset**: MNIST (28×28 grayscale digit images, 10 classes)
- Training set: 60,000 samples
- Test set: 10,000 samples (centralized evaluation)

**Preprocessing**:
- Normalization: mean=0.1307, std=0.3081
- Tensor conversion: [0, 255] → [0, 1] → Normalized

**Data Loading**:
- Batch size: Configurable (default: 32)
- Client-specific DataLoaders with shuffling
- Centralized test set for unbiased global model evaluation

## 3. Experiment Execution Methodology

### 3.1 Federated Learning Protocol

The system follows the standard Federated Averaging (FedAvg) protocol with defence enhancements:

```
FOR round r = 1 to R:
    1. Server broadcasts global model θ_r to clients
    2. Server selects subset of clients S_r (min_clients)
    3. FOR each client i ∈ S_r IN PARALLEL:
        a. Download global model θ_r
        b. Train locally for E epochs → get Δθ_i
        c. [Optional] Apply adversarial attack
        d. Upload update Δθ_i to server
    4. Server collects updates {Δθ_i | i ∈ S_r}
    5. Defence mechanism processes updates → filtered updates
    6. Aggregate filtered updates → θ_{r+1}
    7. Evaluate θ_{r+1} on centralized test set
    8. Log metrics and decisions
END FOR
```

### 3.2 Orchestration Details

**Client Management**:
- **Batch Spawning**: Clients are spawned in batches (default: 2-3) to manage memory
- **Resource Monitoring**: Tracks memory usage to prevent system overload
- **Spawn Delay**: Configurable delay between batches (default: 2s)
- **Process Isolation**: Each client runs as an independent process

**Configuration System**:
- YAML-based experiment configuration
- Separate configs for experiment, defence, attacks, and orchestration
- Deterministic seeding for reproducibility (seed: 42)

**Example Configuration**:
```yaml
experiment:
  num_rounds: 10
  min_clients: 2
  seed: 42

defence:
  strategy: "cognitive_defence"
  anomaly_threshold: 0.7

attacks:
  - attack_type: "label_flip"
    intensity: 0.1
    target_clients: [0, 1, 2]

orchestration:
  num_clients: 10
  batch_size: 2
```

### 3.3 Deterministic Execution

To ensure reproducibility:
- **Seed Control**: All random operations use fixed seeds
- **PyTorch**: `torch.manual_seed(seed)`, `torch.cuda.manual_seed(seed)`
- **NumPy**: `np.random.seed(seed)`
- **Deterministic Algorithms**: `torch.use_deterministic_algorithms(True)` where applicable

## 4. Cognitive Defence Strategy

### 4.1 Theoretical Foundation

Our Cognitive Defence integrates two complementary frameworks:

**OODA Loop** (Military Decision-Making):
- **Observe**: Collect information about client updates
- **Orient**: Analyze observations in context
- **Decide**: Determine appropriate actions
- **Act**: Execute decisions and aggregate

**MAPE-K Framework** (Autonomous Computing):
- **Monitor**: Track client behavior and update patterns
- **Analyze**: Detect anomalies using statistical methods
- **Plan**: Decide on aggregation weights and filtering
- **Execute**: Apply weighted aggregation
- **Knowledge**: Maintain historical context and reputation

### 4.2 Implementation Details

#### Phase 1: Observe (Monitor)
```python
observations = {
    client_id: {
        'param_norms': [||θ₁||, ||θ₂||, ...],
        'total_norm': Σ||θᵢ||,
        'avg_norm': mean(||θᵢ||),
        'num_samples': n_i
    }
}
```

**Metrics Collected**:
- Parameter-wise L2 norms
- Total update magnitude
- Average parameter norm
- Training sample count
- Timestamp

#### Phase 2: Orient (Analyze)

**Statistical Anomaly Detection**:

For each client update, compute Z-score:
```
Z_i = (||Δθ_i|| - μ_historical) / σ_historical
```

Where:
- μ_historical: Mean of historical update norms
- σ_historical: Standard deviation of historical norms

**Anomaly Detection**:
```
is_anomalous = |Z_i| > threshold (default: 2.0)
confidence = min(|Z_i| / 3.0, 1.0)
```

**Historical Context**:
- Window size: Last 100 updates (configurable)
- Requires ≥3 historical updates for meaningful statistics
- Falls back to neutral assessment if insufficient history

#### Phase 3: Decide (Plan)

**Reputation-Based Decision Making**:

```python
IF is_anomalous:
    reputation_new = reputation_old × decay_factor (default: 0.8)
    weight_multiplier = max(reputation_new, 0.1)
    decision = "reduce_weight"
ELSE:
    reputation_new = min(reputation_old + 0.05, 1.0)
    weight_multiplier = 1.0
    decision = "accept"
```

**Reputation System**:
- Initial reputation: 1.0 (full trust)
- Decay on anomaly: multiply by 0.8
- Reward on normal: add 0.05 (capped at 1.0)
- Minimum weight: 0.1 (never fully exclude)

**Explainable Decisions**:
Each decision includes:
- Decision type: "accept", "reduce_weight", "reject"
- Confidence score: [0, 1]
- Reasoning: Human-readable explanation
- Evidence: {z_score, reputation, historical_context}

#### Phase 4: Act (Execute)

**Weighted Aggregation**:

```
θ_global = Σ(w_i × Δθ_i) / Σ(w_i)
```

Where:
```
w_i = reputation_i × n_i
```

- n_i: Number of training samples (standard FedAvg weight)
- reputation_i: Cognitive defence weight adjustment

**Aggregation Process**:
1. Apply reputation multiplier to sample weights
2. Compute weighted sum of parameters
3. Normalize by total adjusted weight
4. Log aggregation statistics

### 4.3 Key Differentiators

**Adaptive Learning**:
- Reputation evolves over time based on client behavior
- Historical context informs current decisions
- Gradual trust adjustment (not binary accept/reject)

**Explainability**:
- Every decision includes detailed reasoning
- Evidence-based explanations for auditing
- Transparency for debugging and analysis

**Resilience**:
- Minimum weight prevents complete exclusion
- Handles insufficient historical data gracefully
- Robust to temporary anomalies

## 5. Comparison with Alternative Defence Strategies

### 5.1 Krum Defence

**Algorithm Overview**:
Krum selects the client update with the smallest sum of squared distances to its nearest neighbors.

**Formal Definition**:

For n clients with updates {Δθ₁, ..., Δθₙ} and f expected Byzantine clients:

1. Compute pairwise distances:
   ```
   d(i,j) = ||Δθᵢ - Δθⱼ||₂
   ```

2. For each client i, compute score:
   ```
   S_i = Σ d(i,j)² over (n-f-2) nearest neighbors
   ```

3. Select update with minimum score:
   ```
   θ_global = Δθ_argmin(S_i)
   ```

**Multi-Krum Variant**:
Average the top (n-f-2) updates instead of selecting one.

**Key Characteristics**:
- **Selection-Based**: Chooses "representative" update(s)
- **Distance-Based**: Uses geometric similarity
- **Binary Decision**: Clients either fully selected or fully rejected
- **No Learning**: No adaptation over rounds
- **Robust Guarantee**: Tolerates up to f < (n-2)/3 Byzantine clients

**Limitations**:
1. **High Variance**: Single update may not represent all honest clients
2. **Data Loss**: Discards potentially useful information from rejected clients
3. **Computational Cost**: O(n²d) for n clients with d-dimensional updates
4. **No History**: Each round is independent
5. **Byzantine Threshold**: Requires knowing f beforehand

### 5.2 Trimmed Mean Defence

**Algorithm Overview**:
Trimmed Mean removes extreme values (outliers) and averages the remaining updates element-wise.

**Formal Definition**:

For parameter dimension j across n clients:

1. Collect values: {θ₁[j], θ₂[j], ..., θₙ[j]}
2. Sort values: θ₍₁₎[j] ≤ θ₍₂₎[j] ≤ ... ≤ θ₍ₙ₎[j]
3. Remove β fraction from each end:
   ```
   k = ⌊n × β⌋
   trimmed_values = {θ₍ₖ₊₁₎[j], ..., θ₍ₙ₋ₖ₎[j]}
   ```
4. Aggregate:
   ```
   θ_global[j] = mean(trimmed_values)
   ```

**Key Characteristics**:
- **Element-Wise**: Independent processing per parameter
- **Symmetric Trimming**: Removes equal amounts from both extremes
- **Fixed Strategy**: β parameter determines trimming ratio
- **No Learning**: Same trimming across all rounds
- **Statistical Robustness**: Outlier-resistant

**Limitations**:
1. **Loss of Information**: Discards potentially valid extreme values
2. **Fixed Threshold**: β doesn't adapt to attack patterns
3. **Dimension-Blind**: Doesn't consider correlations between parameters
4. **No Client Identity**: Can't track client behavior over time
5. **Inefficient Against Targeted Attacks**: Equal trimming may not match attack distribution

### 5.3 FedAvg (No Defence)

**Algorithm Overview**:
Standard Federated Averaging - weighted average of all client updates by sample count.

**Formal Definition**:

```
θ_global = Σ(nᵢ × Δθᵢ) / Σnᵢ
```

Where nᵢ is the number of samples trained by client i.

**Key Characteristics**:
- **Simplicity**: Straightforward weighted average
- **No Defence**: Fully trusts all clients
- **Efficiency**: O(nd) computational complexity
- **Unbiased**: Treats all clients equally

**Limitations**:
1. **No Byzantine Robustness**: Vulnerable to all attacks
2. **No Anomaly Detection**: Cannot identify malicious clients
3. **Linear Compromise**: Model degradation proportional to Byzantine ratio

### 5.4 Comparative Analysis

#### Aggregation Philosophy

| Strategy | Approach | Client Treatment | Decision Basis |
|----------|----------|-----------------|----------------|
| **Cognitive Defence** | Adaptive weighted averaging | Gradual trust adjustment | Historical behavior + statistics |
| **Krum** | Selection-based | Binary (accept/reject) | Geometric similarity |
| **Trimmed Mean** | Outlier removal | Partially exclude extremes | Statistical distribution |
| **FedAvg** | Equal trust | All clients equal | Sample count only |

#### Robustness Mechanisms

| Strategy | Defence Mechanism | Adaptation | Memory |
|----------|------------------|------------|--------|
| **Cognitive Defence** | Reputation + anomaly detection | Per-client, per-round | Full history (100 rounds) |
| **Krum** | Distance-based filtering | None | Stateless |
| **Trimmed Mean** | Extreme value removal | None | Stateless |
| **FedAvg** | None | None | Stateless |

#### Byzantine Fault Tolerance

| Strategy | Theoretical Guarantee | Practical Robustness | Attack Adaptability |
|----------|---------------------|---------------------|-------------------|
| **Cognitive Defence** | No formal guarantee | High (adaptive) | Excellent (learns patterns) |
| **Krum** | f < (n-2)/3 | Moderate | None (static) |
| **Trimmed Mean** | f ≤ β×n | Moderate | None (fixed β) |
| **FedAvg** | None | None | N/A |

#### Computational Complexity

For n clients, d parameters, R rounds:

| Strategy | Per-Round Complexity | Memory | Comment |
|----------|-------------------|---------|---------|
| **Cognitive Defence** | O(nd) | O(R) | Linear in parameters, stores history |
| **Krum** | O(n²d) | O(1) | Quadratic in clients (pairwise distances) |
| **Trimmed Mean** | O(nd log n) | O(1) | Sorting overhead per parameter |
| **FedAvg** | O(nd) | O(1) | Most efficient |

#### Decision Explainability

| Strategy | Explainability | Audit Trail | Reasoning Transparency |
|----------|---------------|------------|----------------------|
| **Cognitive Defence** | High | Full decision log with evidence | Detailed reasoning + confidence |
| **Krum** | Moderate | Score-based decisions | Distance scores available |
| **Trimmed Mean** | Low | Trimmed/kept labels | Statistical thresholds |
| **FedAvg** | None | N/A | No decisions made |

#### Handling Attack Types

| Attack Type | Cognitive Defence | Krum | Trimmed Mean | FedAvg |
|-------------|------------------|------|--------------|---------|
| **Label Flipping** | Detects via norm deviation | May select if majority Byzantine | Partially mitigates | Vulnerable |
| **Gradient Noise** | Detects high-variance updates | Effective if noise is large | Effective | Vulnerable |
| **Model Replacement** | Detects via large norm differences | Very effective | Effective | Vulnerable |
| **Sybil Attack** | Effective (reputation per client) | Vulnerable if >f attackers | Vulnerable if >(β×n) attackers | Vulnerable |
| **Adaptive Attack** | Can adapt over time | Vulnerable (static) | Vulnerable (static) | Vulnerable |

### 5.5 Key Innovations in Cognitive Defence

#### 1. **Temporal Context Integration**
- **Innovation**: Maintains historical update patterns for context-aware decisions
- **Advantage**: Can detect gradual attacks and temporary anomalies
- **Comparison**: Krum and Trimmed Mean are memoryless

#### 2. **Gradual Trust Adjustment**
- **Innovation**: Reputation system with decay and reward
- **Advantage**: Doesn't permanently exclude clients; allows recovery
- **Comparison**: Krum/Trimmed Mean make binary per-round decisions

#### 3. **Multi-Faceted Decision Making**
- **Innovation**: Combines OODA loop (tactical) + MAPE-K (strategic)
- **Advantage**: Both reactive (per-round) and proactive (long-term)
- **Comparison**: Other methods are purely reactive

#### 4. **Explainable AI Integration**
- **Innovation**: Every decision includes reasoning, evidence, and confidence
- **Advantage**: Auditable, debuggable, trustworthy
- **Comparison**: Other methods provide minimal decision explanation

#### 5. **Soft Aggregation**
- **Innovation**: Weight adjustment rather than hard exclusion
- **Advantage**: Retains partial information from suspicious clients
- **Comparison**: Krum discards, Trimmed Mean partially discards

#### 6. **Adaptive Learning**
- **Innovation**: Defence evolves based on observed attack patterns
- **Advantage**: Potential to learn optimal weights over time
- **Comparison**: Static defences cannot adapt

### 5.6 Trade-offs Summary

**Cognitive Defence**:
- ✅ Adaptive, explainable, soft aggregation
- ✅ Handles diverse attacks, temporal analysis
- ⚠️ Requires hyperparameter tuning (threshold, decay)
- ⚠️ Memory overhead for history storage

**Krum**:
- ✅ Strong theoretical guarantees, simple concept
- ✅ Effective against large-scale attacks
- ⚠️ High computational cost (O(n²d))
- ⚠️ High variance, data loss from rejection

**Trimmed Mean**:
- ✅ Simple, efficient O(nd log n), parameter-wise robustness
- ✅ No hyperparameter estimation required
- ⚠️ Fixed strategy, loses information
- ⚠️ May trim honest clients in high-variance scenarios

**FedAvg**:
- ✅ Most efficient, unbiased baseline
- ⚠️ No defence, vulnerable to all attacks
- ⚠️ Only suitable for trusted environments

## 6. Attack Simulation Framework

To evaluate defence effectiveness, the system includes a comprehensive attack simulation framework.

### 6.1 Supported Attacks

**Label Flipping Attack**:
- **Mechanism**: Randomly flips labels to incorrect classes
- **Intensity**: Percentage of labels to flip (e.g., 0.1 = 10%)
- **Impact**: Corrupts training data, misleads model

**Gradient Noise Attack**:
- **Mechanism**: Adds Gaussian noise to model parameters
- **Intensity**: Noise standard deviation
- **Impact**: Introduces randomness into updates

**Model Replacement Attack**:
- **Mechanism**: Replaces model with malicious parameters
- **Intensity**: Magnitude of replacement
- **Impact**: Attempts to dominate aggregation

### 6.2 Attack Configuration

Attacks are configured per-client in experiment YAML:

```yaml
attacks:
  - attack_type: "label_flip"
    intensity: 0.1
    target_clients: [0, 1, 2]
  
  - attack_type: "gradient_noise"
    intensity: 0.05
    target_clients: [7, 8]
```

## 7. Evaluation Methodology

### 7.1 Metrics

**Global Model Performance**:
- **Centralized Accuracy**: Test accuracy on clean centralized test set
- **Loss**: Cross-entropy loss on test set
- Evaluated after each federated round

**Client-Level Metrics**:
- **Training Loss**: Per-client local training loss
- **Training Accuracy**: Per-client local accuracy
- **Update Norms**: Magnitude of parameter updates

**Defence Metrics**:
- **Reputation Scores**: Client reputation over time
- **Decision Statistics**: Accept/reduce/reject counts
- **Anomaly Detection Rate**: Percentage of updates flagged

**Attack Impact**:
- **Attack Success Rate**: How often malicious updates affect model
- **Model Degradation**: Accuracy drop compared to no-attack baseline

### 7.2 Logging and Traceability

**Experiment Logs**:
- Server-level aggregation decisions
- Client-level training history
- Round-by-round metrics
- Defence reasoning and evidence

**Explainable Decision Logs**:
```json
{
  "decision": "reduce_weight",
  "confidence": 0.85,
  "reasoning": "Anomalous update detected with z-score 2.34...",
  "evidence": {
    "z_score": 2.34,
    "previous_reputation": 0.95,
    "new_reputation": 0.76
  }
}
```

## 8. Implementation Technologies

**Framework**: Flower (Federated Learning Framework)
- Server-client architecture
- Pluggable aggregation strategies
- Support for heterogeneous clients

**Deep Learning**: PyTorch
- Model definition (CNN for MNIST)
- Training and inference
- Deterministic operations

**Data Processing**: NumPy, torchvision
- Dirichlet distribution for data partitioning
- MNIST dataset handling

**Configuration**: YAML
- Experiment specifications
- Defence parameters
- Attack configurations

**Orchestration**: Python multiprocessing
- Parallel client execution
- Resource monitoring

## 9. Experimental Workflow

### 9.1 Typical Experiment

1. **Configuration**: Define experiment in YAML
2. **Initialization**: Load config, setup seeds, initialize logger
3. **Server Start**: Launch FL server with defence strategy
4. **Client Orchestration**: Spawn clients in batches
5. **Federated Training**: Execute federated rounds
6. **Logging**: Record decisions, metrics, and results
7. **Evaluation**: Assess model performance and defence effectiveness
8. **Analysis**: Review logs, reputation trends, attack impact

### 9.2 Results

Results are saved in structured format:
- `experiments/results/<experiment_name>_results.json`
- `logs/<experiment_name>.log`
- Individual client training logs

## 10. Research Contributions

### 10.1 Novel Aspects

1. **Unified Framework**: Integration of OODA loop and MAPE-K in FL context
2. **Reputation-Based FL**: Long-term client trust management
3. **Explainable Defence**: Transparent decision-making with reasoning
4. **Adaptive Aggregation**: Evolving weights based on behavior
5. **Comprehensive Comparison**: Side-by-side evaluation with established methods

### 10.2 Advantages Over Prior Work

**Compared to Krum**:
- Softer aggregation (weighted vs. selection)
- Historical awareness
- Lower computational complexity O(nd) vs. O(n²d)

**Compared to Trimmed Mean**:
- Client-aware (tracks individual behavior)
- Adaptive thresholds
- Explainable decisions

**Compared to FedAvg**:
- Byzantine robustness
- Anomaly detection
- Attack resilience

### 10.3 Future Research Directions

1. **Advanced Anomaly Detection**: Machine learning-based anomaly models
2. **Adaptive Hyperparameters**: Online tuning of thresholds and decay
3. **Differential Privacy Integration**: Combine with privacy guarantees
4. **Cross-Silo FL**: Extend to enterprise federation scenarios
5. **Theoretical Guarantees**: Formal analysis of convergence and robustness

## Conclusion

This Cognitive Defence architecture represents a significant advancement in Byzantine-robust federated learning. By combining adaptive learning, explainable decision-making, and historical context awareness, it provides a more nuanced and effective defence against adversarial attacks compared to traditional methods like Krum and Trimmed Mean. The system's modular design enables rigorous evaluation and comparison, facilitating both practical deployment and academic research.

The implementation balances theoretical soundness with practical considerations, offering a framework that is both research-oriented (for experimentation and analysis) and deployment-ready (for real-world federated scenarios). Through comprehensive logging, explainability, and deterministic execution, the system supports reproducible research and transparent operation.

---

**Authors**: Federated Learning Cognitive Defence Research Team  
**Date**: December 2024  
**Repository**: https://github.com/self1am/FL_CognitiveDefence
