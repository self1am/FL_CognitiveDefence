# Experiment Configuration Summary

## Overview
All 8 experiment configurations are now properly configured and ready to run with the simulation framework.

## Configuration Files Created

### 1. Static Attacks - No Defense
**File**: `static_attacks_no_defence.yaml`
- **Attack**: Label Flip (static)
- **Defense**: None (Simple FedAvg)
- **Target Clients**: [0-9] (10 malicious clients)
- **Attack Intensity**: 0.5

### 2. Static Attacks - Horizontal Defense
**File**: `static_attacks_horizontal_defence.yaml`
- **Attack**: Label Flip (static)
- **Defense**: Horizontal (Aggregation-based)
- **Anomaly Threshold**: 0.85
- **Target Clients**: [0-9]
- **Attack Intensity**: 0.5

### 3. Static Attacks - Vertical Defense
**File**: `static_attacks_vertical_defence.yaml`
- **Attack**: Label Flip (static)
- **Defense**: Vertical (Differential Privacy)
- **Anomaly Threshold**: 0.80
- **Target Clients**: [0-9]
- **Attack Intensity**: 0.5

### 4. Adaptive Attacks - No Defense
**File**: `adaptive_attacks_no_defence.yaml`
- **Attack**: Stat-Opt (adaptive statistical optimization)
- **Defense**: None
- **Target Clients**: [0-9]
- **Attack Intensity**: 0.5
- **Constraint Factor**: 1.5

### 5. Adaptive Attacks - Horizontal Defense
**File**: `adaptive_attacks_horizontal_defence.yaml`
- **Attack**: Dny-Opt (dynamic optimization)
- **Defense**: Horizontal
- **Target Clients**: [0-9]
- **Attack Intensity**: 0.5
- **Learning Rate**: 0.1
- **Exploration Rate**: 0.1

### 6. Adaptive Attacks - Vertical Defense
**File**: `adaptive_attacks_vertical_defence.yaml`
- **Attack**: Min-Max (game-theoretic)
- **Defense**: Vertical
- **Target Clients**: [0-9]
- **Attack Intensity**: 0.5
- **Optimization Steps**: 10

### 7. Static Attacks - Cognitive Defense
**File**: `static_attacks_cognitive_defence.yaml`
- **Attack**: Label Flip (static)
- **Defense**: Cognitive Defense (Multi-parameter anomaly detection)
- **Anomaly Threshold**: 0.75
- **Target Clients**: [0-9]
- **Attack Intensity**: 0.5

### 8. Adaptive Attacks - Cognitive Defense
**File**: `adaptive_attacks_cognitive_defence.yaml`
- **Attack**: Min-Sum (distance minimization)
- **Defense**: Cognitive Defense
- **Anomaly Threshold**: 0.75
- **Target Clients**: [0-9]
- **Attack Intensity**: 0.5

## Common Parameters Across All Configs

### Experiment Settings
- **Num Rounds**: 10
- **Num Clients**: 100
- **Min Clients**: 20
- **Seed**: 123 (optional - for reproducibility)

### Orchestration
- **Batch Size**: 2
- **Max Memory**: 6000 MB
- **Epochs**: 1
- **Batch Size (Client)**: 64
- **Spawn Delay**: 2.0s

### Simulation
- **CPU per Client**: 0.25
- **Total CPUs**: 8
- **Ray Dashboard**: Disabled
- **Logging to Driver**: Disabled

### Evaluation
- **Test Samples**: 5000

## Attack Types Used

1. **Label Flip** (Static): Simple label corruption attack
   - Source: Converts specific class labels to target class
   - Difficulty: Easy to detect but effective baseline

2. **Stat-Opt** (Adaptive): Statistical optimization attack
   - Uses constraint factors to craft updates
   - Adaptive learning rate adjustment
   - More evasive than simple attacks

3. **Dny-Opt** (Adaptive): Dynamic optimization
   - Q-learning based strategy selection
   - Learns effective attack patterns over rounds
   - Exploration-exploitation tradeoff

4. **Min-Max** (Adaptive): Game-theoretic attack
   - Optimizes against multiple defense strategies
   - Assumes defender responds optimally
   - Highly sophisticated

5. **Min-Sum** (Adaptive): Distance minimization
   - Minimizes sum of distances to benign updates
   - Evades distance-based defenses (Krum, Multi-Krum)
   - Appears as consensus

## Defense Strategies

1. **None**: Baseline FedAvg (no special defenses)
2. **Horizontal**: Aggregation-based defenses (Krum, Trimmed Mean)
3. **Vertical**: Differential Privacy-based defenses
4. **Cognitive**: Multi-parameter anomaly detection with reputation scoring

## Usage

Run any configuration with:
```bash
python -m src.orchestration.simulation_runner --config experiments/configs/<config_name>.yaml
```

Example:
```bash
python -m src.orchestration.simulation_runner --config experiments/configs/static_attacks_cognitive_defence.yaml
```

## Expected Behavior

- **Static + No Defense**: Should show significant accuracy degradation
- **Static + Horizontal**: Should show partial recovery (defense effectiveness)
- **Static + Vertical**: Should show privacy-utility tradeoff
- **Adaptive + No Defense**: Should show accelerating attacks over rounds
- **Adaptive + Horizontal**: Moderate defense against adaptive attacks
- **Adaptive + Vertical**: Privacy preservation at cost of utility
- **Static + Cognitive**: Cognitive defense with pattern learning
- **Adaptive + Cognitive**: Most robust against evolving attacks

## Key Features

✓ All configs use valid field names from AttackConfig and defenceConfig
✓ All attack types are supported by the framework
✓ All defense strategies are implemented
✓ Consistent parameter structure across all configs
✓ 10 target clients for realistic attack scenarios
✓ Ready for comparative analysis

