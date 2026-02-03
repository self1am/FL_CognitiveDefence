# Adaptive Attacks Implementation Summary

## Overview

This implementation adds four sophisticated adaptive attack strategies to the FL_CognitiveDefence federated learning framework. These attacks learn from defense mechanism responses and adapt their strategies to evade detection while maximizing impact on the global model.

## Implemented Attacks

### 1. Statistical Optimization Attack (stat-opt)
**File**: `src/attacks/stat_opt_attack.py`

Crafts malicious updates that stay within statistical bounds of benign client updates to evade detection by statistical defenses.

**Key Features**:
- Computes mean and standard deviation of benign updates
- Constrains malicious updates to k·σ from the mean
- Adapts constraint factor based on detection feedback
- Effective against trimmed mean, Krum, and median defenses

**Parameters**:
- `intensity`: Base attack strength (0.0-1.0)
- `constraint_factor`: Multiplier for std deviation bound (default: 1.5)
- `adaptive_learning_rate`: Rate of constraint adjustment (default: 0.1)

### 2. Dynamic Optimization Attack (dny-opt)
**File**: `src/attacks/dny_opt_attack.py`

Uses reinforcement learning (Q-learning) to continuously adapt attack parameters based on real-time feedback.

**Key Features**:
- Q-learning with epsilon-greedy exploration
- Multiple attack techniques (sign flip, gradient noise, scaling)
- State discretization based on detection rate
- Reward function balancing stealth and impact

**Parameters**:
- `learning_rate`: Q-learning update rate (default: 0.1)
- `exploration_rate`: ε for exploration (default: 0.1)
- `discount_factor`: γ for future rewards (default: 0.95)
- `intensity_levels`: Discrete set of intensities to select from
- `detection_threshold`: Triggers defensive mode (default: 0.7)

### 3. Minimax Attack (min-max)
**File**: `src/attacks/min_max_attack.py`

Game-theoretic attack that finds optimal strategy assuming the defender will respond optimally.

**Key Features**:
- Considers multiple defense strategies
- Minimax optimization over defense ensemble
- Adapts threat model based on observed defenses
- Balances effectiveness across different defense types

**Parameters**:
- `defense_models`: List of defenses to consider
- `optimization_steps`: Iterations for minimax solution (default: 10)
- `threat_model_weights`: Prior probabilities over defenses

### 4. Minimum Sum Attack (min-sum)
**File**: `src/attacks/min_sum_attack.py`

Minimizes total distance to benign updates while maintaining attack effectiveness.

**Key Features**:
- Computes centroid of benign updates
- Gradient descent optimization of attack magnitude
- Balances distance minimization and attack impact
- Appears as "consensus" update to distance-based defenses

**Parameters**:
- `distance_weight`: Balance between distance and impact (default: 0.7)
- `optimization_lr`: Learning rate for optimization (default: 0.01)
- `max_iterations`: Maximum optimization steps (default: 100)
- `convergence_threshold`: Stopping criterion (default: 1e-5)

## Architecture

### Base Class: AdaptiveAttack
**File**: `src/attacks/adaptive_base.py`

Provides common functionality for all adaptive attacks:
- Feedback collection and tracking
- Detection/acceptance rate calculation
- Adaptation summary generation
- State management across rounds

**Key Methods**:
- `update_feedback()`: Records defense responses
- `adapt_strategy()`: Triggers strategy adaptation (abstract)
- `get_detection_rate()`: Computes rejection rate
- `get_adaptation_summary()`: Returns adaptation statistics

## Integration

### Client Runner Integration
**File**: `src/clients/client_runner.py`

Updated `create_attack()` function to support all four adaptive attacks with their specific parameters. Attack configurations are parsed from YAML and instantiated with appropriate settings.

### Configuration Support
**File**: `src/utils/config.py`

Extended `AttackConfig` dataclass to include all adaptive attack parameters:
- stat-opt: `constraint_factor`, `adaptive_learning_rate`
- dny-opt: `learning_rate`, `exploration_rate`, `discount_factor`
- min-max: `defense_models`, `optimization_steps`, `threat_model_weights`
- min-sum: `distance_weight`, `optimization_lr`, `max_iterations`

## Testing

### Test Suite
**File**: `test_adaptive_attacks.py`

Comprehensive test suite covering:
1. **Attack Instantiation**: Verifies all attacks can be created
2. **Parameter Modification**: Tests parameter attack functionality
3. **Feedback Mechanism**: Validates feedback collection and adaptation
4. **Benign Statistics**: Tests stat-opt and min-sum statistics tracking

**Results**: All tests passing (4/4)

### Integration Tests
**File**: `test_local_setup.py`

Updated to verify adaptive attacks import correctly alongside existing static attacks.

**Results**: All tests passing (8/8)

## Documentation

### Main Documentation
**File**: `docs/ADAPTIVE_ATTACKS.md`

Comprehensive documentation including:
- Detailed algorithm descriptions
- Mathematical formulations
- Defense evasion strategies
- Parameter explanations
- Usage examples
- Academic references

### README Updates
**File**: `README.md`

Added "Attack Strategies" section documenting:
- Static attacks (label flip, gradient noise)
- Adaptive attacks (stat-opt, dny-opt, min-max, min-sum)
- Configuration file references

## Configuration Examples

Five example configuration files provided:

1. **`stat_opt_attack_test.yaml`**: Tests stat-opt against trimmed mean
2. **`dny_opt_attack_test.yaml`**: Tests dny-opt against cognitive defense
3. **`min_max_attack_test.yaml`**: Tests min-max against Krum
4. **`min_sum_attack_test.yaml`**: Tests min-sum against Multi-Krum
5. **`all_adaptive_attacks_test.yaml`**: Tests all four attacks simultaneously

## Security Review

### Code Review Results
All issues identified in code review have been addressed:
- ✅ Fixed threat_model_weights initialization with uniform defaults
- ✅ Added defensive programming in test assertions
- ✅ Added logging for parameter size mismatches
- ✅ Documented intensity field update behavior
- ✅ Commented magic numbers with explanations

### CodeQL Analysis
**Result**: ✅ **0 security alerts found**

No vulnerabilities detected in the implementation.

## Usage Example

```python
from src.attacks import StatOptAttack, DnyOptAttack, MinMaxAttack, MinSumAttack

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
    exploration_rate=0.1
)

# Minimax Attack
minmax_attack = MinMaxAttack(
    intensity=0.2,
    defense_models=['krum', 'trimmed_mean', 'cognitive']
)

# Minimum Sum Attack
minsum_attack = MinSumAttack(
    intensity=0.2,
    distance_weight=0.7
)
```

## YAML Configuration Example

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
    exploration_rate: 0.1
    target_clients: [3, 4]
```

## Key Implementation Details

### Feedback Mechanism
All adaptive attacks inherit from `AdaptiveAttack` which provides:
- Round-by-round feedback tracking
- Detection rate computation
- Acceptance rate monitoring
- Adaptation trigger mechanism

### Benign Statistics
Two attacks require knowledge of benign client updates:
- **stat-opt**: Uses `update_benign_statistics()` to track mean/std
- **min-sum**: Uses `update_benign_estimates()` to compute centroid

These methods should be called server-side with benign client parameters.

### State Management
Each attack maintains internal state:
- **stat-opt**: `constraint_factor`, `benign_stats`
- **dny-opt**: `q_table`, `current_technique`, Q-learning state
- **min-max**: `threat_model_weights`, `observed_defenses`
- **min-sum**: `benign_centroid`, `optimized_magnitude`

## Academic References

1. Fang et al. (2020) - "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning" (USENIX Security)
2. Baruch et al. (2019) - "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS)
3. Shejwalkar & Houmansadr (2021) - "Manipulating the Byzantine" (NDSS)
4. Bhagoji et al. (2019) - "Analyzing Federated Learning through an Adversarial Lens" (ICML)
5. Cao et al. (2021) - "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping" (NDSS)

## Files Modified/Created

### New Files (11)
- `src/attacks/adaptive_base.py` - Base class for adaptive attacks
- `src/attacks/stat_opt_attack.py` - Statistical optimization attack
- `src/attacks/dny_opt_attack.py` - Dynamic optimization attack
- `src/attacks/min_max_attack.py` - Minimax attack
- `src/attacks/min_sum_attack.py` - Minimum sum attack
- `docs/ADAPTIVE_ATTACKS.md` - Comprehensive documentation
- `test_adaptive_attacks.py` - Test suite
- `experiments/configs/stat_opt_attack_test.yaml` - Config example
- `experiments/configs/dny_opt_attack_test.yaml` - Config example
- `experiments/configs/min_max_attack_test.yaml` - Config example
- `experiments/configs/min_sum_attack_test.yaml` - Config example
- `experiments/configs/all_adaptive_attacks_test.yaml` - Config example

### Modified Files (5)
- `src/attacks/__init__.py` - Export new attacks
- `src/clients/client_runner.py` - Support attack loading
- `src/utils/config.py` - Extended AttackConfig
- `test_local_setup.py` - Added import test
- `README.md` - Documentation updates

## Summary

This implementation provides a complete suite of adaptive attacks for evaluating the robustness of federated learning defenses. The attacks are:
- **Well-documented** with academic references
- **Thoroughly tested** with comprehensive test coverage
- **Properly integrated** into the existing framework
- **Secure** with no vulnerabilities detected
- **Ready for use** in defense evaluation experiments

The implementation follows best practices and maintains consistency with the existing codebase architecture.
