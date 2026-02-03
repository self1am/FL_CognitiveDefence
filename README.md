# Federated Learning with Cognitive defence Mechanisms

A modular federated learning framework implementing cognitive defence strategies based on OODA loop and MAPE-K frameworks, with support for various attacks and defences.

## Features

- **Modular Architecture**: Separation of concerns with pluggable attacks and defences
- **Multi-Client Orchestration**: Automated management of 10+ client processes with resource monitoring
- **Multiple Defence Strategies**: 
  - **Cognitive Defence**: OODA loop and MAPE-K framework implementation
  - **Krum**: Byzantine-robust aggregation selecting updates with minimal distance scores
  - **Trimmed Mean**: Robust aggregation removing outliers from both ends
- **Attack Simulation**: 
  - **Static Attacks**: Label flipping, gradient noise
  - **Adaptive Attacks**: stat-opt, dny-opt, min-max, min-sum (see [docs/ADAPTIVE_ATTACKS.md](docs/ADAPTIVE_ATTACKS.md))
- **Explainable AI**: Decision logging with reasoning and evidence
- **Deterministic Experiments**: Reproducible results with proper seeding

## Quick Start

### 1. Setup
```bash
# Clone and setup
git clone https://github.com/self1am/FL_CognitiveDefence.git
cd FL_CognitiveDefence
make setup
```

### 2. Run Basic Experiment
```bash
make run-basic
```

### 3. Run Custom Experiment
```bash
python -m src.orchestration.experiment_runner --config experiments/configs/your_config.yaml
```

## Defence Strategies

### Cognitive Defence
OODA loop-based adaptive defence with reputation system:
- Observes client update patterns
- Orients using historical context
- Decides on weighted aggregation
- Acts with explainable decisions

### Krum
Byzantine-robust aggregation from literature (Blanchard et al., NeurIPS 2017):
- Selects client update(s) with smallest distance score
- Robust against up to f Byzantine clients
- Supports both single-Krum and Multi-Krum variants

### Trimmed Mean
Byzantine-robust aggregation (Yin et al., ICML 2018):
- Removes β fraction of extreme values per dimension
- Aggregates remaining values
- Parameter-wise robustness

## Project Structure

```
federated-cognitive-defence/
├── src/
│   ├── attacks/          # Attack implementations
│   ├── defences/         # defence strategies
│   ├── clients/          # Client implementations
│   ├── server/           # Server implementations
│   ├── models/           # Neural network models
│   ├── datasets/         # Dataset handlers
│   ├── orchestration/    # Multi-client orchestration
│   └── utils/            # Utilities and configuration
├── experiments/
│   ├── configs/          # Experiment configurations
│   ├── scripts/          # Helper scripts
│   └── results/          # Experiment results
└── tests/                # Unit and integration tests
```

## Configuration

Experiments are configured using YAML files. Example configurations:

### Cognitive Defence
```yaml
experiment:
  experiment_name: "cognitive_defence_test"
  seed: 42
  num_rounds: 10
  server_address: "0.0.0.0:8080"

defence:
  strategy: "cognitive_defence"
  anomaly_threshold: 0.7
  reputation_decay: 0.8

attacks:
  - enabled: true
    attack_type: "label_flip"
    intensity: 0.1
    target_clients: [0, 1, 2]

orchestration:
  num_clients: 10
  batch_size: 3
```

### Krum Defence
```yaml
defence:
  strategy: "krum"
  num_byzantine: 2      # Expected number of malicious clients
  multi_krum: false     # Use standard Krum (true for Multi-Krum)
```

### Trimmed Mean Defence
```yaml
defence:
  strategy: "trimmed_mean"
  beta: 0.2            # Trim 20% from each end
```

## Hardware Requirements

- **MacBook M1 (8GB)**: Orchestrator + 2-3 lightweight clients
- **Azure VMs (4GB each)**: Server + 3-4 clients each
- **Total**: Support for 10+ concurrent clients

## Distributed Setup

For multi-machine experiments:

```bash
# Setup distributed environment
./scripts/run_distributed.sh

# Monitor progress
tail -f logs/experiment_name.log
```

## Development

```bash
# Install development dependencies
make dev-install

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

## Experiment Results

Results are automatically saved to:
- `experiments/results/`: JSON experiment summaries
- `logs/`: Detailed execution logs
- Individual client logs with training history

## Attack Strategies

### Static Attacks
- **Label Flipping**: Randomly flips labels during training to corrupt the model
- **Gradient Noise**: Adds Gaussian/uniform noise to gradient updates

### Adaptive Attacks
Advanced attacks that learn from defense responses. See [docs/ADAPTIVE_ATTACKS.md](docs/ADAPTIVE_ATTACKS.md) for detailed documentation.

1. **stat-opt (Statistical Optimization)**: Crafts updates that stay within statistical bounds of benign clients to evade detection
2. **dny-opt (Dynamic Optimization)**: Uses reinforcement learning to adapt attack parameters based on real-time feedback
3. **min-max (Minimax)**: Game-theoretic attack that assumes optimal defender response
4. **min-sum (Minimum Sum)**: Minimizes total distance to benign updates while maintaining attack impact

Example configurations available in `experiments/configs/`:
- `stat_opt_attack_test.yaml`
- `dny_opt_attack_test.yaml`
- `min_max_attack_test.yaml`
- `min_sum_attack_test.yaml`
- `all_adaptive_attacks_test.yaml`

## Next Steps

1. **FEMNIST Integration**: More realistic FL dataset
2. **Quantum Neural Networks**: PennyLane integration
3. **Advanced defences**: FreqFed, Median, FoolsGold
4. ~~**Adaptive Attacks**: Learning-based adversarial strategies~~ ✓ Completed
5. **Comparative Analysis**: Benchmark defences against various attack scenarios

## Contributing

1. Create feature branches for new functionality
2. Add tests for new components
3. Update documentation
4. Submit pull requests

## License

MIT License - see [LICENSE](LICENSE) file for details.
