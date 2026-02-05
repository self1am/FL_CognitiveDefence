# Production-Scale Experiment Guide (100 Clients, 30-50 Rounds)

## Hardware Requirements
- **CPU**: 8 vCPU (yours: 8 vCPU ✓)
- **Memory**: 64GB (yours: 64GB ✓)
- **Disk**: 100GB+ (for datasets and logs)
- **Network**: Stable connection for FL communication

## Resource Breakdown for 100 Clients

### Memory Usage Estimation
```
Base System:           ~2-3 GB
Python + PyTorch:      ~3-4 GB
Server Process:        ~2-3 GB
10 Concurrent Clients: ~40-45 GB (4-4.5 GB per client)
Data Loading:          ~5 GB (MNIST)
─────────────────────────────
Total:                ~55-60 GB
```

### CPU Usage Estimation
```
Server Aggregation:    ~1-2 vCPU (mostly idle between rounds)
8 Concurrent Clients:  ~6-7 vCPU (during training)
System:                ~0.5 vCPU
─────────────────────────────
Peak:                 ~8 vCPU (fully utilized)
```

### Timeline Estimation (40 rounds, 100 clients)
```
Per-round time: ~4-6 minutes
  - Client training:    ~3-4 min (8 clients × 2 epochs)
  - Communication:      ~30 sec (aggregation + upload)
  - Server evaluation:  ~20 sec (centralized test set)
  
Total duration: 40 rounds × 5 min/round ≈ 3.3 hours
Actual with setup:     ~4 hours
```

---

## Step-by-Step Execution Plan

### **Phase 1: Pre-Experiment Setup (30 minutes)**

#### 1.1 SSH into GCP Instance
```bash
# On your local machine
gcloud compute ssh your-instance-name --zone=your-zone
# Or direct SSH
ssh user@your.gcp.instance.ip
```

#### 1.2 Clone Repository
```bash
cd /tmp  # Or your preferred directory
git clone <your-repo-url>
cd FL_CognitiveDefence
```

#### 1.3 Setup Environment
```bash
# Create virtual environment (RECOMMENDED)
python3 -m venv fl_env
source fl_env/bin/activate

# Install dependencies
pip install --upgrade pip
make install
# This runs: pip install -r requirements.txt && pip install -e .
```

#### 1.4 Verify CUDA/GPU (if available)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

#### 1.5 Download MNIST Dataset
```bash
python -c "
from src.datasets.mnist_handler import MNISTDataHandler
handler = MNISTDataHandler(batch_size=32)
print('MNIST dataset downloaded successfully')
"
```

---

### **Phase 2: Run Production Experiments (4-8 hours)**

#### 2.1 Single Experiment (4 hours)
```bash
# Option A: Cognitive Defence with 100 clients
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml

# Option B: Adaptive Attacks with 100 clients (longer - 6-8 hours)
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_adaptive.yaml
```

#### 2.2 Recommended Sequence (15 hours total)
Run these sequentially to compare defences:

```bash
# Experiment 1: Cognitive Defence (4 hours)
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml

# Experiment 2: Krum Defence (4 hours) - requires config file edit
# Edit experiments/configs/production_100_clients_multidefence.yaml
# Set: defence.strategy: "krum"
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_multidefence.yaml

# Experiment 3: Adaptive Attacks (6-8 hours) - more comprehensive
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_adaptive.yaml
```

---

### **Phase 3: Monitoring During Execution**

#### 3.1 In a separate terminal, monitor system resources:
```bash
# Real-time monitoring
watch -n 2 'echo "=== CPU ==="; top -bn1 | head -12; echo "=== Memory ==="; free -h; echo "=== Processes ==="; ps aux | grep python | wc -l'

# Or use better tool if available
htop

# Or check specific metrics
while true; do
  clear
  echo "=== System Resources ==="
  echo "Timestamp: $(date)"
  echo ""
  echo "Memory Usage:"
  free -h | grep Mem
  echo ""
  echo "CPU Usage:"
  uptime
  echo ""
  echo "Python Processes:"
  ps aux | grep "python" | grep -v grep | wc -l
  echo ""
  echo "Disk Usage:"
  df -h / | tail -1
  sleep 5
done
```

#### 3.2 Monitor experiment logs in real-time:
```bash
# Terminal for logs
tail -f logs/<experiment_name>_complete.json | jq '.' 2>/dev/null || tail -f logs/<experiment_name>_complete.json
```

#### 3.3 Kill experiment if needed (graceful):
```bash
# Let it finish current round
pkill -f "experiment_runner"

# Force kill after 10 seconds
pkill -9 -f "experiment_runner"
```

---

### **Phase 4: Post-Experiment Analysis (30 minutes - 1 hour)**

#### 4.1 Analyze Results
```bash
# Visualize experiment results
python experiments/visualize_results.py \
  --config experiments/configs/production_100_clients_cognitive.yaml

# Analyze attack impact
python analyze_attack_impact.py

# Analyze server logs
python analyze_server_logs.py
```

#### 4.2 Generate Summary Report
```bash
# Create comparison results
python -c "
import json
import glob
from pathlib import Path

results_dir = Path('logs')
experiments = {}

for log_file in results_dir.glob('*_complete.json'):
    with open(log_file) as f:
        data = json.load(f)
        exp_name = log_file.stem.replace('_complete', '')
        experiments[exp_name] = {
            'rounds': len(data.get('centralized_accuracy', [])),
            'final_accuracy': data.get('centralized_accuracy', [])[-1] if data.get('centralized_accuracy') else None,
            'final_loss': data.get('centralized_loss', [])[-1] if data.get('centralized_loss') else None,
        }

print('=' * 80)
print('EXPERIMENT SUMMARY')
print('=' * 80)
for exp, results in experiments.items():
    print(f'{exp}:')
    print(f'  Final Accuracy: {results[\"final_accuracy\"]:.4f}')
    print(f'  Final Loss: {results[\"final_loss\"]:.4f}')
    print()
"
```

#### 4.3 Archive Results
```bash
# Create backup
tar -czf experiment_results_$(date +%Y%m%d_%H%M%S).tar.gz logs/ experiments/results/

# Upload to cloud storage (if available)
gsutil -m cp experiment_results_*.tar.gz gs://your-bucket/
```

---

## Configuration Parameters Guide

### Key Parameters for 100-Client Experiments

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_clients` | 100 | Full-scale production test |
| `num_rounds` | 40-50 | Sufficient for convergence |
| `batch_size` (orchestration) | 8 | 8 concurrent clients = ~40GB memory |
| `max_memory_mb` | 58000 | Leave 6GB for OS/buffers |
| `min_clients` | 80 | Tolerate up to 20% failures |
| `anomaly_threshold` | 0.60-0.65 | Balanced detection |
| `reputation_decay` | 0.75-0.80 | Longer client memory |
| `history_size` | 200-250 | More historical data for 100 clients |

### Attack Configuration Examples

**35% Attack Rate (35 clients)**
```yaml
attacks:
  - attack_type: "label_flip"
    intensity: 0.15
    target_clients: [0..14]  # 15 clients
  
  - attack_type: "gradient_noise"
    intensity: 0.12
    target_clients: [15..34]  # 20 clients
```

**20% Attack Rate (20 clients)**
```yaml
attacks:
  - attack_type: "label_flip"
    intensity: 0.15
    target_clients: [0..19]  # 20 clients only
```

---

## Troubleshooting

### Issue: "Not enough memory"
**Solution:**
```bash
# Reduce batch_size in config
orchestration:
  batch_size: 6  # Instead of 8

# Or reduce number of clients
experiment:
  num_clients: 75  # Instead of 100
```

### Issue: "Clients timeout or disconnect"
**Solution:**
```bash
# Increase client timeout
orchestration:
  client_timeout_seconds: 2400  # 40 minutes

# Reduce spawn_delay to let clients complete faster
orchestration:
  spawn_delay: 1.0  # Instead of 2.0
```

### Issue: "Server evaluation is slow"
**Solution:**
```bash
# Reduce evaluation batch size in code or skip some rounds
# Edit run_server_with_eval.py to reduce test set size
```

### Issue: "Disk space running out"
```bash
# Check disk usage
du -sh logs/ experiments/

# Clean old logs
rm -rf logs/*_complete.json  # Keep only recent

# Compress old results
gzip logs/*.json
```

---

## Expected Results Benchmark

### Cognitive Defence vs No Defence (100 clients, 40 rounds)

| Metric | No Defence | With Cognitive Defence |
|--------|-----------|----------------------|
| Final Accuracy (Clean) | 65-70% | 92-96% |
| Final Loss (Clean) | 1.2-1.5 | 0.08-0.12 |
| Attack Resilience | 0% | 90%+ |
| Detected Anomalies | N/A | 35-38 clients |

---

## Best Practices

1. **Run sequentially, not in parallel**
   - Each experiment uses ~60GB → can't run 2 simultaneously
   - Allow 30 min cooldown between experiments

2. **Monitor memory continuously**
   - Use `watch` command to catch issues early
   - Set alert at 90% memory usage

3. **Save logs immediately**
   - Upload results to cloud storage after each experiment
   - Don't rely on local disk (GCP instances can be deleted)

4. **Use detached sessions (screen/tmux)**
   ```bash
   # Start experiment in background
   screen -S experiment_1
   # Ctrl+A then D to detach
   
   # Reattach later
   screen -r experiment_1
   ```

5. **Document configurations used**
   ```bash
   # Save config with results
   cp experiments/configs/production_100_clients_cognitive.yaml \
      experiments/results/config_$(date +%Y%m%d_%H%M%S).yaml
   ```

---

## Example: Complete 24-Hour Experiment Campaign

```bash
#!/bin/bash
# Run all production experiments in sequence

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/$TIMESTAMP"
mkdir -p $LOG_DIR

echo "Starting production experiment campaign at $(date)" | tee $LOG_DIR/campaign.log

# Experiment 1: Baseline (no attacks)
echo "Starting Baseline..." | tee -a $LOG_DIR/campaign.log
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml 2>&1 | tee -a $LOG_DIR/experiment1.log
echo "Experiment 1 completed at $(date)" | tee -a $LOG_DIR/campaign.log
sleep 300  # 5 min cooldown

# Experiment 2: Adaptive Attacks
echo "Starting Adaptive Attacks..." | tee -a $LOG_DIR/campaign.log
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_adaptive.yaml 2>&1 | tee -a $LOG_DIR/experiment2.log
echo "Experiment 2 completed at $(date)" | tee -a $LOG_DIR/campaign.log

echo "All experiments completed at $(date)" | tee -a $LOG_DIR/campaign.log
```

---

## Performance Optimization Tips

1. **CPU Affinity**: Pin Python processes to specific cores
   ```bash
   taskset -c 0-7 python -m src.orchestration.experiment_runner ...
   ```

2. **Memory Prefetching**: Use `MALLOC_MMAP_THRESHOLD_`
   ```bash
   export MALLOC_MMAP_THRESHOLD_=131072
   ```

3. **Disable Swap** (for predictable performance)
   ```bash
   sudo swapoff -a  # Careful! Only if memory is sufficient
   ```

4. **Increase File Descriptors**
   ```bash
   ulimit -n 4096  # For many client connections
   ```

