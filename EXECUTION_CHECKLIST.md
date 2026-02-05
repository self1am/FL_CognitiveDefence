# Production FL Experiments - Complete Execution Guide

## 📋 Complete Checklist & Steps

### Pre-Experiment Phase (Do Once)

#### ✅ GCP Instance Setup
- [ ] GCP instance created with specifications:
  - [ ] 64 GB RAM
  - [ ] 8 vCPU (4-core, 8vCPU)
  - [ ] ~100GB disk space
  - [ ] Ubuntu 20.04 or later
- [ ] SSH access configured and tested
- [ ] Static IP assigned (optional but recommended)

#### ✅ Project Setup on GCP Instance
```bash
# 1. SSH into instance
gcloud compute ssh your-instance-name --zone=your-zone

# 2. Clone project
git clone <your-repository-url>
cd FL_CognitiveDefence

# 3. Run optimization script
chmod +x scripts/*.sh
./scripts/optimize_gcp_instance.sh

# 4. Create and activate virtual environment
python3 -m venv fl_env
source fl_env/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 6. Verify installation
python -c "import torch, flwr, numpy; print('✓ All dependencies installed')"

# 7. Download MNIST dataset
python -c "
from src.datasets.mnist_handler import MNISTDataHandler
handler = MNISTDataHandler(batch_size=32)
print('✓ MNIST dataset ready')
"
```

#### ✅ Load Optimization Profile (Every Session)
```bash
# Add to ~/.bashrc or ~/.zshrc for persistence
source ~/.fl_optimization.sh

# Or run manually each session
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NUM_THREADS=8
export MALLOC_MMAP_THRESHOLD_=131072
```

---

## 🎯 Experiment Execution Phase

### Step 1: Start Base Session (Terminal 1)
```bash
# SSH into GCP instance
gcloud compute ssh your-instance-name --zone=your-zone

# Navigate to project
cd FL_CognitiveDefence

# Activate environment
source ~/.fl_optimization.sh
source fl_env/bin/activate
```

### Step 2: Start Resource Monitoring (Terminal 2 - Optional but Recommended)
```bash
# SSH into same instance
gcloud compute ssh your-instance-name --zone=your-zone
cd FL_CognitiveDefence

# Start monitoring
./scripts/run_production_experiments.sh --monitor

# Or use watch command
watch -n 2 'echo "=== Memory ==="; free -h | grep Mem; echo "=== CPU ==="; uptime; echo "=== Python ==="; ps aux | grep python | wc -l'
```

### Step 3: Run Experiments (Terminal 1)

#### Option A: Single Experiment
```bash
# Cognitive Defence (4 hours)
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml

# Or Adaptive Attacks (6-8 hours)
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_adaptive.yaml
```

#### Option B: Automated All Experiments
```bash
# Runs all experiments in sequence with cleanup
./scripts/run_production_experiments.sh --all
```

#### Option C: Custom Configuration
```bash
# Edit a config file first
nano experiments/configs/production_100_clients_cognitive.yaml

# Then run it
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml
```

---

## 📊 Expected Timeline

### For Single 100-Client Experiment (40 rounds)

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup & Startup | 2-5 min | Server starts, clients connect |
| Rounds 1-10 | 40-60 min | ~4-6 min per round |
| Rounds 11-20 | 40-60 min | Steady pace |
| Rounds 21-30 | 40-60 min | Model converging |
| Rounds 31-40 | 40-60 min | Final convergence |
| Shutdown & Logging | 5-10 min | Results saved |
| **Total** | **~4 hours** | Can vary ±30 min |

### For Adaptive Attacks (50 rounds)
- **Expected Duration**: 5-6 hours
- **Max Duration**: 8 hours (with system load)

### For Full Campaign (Cognitive + Adaptive + Comparison)
- **Total Time**: 15-20 hours
- **Best Approach**: Run overnight or over several days

---

## 🔍 Real-Time Monitoring During Execution

### View Accuracy/Loss Curves (Terminal 3)
```bash
# Watch experiment results in real-time
watch -n 10 'tail -50 logs/*_complete.json | grep "centralized" | tail -5'

# Or use jq for pretty output
watch -n 10 'tail -100 logs/*_complete.json | jq ".centralized_accuracy[-5:]" 2>/dev/null || echo "waiting..."'
```

### Check Client Status
```bash
# See how many clients are running
watch -n 5 'ps aux | grep "client_runner\|client_orchestrator" | grep -v grep | wc -l'
```

### Monitor Logs
```bash
# Watch for error messages
tail -f logs/*_complete.json | grep -i "error\|failed\|anomal" || true
```

---

## ✅ Post-Experiment Phase

### After Each Experiment Completes

```bash
# 1. Wait for completion message
# Look for: "Experiment completed successfully!"

# 2. Save timestamp of completion
date >> completion_log.txt

# 3. Backup results immediately
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz logs/ experiments/results/

# 4. Upload to cloud storage (if configured)
gsutil -m cp results_*.tar.gz gs://your-bucket/fl-results/
```

### Analyze Results

```bash
# Generate analysis report
python analyze_experiments.py

# View CSV results
cat experiment_analysis.csv

# View detailed report
cat experiment_analysis_report.txt

# Visualize results
python experiments/visualize_results.py \
  --config experiments/configs/production_100_clients_cognitive.yaml
```

### Generate Comparison Report
```bash
# After running multiple experiments
python -c "
import json
from pathlib import Path

print('\\n' + '='*60)
print('EXPERIMENT COMPARISON')
print('='*60)

for log_file in Path('logs').glob('*_complete.json'):
    with open(log_file) as f:
        data = json.load(f)
        acc = data.get('centralized_accuracy', [])
        loss = data.get('centralized_loss', [])
        
        exp_name = log_file.stem.replace('_complete', '')
        
        if acc and loss:
            print(f'{exp_name}:')
            print(f'  Final Accuracy: {acc[-1]:.4f}')
            print(f'  Final Loss:     {loss[-1]:.6f}')
            print(f'  Improvement:    +{acc[-1] - acc[0]:.4f}')
            print()
"
```

---

## 🚨 Troubleshooting & Common Issues

### Issue: "CUDA out of memory" or "OOM Killer triggered"

**Solution:**
```bash
# Edit the config and reduce batch size
# In experiments/configs/production_100_clients_cognitive.yaml:

orchestration:
  batch_size: 6        # Reduce from 8 to 6
  max_memory_mb: 48000 # Reduce from 58000 to 48000
  num_clients: 75      # Or reduce from 100 to 75

# Then try again
python -m src.orchestration.experiment_runner --config ...
```

### Issue: "Clients disconnected" or "Cannot connect to server"

**Solution:**
```bash
# Increase timeout in config
orchestration:
  client_timeout_seconds: 2400  # Increase from 1800 to 2400

# Reduce spawn rate
orchestration:
  spawn_delay: 3.0  # Increase from 2.0 to 3.0

# Kill any stuck processes and retry
pkill -9 -f "python.*client"
sleep 10
# Run experiment again
```

### Issue: "Disk space low" or "No space left on device"

**Solution:**
```bash
# Check disk usage
df -h /

# Clean old logs
rm -rf logs/*_complete.json  # Keep only recent experiments

# Or compress old ones
gzip logs/*.json

# Delete very old logs
find logs -name "*.json" -mtime +7 -delete  # Delete files older than 7 days
```

### Issue: Experiment stops without error

**Solution:**
```bash
# Check system resources
free -h  # Check memory
ps aux | grep python  # Check processes

# Kill stuck processes
pkill -f "experiment_runner"
sleep 10

# Retry (should recover or start fresh)
python -m src.orchestration.experiment_runner --config ...
```

### Issue: "Network connection reset" or timeout

**Solution:**
```bash
# Check GCP instance network status
gcloud compute instances describe your-instance-name

# Reduce network load
orchestration:
  batch_size: 4  # Fewer concurrent clients

# Restart networking
sudo systemctl restart networking

# Retry experiment
```

### Issue: Results look wrong (accuracy too low, loss too high)

**Possible Causes & Solutions:**
```
1. Model not training:
   - Check batch size is not 0
   - Verify learning rate is reasonable (0.001)
   
2. Attacks too strong:
   - Reduce attack intensity: 0.15 → 0.10
   - Reduce number of attacking clients
   
3. Defence threshold wrong:
   - For Cognitive: increase anomaly_threshold from 0.65 to 0.75
   - For Krum: increase num_byzantine tolerance
   
4. Data distribution wrong:
   - Check alpha parameter (0.5 = IID)
   - Try alpha: 0.1 for non-IID

# Edit config and retry
```

---

## 📈 Performance Benchmarks & Expectations

### Hardware Utilization (100 clients, batch size 8)

| Resource | During Idle | During Training | Peak Usage |
|----------|------------|-----------------|-----------|
| Memory | 5GB | 40-50GB | 58GB |
| CPU | 0.5vCPU | 2-3vCPU | 7-8vCPU |
| Disk I/O | Minimal | 100-200 MB/s | 300 MB/s |
| Network | <1 Mbps | 10-50 Mbps | 100 Mbps |

### Model Performance Targets

| Experiment Type | Final Accuracy | Final Loss | Rounds |
|-----------------|----------------|-----------|--------|
| Baseline (no attack) | 97-99% | 0.05-0.08 | 40 |
| With Cognitive Defence | 92-96% | 0.08-0.12 | 40 |
| Attack Only (no defence) | 65-75% | 1.0-1.5 | 40 |
| Adaptive Attacks | 88-94% | 0.10-0.15 | 50 |

---

## 📚 Documentation Reference

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick commands & cheatsheet | Before running experiments |
| [PRODUCTION_EXPERIMENT_GUIDE.md](PRODUCTION_EXPERIMENT_GUIDE.md) | Detailed guide & explanation | For deep understanding |
| [README.md](README.md) | Project overview | Project introduction |
| [docs/ADAPTIVE_ATTACKS.md](docs/ADAPTIVE_ATTACKS.md) | Attack details | Understanding attack types |
| [CENTRALIZED_EVAL_GUIDE.md](CENTRALIZED_EVAL_GUIDE.md) | Evaluation metrics | Understanding results |
| [ANOMALY_SCORING_EXPLAINED.md](ANOMALY_SCORING_EXPLAINED.md) | Cognitive defence details | Understanding defence mechanism |

---

## 💡 Pro Tips & Best Practices

### 1. Use GNU Screen for Persistent Sessions
```bash
# Start screen
screen -S my_experiment

# Run experiment
python -m src.orchestration.experiment_runner --config ...

# Detach (Ctrl+A then D)

# Later, reattach
screen -r my_experiment

# Kill session
screen -X -S my_experiment quit
```

### 2. Run Multiple Experiments in Sequence
```bash
#!/bin/bash
# save as run_all.sh

configs=(
  "production_100_clients_cognitive.yaml"
  "production_100_clients_adaptive.yaml"
)

for config in "${configs[@]}"; do
  echo "Running $config..."
  python -m src.orchestration.experiment_runner \
    --config experiments/configs/$config
  
  # Wait 5 minutes between experiments
  echo "Cooling down..."
  sleep 300
done

echo "All experiments completed!"
```

### 3. Monitor Multiple Metrics
```bash
# Create monitoring dashboard
watch -n 2 'clear; \
  echo "=== System ==="; \
  free -h | grep Mem; \
  uptime; \
  echo "=== Processes ==="; \
  ps aux | grep python | wc -l; \
  echo "=== Disk ==="; \
  df -h / | tail -1'
```

### 4. Automated Backup to Cloud
```bash
# Add to crontab (runs hourly)
0 * * * * cd /path/to/FL_CognitiveDefence && \
  tar -czf backup_$(date +\%Y\%m\%d_\%H\%M\%S).tar.gz logs/ && \
  gsutil cp backup_*.tar.gz gs://your-bucket/fl-backups/ && \
  rm backup_*.tar.gz

# Enable crontab
crontab -e  # Add above line
```

### 5. Email Notifications on Completion
```bash
# Add to end of run script
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml && \
  echo "Experiment completed!" | mail -s "FL Experiment Done" your-email@example.com
```

---

## 🎬 Example: Full 24-Hour Campaign

```bash
#!/bin/bash
# Complete campaign script

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/campaign_$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "Starting full campaign at $(date)" | tee "$LOG_DIR/log.txt"

# Load optimization
source ~/.fl_optimization.sh
source fl_env/bin/activate

# Experiment 1: Cognitive Defence (4 hours)
echo "Experiment 1: Cognitive Defence" | tee -a "$LOG_DIR/log.txt"
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml \
  2>&1 | tee -a "$LOG_DIR/exp1.log"
sleep 300

# Experiment 2: Adaptive Attacks (6 hours)
echo "Experiment 2: Adaptive Attacks" | tee -a "$LOG_DIR/log.txt"
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_adaptive.yaml \
  2>&1 | tee -a "$LOG_DIR/exp2.log"

# Analysis
echo "Analyzing results..." | tee -a "$LOG_DIR/log.txt"
python analyze_experiments.py 2>&1 | tee -a "$LOG_DIR/analysis.txt"

# Backup
echo "Backing up results..." | tee -a "$LOG_DIR/log.txt"
tar -czf results_$TIMESTAMP.tar.gz logs/ experiments/results/
gsutil cp results_$TIMESTAMP.tar.gz gs://your-bucket/fl-results/

echo "Campaign completed at $(date)" | tee -a "$LOG_DIR/log.txt"
```

---

## 🎯 Final Checklist Before Running

- [ ] GCP instance is running and accessible
- [ ] All dependencies installed and verified
- [ ] MNIST dataset downloaded
- [ ] System optimized with `optimize_gcp_instance.sh`
- [ ] At least 20GB free disk space
- [ ] Network connection is stable
- [ ] SSH keys configured for secure access
- [ ] Cloud storage credentials configured (if using backup)
- [ ] Monitoring terminals ready
- [ ] Configuration files reviewed and customized
- [ ] Backup/archive plan in place
- [ ] Post-analysis scripts ready

---

Once all checks pass, you're ready to run production-level experiments! 🚀

