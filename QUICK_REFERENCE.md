# Production FL Experiments - Quick Reference Card

## 🚀 TL;DR: Get Started in 5 Minutes

```bash
# 1. SSH into GCP instance
gcloud compute ssh your-instance --zone=your-zone

# 2. Clone project
git clone <repo-url>
cd FL_CognitiveDefence

# 3. Optimize instance
chmod +x scripts/*.sh
./scripts/optimize_gcp_instance.sh

# 4. Activate virtual environment
source ~/.fl_optimization.sh
source fl_env/bin/activate

# 5. Install dependencies
pip install -r requirements.txt -e .

# 6. Run production experiments
./scripts/run_production_experiments.sh --all
```

---

## 📊 Experiment Configurations Available

| Config | Clients | Rounds | Attacks | Duration | File |
|--------|---------|--------|---------|----------|------|
| Cognitive Defence | 100 | 40 | Label Flip + Gradient Noise | ~4h | `production_100_clients_cognitive.yaml` |
| Adaptive Attacks | 100 | 50 | All 4 Attack Types | ~6-8h | `production_100_clients_adaptive.yaml` |
| Multi-Defence | 100 | 40 | Mixed (configurable) | ~4h | `production_100_clients_multidefence.yaml` |

---

## 🎯 Running Experiments

### Option 1: Single Experiment
```bash
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml
```

### Option 2: All Experiments (Automated)
```bash
./scripts/run_production_experiments.sh --all
```

### Option 3: With Resource Monitoring (in separate terminal)
```bash
./scripts/run_production_experiments.sh --monitor
```

---

## 📈 Monitoring During Execution

### In Separate Terminal - Watch Resources
```bash
watch -n 2 'free -h; echo "---"; ps aux | grep python | wc -l'
```

### View Live Logs
```bash
tail -f logs/<experiment_name>_complete.json
```

### Check Top Processes
```bash
top -c -u $USER
```

---

## 💾 Resource Breakdown

### Memory Usage (100 clients)
- **Base System**: 2-3 GB
- **Server Process**: 2-3 GB
- **8 Concurrent Clients**: 40-45 GB
- **Data/Buffers**: 5-10 GB
- **Total**: ~55-60 GB of 64GB ✓

### CPU Usage (100 clients)
- **Peak**: 7-8 vCPU fully utilized ✓
- **Server Overhead**: 1-2 vCPU

### Duration Estimation
- **Per Round**: 4-6 minutes
- **40 Rounds**: ~3-4 hours
- **With Setup**: 4 hours
- **50 Rounds**: ~4-5 hours
- **With Setup**: 5-6 hours

---

## 🔧 Configuration Customization

### Increase Attack Rate (to 40%)
Edit config file:
```yaml
attacks:
  - attack_type: "label_flip"
    target_clients: [0, 1, ..., 19]  # 20 clients
  - attack_type: "gradient_noise"
    target_clients: [20, 21, ..., 39]  # 20 clients
```

### Increase Rounds (to 50)
```yaml
experiment:
  num_rounds: 50  # Instead of 40
```

### Reduce Batch Size (if memory is tight)
```yaml
orchestration:
  batch_size: 6  # Instead of 8
  max_memory_mb: 48000  # Instead of 58000
```

### Reduce Clients (for faster testing)
```yaml
orchestration:
  num_clients: 50  # Instead of 100
```

---

## 📊 Analysis & Results

### After Experiment Completes

```bash
# View results summary
cat logs/campaign_*/results_summary.txt

# Analyze attack impact
python analyze_attack_impact.py

# Visualize results
python experiments/visualize_results.py \
  --config experiments/configs/production_100_clients_cognitive.yaml

# Generate comparison
python -c "
import json
from pathlib import Path

for log in Path('logs').glob('*_complete.json'):
    with open(log) as f:
        data = json.load(f)
        acc = data.get('centralized_accuracy', [])
        if acc:
            print(f'{log.stem}: Final Accuracy={acc[-1]:.4f}')
"
```

---

## ⚠️ Common Issues & Solutions

### Issue: OOM (Out of Memory)
```bash
# Reduce concurrent clients
batch_size: 6  # Instead of 8

# Or reduce total clients
num_clients: 75  # Instead of 100
```

### Issue: Clients Timeout/Disconnect
```yaml
orchestration:
  client_timeout_seconds: 2400  # 40 min instead of 30 min
  spawn_delay: 1.5  # Reduce stagger
```

### Issue: Disk Space Low
```bash
# Clean old logs
rm -rf logs/*_complete.json

# Or compress
gzip logs/*.json
```

### Issue: Server Crashes
```bash
# Restart and resume
pkill -9 -f "experiment_runner"
sleep 10
# Run experiment again (it will try to recover)
```

---

## 🔍 Expected Results

### Cognitive Defence vs No Defence (100 clients, 40 rounds)

**With Cognitive Defence:**
- Final Accuracy: 92-96% ✓
- Final Loss: 0.08-0.12 ✓
- Detected Anomalies: 35-38/100 clients

**Without Defence (Attack Only):**
- Final Accuracy: 65-70% ✗
- Final Loss: 1.2-1.5 ✗

---

## 📋 Pre-Experiment Checklist

- [ ] GCP instance created (64GB, 8vCPU)
- [ ] SSH access verified
- [ ] Project cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] MNIST dataset downloaded
- [ ] System optimized (ran optimize_gcp_instance.sh)
- [ ] At least 20GB free disk space available
- [ ] Network stable
- [ ] Ready to run!

---

## 🎬 Running First Experiment (Step-by-Step)

```bash
# 1. SSH in
ssh user@your.gcp.ip

# 2. Navigate to project
cd FL_CognitiveDefence

# 3. Load optimization profile
source ~/.fl_optimization.sh

# 4. Activate virtual environment
source fl_env/bin/activate

# 5. Start monitoring in another terminal
# (in new SSH window)
ssh user@your.gcp.ip
cd FL_CognitiveDefence
watch -n 2 'free -h; uptime; ps aux | grep python | wc -l'

# 6. Run single experiment (Terminal 1)
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml

# 7. Watch for completion (~4 hours)
# Results will be saved to logs/<experiment_name>_complete.json
```

---

## 📞 Support

### Debug Mode
```bash
# Run with verbose logging
export LOGLEVEL=DEBUG
python -m src.orchestration.experiment_runner --config ...
```

### Check Logs
```bash
# See latest experiment log
tail -100f logs/*_complete.json | jq '.' 2>/dev/null

# See campaign summary
cat logs/campaign_*/campaign.log
```

### Kill Stuck Experiment
```bash
# Graceful (wait for current round)
pkill -f "experiment_runner"

# Force kill (immediate)
pkill -9 -f "python.*client"
pkill -9 -f "experiment_runner"
```

---

## 💡 Pro Tips

1. **Use tmux/screen for detachable sessions**
   ```bash
   screen -S fl_experiment
   # Run experiment
   # Ctrl+A then D to detach
   screen -r fl_experiment  # Reattach later
   ```

2. **Run analysis in parallel**
   ```bash
   # Terminal 1: Run experiment
   python -m src.orchestration.experiment_runner --config ...
   
   # Terminal 2: Monitor resources
   watch -n 2 'free -h'
   
   # Terminal 3: Watch logs
   tail -f logs/*_complete.json
   ```

3. **Archive results immediately**
   ```bash
   # After experiment completes
   tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz logs/ experiments/results/
   gsutil cp results_*.tar.gz gs://your-bucket/  # Upload to cloud
   ```

4. **Set up automated backup**
   ```bash
   # Add to crontab
   0 * * * * cd /path/to/project && tar -czf backup_$(date +\%Y\%m\%d_\%H\%M\%S).tar.gz logs/ && gsutil cp backup_*.tar.gz gs://your-bucket/
   ```

---

## 📚 Documentation Files

- `PRODUCTION_EXPERIMENT_GUIDE.md` - Comprehensive guide (this file)
- `README.md` - Project overview
- `docs/ADAPTIVE_ATTACKS.md` - Attack details
- `CENTRALIZED_EVAL_GUIDE.md` - Evaluation metrics
- `ANOMALY_SCORING_EXPLAINED.md` - Cognitive defence details

