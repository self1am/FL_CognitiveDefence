# Quick Start: Running Experiments on GCP VM

## 🚀 One-Time Setup

SSH into your VM and run:

```bash
cd FL_CognitiveDefence
./scripts/setup_vm.sh
```

This will:
- Make all scripts executable
- Install dependencies
- Create log directories
- Verify Ray works

---

## 📊 Running Experiments

### Method 1: Simple Background Run

```bash
# Start experiment
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml

# Monitor in real-time (in another SSH session or tmux pane)
python scripts/monitor_experiment.py baseline_100_clients_optimized --show-logs
```

### Method 2: With Auto-Restart (Recommended)

```bash
# Use tmux to keep session alive
tmux new -s experiment

# Run with auto-restart on failure
./scripts/supervise_experiment.sh experiments/configs/baseline_100_clients_optimized.yaml

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t experiment
```

### Method 3: Web Dashboard (Best Visibility)

```bash
# Terminal 1: Start API server
tmux new -s api
python scripts/monitoring_api.py
# Detach: Ctrl+B, then D

# Terminal 2: Start experiment
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml

# On your LOCAL machine, setup port forwarding:
ssh -L 5000:localhost:5000 miraahanafee@<VM_EXTERNAL_IP>

# Open browser: http://localhost:5000
```

---

## 🔍 Monitoring

### Check Running Experiments
```bash
python scripts/monitor_experiment.py --list
```

### Monitor Specific Experiment
```bash
python scripts/monitor_experiment.py baseline_100_clients_optimized
```

### View Logs
```bash
tail -f logs/experiments/baseline_100_clients_optimized_*.log
```

### Check System Resources
```bash
./scripts/check_system_resources.sh
```

---

## 🛑 Stopping Experiments

### Find PID
```bash
cat logs/experiments/baseline_100_clients_optimized.pid
```

### Kill Process
```bash
kill <PID>
```

### Or Use Monitor
```bash
python scripts/monitor_experiment.py --list
# Note the PID, then kill it
```

---

## 🐛 Troubleshooting

### Experiment Dies Silently
Check for OOM kills:
```bash
./scripts/check_system_resources.sh | grep -i "oom"
```

### Ray Issues
```bash
# Kill all Ray processes
pkill -9 -f ray

# Restart experiment
./scripts/run_experiment_robust.sh <config>
```

### SSH Disconnection
Always use `tmux` or the robust runner script (has nohup built-in).

---

## 📁 Where Are My Logs?

All logs are in `logs/experiments/`:
- `{experiment}_{timestamp}.log` - Main experiment log
- `{experiment}_{timestamp}_monitor.log` - Resource usage
- `{experiment}_status.json` - Current status (for API)
- `{experiment}.pid` - Process ID

---

## ⚡ Performance Tips

1. **Use optimized config**: `baseline_100_clients_optimized.yaml` (not the regular one)
2. **Monitor memory**: `watch -n 5 free -h`
3. **Use tmux**: Prevents SSH disconnection issues
4. **Enable swap**: If OOM persists (see PRODUCTION_RUNNING_GUIDE.md)

---

## 📖 Full Documentation

See [PRODUCTION_RUNNING_GUIDE.md](PRODUCTION_RUNNING_GUIDE.md) for complete documentation.

---

## 🎯 Typical Workflow

```bash
# 1. SSH into VM
ssh miraahanafee@<VM_IP>

# 2. Navigate to project
cd FL_CognitiveDefence
source fl_env/bin/activate

# 3. Start experiment in tmux
tmux new -s exp1
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml

# 4. Detach (Ctrl+B, then D)

# 5. Monitor from another session or your local machine
python scripts/monitor_experiment.py baseline_100_clients_optimized

# 6. Check back later
tmux attach -t exp1

# 7. View results
ls logs/experiments/
```

---

## 🆘 Quick Help

| Task | Command |
|------|---------|
| Run experiment | `./scripts/run_experiment_robust.sh <config>` |
| List experiments | `python scripts/monitor_experiment.py --list` |
| Monitor experiment | `python scripts/monitor_experiment.py <name>` |
| View logs | `tail -f logs/experiments/<name>_*.log` |
| Check OOM kills | `./scripts/check_system_resources.sh` |
| Stop experiment | `kill $(cat logs/experiments/<name>.pid)` |
| Web dashboard | `python scripts/monitoring_api.py` |
