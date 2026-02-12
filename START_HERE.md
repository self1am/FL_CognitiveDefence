# 🚀 READY TO USE: Fast & Failproof Experiments

## ⚡ Quick Start (3 Steps)

### 1️⃣ Sync to VM (from your local machine)
```bash
cd FL_CognitiveDefence
./scripts/sync_to_vm.sh <YOUR_VM_IP> miraahanafee
```

### 2️⃣ Setup on VM (one time, SSH into VM first)
```bash
ssh miraahanafee@<YOUR_VM_IP>
cd FL_CognitiveDefence
./scripts/setup_vm.sh
```

### 3️⃣ Run Experiment
```bash
# Best option: with auto-restart
tmux new -s exp1
./scripts/supervise_experiment.sh experiments/configs/baseline_100_clients_optimized.yaml
# Ctrl+B, then D to detach

# Or simple background run:
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml
```

---

## 📊 Monitor Your Experiment

### Quick Status Check
```bash
./scripts/what_is_running.sh
```

### Live Monitoring
```bash
python scripts/monitor_experiment.py baseline_100_clients_optimized --show-logs
```

### Web Dashboard (Best)
```bash
# On VM (in separate tmux session):
tmux new -s dashboard
python scripts/monitoring_api.py
# Ctrl+B, D

# On your local machine (new terminal):
ssh -L 5000:localhost:5000 miraahanafee@<VM_IP>
# Open browser: http://localhost:5000
```

---

## 🎯 What Problems Are Solved?

| Problem | Solution |
|---------|----------|
| ❌ Takes forever | ✅ Optimized memory config |
| ❌ Fails silently | ✅ Comprehensive logging |
| ❌ No awareness | ✅ 3 monitoring tools |
| ❌ SSH disconnect kills it | ✅ Background process (nohup) |
| ❌ OOM killer | ✅ Memory optimization + detection |

---

## 📁 New Files You Have

### Scripts (in `scripts/`)
- ✅ `run_experiment_robust.sh` - Main runner
- ✅ `supervise_experiment.sh` - Auto-restart on failure
- ✅ `monitor_experiment.py` - Terminal dashboard
- ✅ `monitoring_api.py` - Web dashboard
- ✅ `what_is_running.sh` - Status checker
- ✅ `check_system_resources.sh` - Diagnostics
- ✅ `setup_vm.sh` - One-time setup
- ✅ `sync_to_vm.sh` - Sync from local

### Configs
- ✅ `baseline_100_clients_optimized.yaml` - Memory-optimized

### Docs
- ✅ `VM_COMMANDS.md` - Copy-paste commands
- ✅ `VM_QUICK_START.md` - Quick reference
- ✅ `PRODUCTION_RUNNING_GUIDE.md` - Full guide
- ✅ `SOLUTION_SUMMARY.md` - This file

---

## 🆘 Common Commands

```bash
# Check status
./scripts/what_is_running.sh

# Start experiment
./scripts/run_experiment_robust.sh <config>

# Monitor
python scripts/monitor_experiment.py <name>

# View logs
tail -f logs/experiments/<experiment>_*.log

# Stop
kill $(cat logs/experiments/<name>.pid)

# Check for OOM
./scripts/check_system_resources.sh

# Web dashboard
python scripts/monitoring_api.py
```

---

## 📖 Full Documentation

- **Quick Commands:** [VM_COMMANDS.md](VM_COMMANDS.md)
- **Quick Start:** [VM_QUICK_START.md](VM_QUICK_START.md)
- **Complete Guide:** [PRODUCTION_RUNNING_GUIDE.md](PRODUCTION_RUNNING_GUIDE.md)

---

## ✅ You're Ready!

Everything is set up. Just follow the 3 quick start steps above and your experiments will run reliably in the background with full monitoring! 🎉
