# Production Experiment Running Guide

## Problem Overview
When running large-scale federated learning experiments (100 clients) on a GCP VM with 64GB RAM, experiments fail silently during rounds 1-2 due to:
- Memory exhaustion (OOM killer)
- Ray worker crashes
- SSH disconnection (although process should survive)
- Lack of visibility into failures

## Solutions Implemented

### 1. Robust Experiment Runner (`scripts/run_experiment_robust.sh`)
**Purpose**: Run experiments reliably in the background with comprehensive logging

**Features**:
- Runs with `nohup` to survive SSH disconnections
- Comprehensive logging to timestamped files
- Real-time resource monitoring (memory, CPU)
- PID file management to prevent duplicate runs
- Status JSON file for external monitoring

**Usage**:
```bash
# On your GCP VM:
cd ~/FL_CognitiveDefence
chmod +x scripts/run_experiment_robust.sh
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml
```

**Output Files**:
- Log file: `logs/experiments/{experiment}_{timestamp}.log`
- Monitor log: `logs/experiments/{experiment}_{timestamp}_monitor.log`
- PID file: `logs/experiments/{experiment}.pid`
- Status file: `logs/experiments/{experiment}_status.json`

**Monitor Progress**:
```bash
# Watch experiment log in real-time
tail -f logs/experiments/baseline_100_clients_optimized_*.log

# Watch resource usage
tail -f logs/experiments/baseline_100_clients_optimized_*_monitor.log
```

---

### 2. Process Supervisor (`scripts/supervise_experiment.sh`)
**Purpose**: Auto-restart experiments on failure with intelligent retry logic

**Features**:
- Automatic restart on failure (up to 3 retries)
- Detects OOM kills and other critical errors
- 60-second delay between retries
- Logs failure analysis
- Graceful cleanup on termination

**Usage**:
```bash
# Run with supervision (recommended for unreliable experiments)
chmod +x scripts/supervise_experiment.sh
./scripts/supervise_experiment.sh experiments/configs/baseline_100_clients_optimized.yaml
```

**When to Use**:
- First time running a new experiment configuration
- Known unstable experiments
- When you can't monitor manually

---

### 3. Experiment Monitor (`scripts/monitor_experiment.py`)
**Purpose**: Real-time monitoring dashboard in terminal

**Features**:
- Shows experiment progress (rounds, accuracy, loss)
- Displays resource usage (memory, CPU)
- Lists all active experiments
- Auto-refreshing dashboard

**Usage**:
```bash
# Monitor specific experiment
python scripts/monitor_experiment.py baseline_100_clients_optimized

# Monitor all experiments
python scripts/monitor_experiment.py

# Show recent log entries
python scripts/monitor_experiment.py baseline_100_clients_optimized --show-logs

# List all experiments
python scripts/monitor_experiment.py --list
```

---

### 4. Web Monitoring API (`scripts/monitoring_api.py`)
**Purpose**: Web-based dashboard and REST API for remote monitoring

**Features**:
- JSON REST API for external tools
- Built-in HTML dashboard
- Real-time experiment status
- System resource monitoring
- Log retrieval

**Setup**:
```bash
# Install dependencies
pip install flask flask-cors psutil

# Start API server (in a separate tmux/screen session)
python scripts/monitoring_api.py
```

**Access**:
- Dashboard: `http://<VM_IP>:5000/`
- API: `http://<VM_IP>:5000/api/experiments`

**Port Forwarding** (from local machine):
```bash
ssh -L 5000:localhost:5000 miraahanafee@<VM_EXTERNAL_IP>
# Then access http://localhost:5000 in your browser
```

**API Endpoints**:
- `GET /api/experiments` - List all experiments
- `GET /api/experiments/<name>` - Get experiment details
- `GET /api/experiments/<name>/logs?lines=100` - Get recent logs
- `GET /api/system` - System resource info
- `GET /api/health` - Health check

---

### 5. Optimized Configuration (`baseline_100_clients_optimized.yaml`)
**Purpose**: Prevent OOM kills by tuning memory parameters

**Key Changes**:
- Reduced `max_memory_mb` from 6000 to 4000
- Added Ray memory limits (50GB total, 10GB object store)
- Enabled automatic object spilling to disk
- Configured min_clients: 20 (trains in batches, not all 100 at once)

**Usage**:
```bash
python -m src.orchestration.simulation_runner --config experiments/configs/baseline_100_clients_optimized.yaml
```

---

### 6. System Diagnostics (`scripts/check_system_resources.sh`)
**Purpose**: Diagnose OOM kills and resource issues

**Usage**:
```bash
chmod +x scripts/check_system_resources.sh
./scripts/check_system_resources.sh
```

**Checks**:
- Memory status
- Recent OOM kills in system logs
- Running Python/Ray processes
- Disk usage
- Swap configuration
- Process limits

---

## Recommended Workflow

### First Time Setup (One-time)
```bash
cd ~/FL_CognitiveDefence
chmod +x scripts/*.sh
pip install flask flask-cors psutil
```

### Running Experiments

**Option A: Simple Background Run** (for stable experiments)
```bash
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml

# Monitor in another terminal
python scripts/monitor_experiment.py baseline_100_clients_optimized --show-logs
```

**Option B: Supervised Run** (recommended for new/unstable experiments)
```bash
# Use tmux or screen to keep session alive
tmux new -s experiment

# Run with supervision
./scripts/supervise_experiment.sh experiments/configs/baseline_100_clients_optimized.yaml

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t experiment
```

**Option C: Web Monitoring** (best visibility)
```bash
# Terminal 1: Start monitoring API
tmux new -s monitoring
python scripts/monitoring_api.py
# Detach: Ctrl+B, then D

# Terminal 2: Start experiment
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml

# From your local machine, setup port forwarding
ssh -L 5000:localhost:5000 miraahanafee@<VM_IP>

# Open browser: http://localhost:5000
```

---

## Debugging Failed Experiments

### 1. Check if OOM Killed
```bash
./scripts/check_system_resources.sh | grep -i "oom\|killed"

# Or check system logs directly
sudo journalctl --since "1 hour ago" | grep -i "oom"
```

### 2. Find Experiment Logs
```bash
# List all experiment logs
ls -lth logs/experiments/

# View latest log
tail -f logs/experiments/baseline_100_clients_optimized_*.log
```

### 3. Check Process Status
```bash
# See if experiment is still running
python scripts/monitor_experiment.py --list

# Or check manually
ps aux | grep simulation_runner
```

### 4. Analyze Failure Point
```bash
# Search for errors in log
grep -i "error\|exception\|killed\|failed" logs/experiments/baseline_100_clients_optimized_*.log
```

---

## Performance Optimization Tips

### 1. Reduce Memory Usage
- Lower `min_clients` in config (trains fewer clients per round)
- Reduce `max_memory_mb` in config
- Enable Ray object spilling (already in optimized config)

### 2. Speed Up Experiments
- Increase `num_cpus` in `ray_init_args` if you have more cores
- Reduce `spawn_delay` in orchestration config
- Use smaller batch sizes if memory allows

### 3. Add Swap Space (if OOM persists)
```bash
# Create 32GB swap file
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 4. Monitor During Experiment
```bash
# Watch memory in real-time
watch -n 5 'free -h && echo "" && ps aux --sort=-%mem | head -n 10'
```

---

## Using tmux/screen (Recommended)

### tmux Commands
```bash
# Create new session
tmux new -s experiment_name

# Detach (keep running in background)
Ctrl+B, then D

# List sessions
tmux ls

# Reattach to session
tmux attach -t experiment_name

# Kill session
tmux kill-session -t experiment_name
```

### screen Commands
```bash
# Create new session
screen -S experiment_name

# Detach (keep running in background)
Ctrl+A, then D

# List sessions
screen -ls

# Reattach to session
screen -r experiment_name

# Kill session
screen -X -S experiment_name quit
```

---

## Troubleshooting

### Problem: Process dies without error
**Solution**: Check for OOM kills
```bash
./scripts/check_system_resources.sh
```

### Problem: Can't connect to Ray dashboard
**Solution**: Use port forwarding
```bash
# From local machine
ssh -L 8265:localhost:8265 miraahanafee@<VM_IP>
# Then open http://localhost:8265
```

### Problem: Experiment stuck at round 1
**Solution**: Check Ray workers
```bash
# See Ray status
python -c "import ray; ray.init(); print(ray.cluster_resources())"

# Kill stuck Ray processes
pkill -9 -f ray
```

### Problem: SSH disconnects kill the process
**Solution**: Always use tmux/screen or nohup
```bash
# Use the robust runner (has nohup built-in)
./scripts/run_experiment_robust.sh <config>

# Or wrap any command
nohup python -m src.orchestration.simulation_runner --config <config> > output.log 2>&1 &
```

---

## File Structure
```
FL_CognitiveDefence/
├── scripts/
│   ├── run_experiment_robust.sh       # Main runner with logging
│   ├── supervise_experiment.sh         # Auto-restart supervisor
│   ├── monitor_experiment.py           # Terminal monitoring tool
│   ├── monitoring_api.py               # Web API + dashboard
│   └── check_system_resources.sh      # System diagnostics
├── experiments/configs/
│   └── baseline_100_clients_optimized.yaml  # Optimized config
└── logs/experiments/
    ├── {experiment}_{timestamp}.log          # Experiment logs
    ├── {experiment}_{timestamp}_monitor.log  # Resource monitoring
    ├── {experiment}.pid                      # Process ID
    └── {experiment}_status.json              # Status for API
```

---

## Quick Reference

**Start experiment with monitoring**:
```bash
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml
python scripts/monitor_experiment.py baseline_100_clients_optimized --show-logs
```

**Check if running**:
```bash
python scripts/monitor_experiment.py --list
```

**View logs**:
```bash
tail -f logs/experiments/baseline_100_clients_optimized_*.log
```

**Stop experiment**:
```bash
# Find PID
cat logs/experiments/baseline_100_clients_optimized.pid
# Kill process
kill <PID>
```

**Web dashboard**:
```bash
python scripts/monitoring_api.py
# Access: http://<VM_IP>:5000
```
