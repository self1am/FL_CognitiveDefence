# Commands to Run on Your GCP VM

Copy and paste these commands directly into your GCP VM SSH session.

---

## 1️⃣ First-Time Setup (Run Once)

```bash
cd ~/FL_CognitiveDefence
chmod +x scripts/*.sh
./scripts/setup_vm.sh
```

---

## 2️⃣ Check Current Status

```bash
./scripts/what_is_running.sh
```

This shows:
- What experiments are running
- Memory/disk usage
- Recent logs
- Suggested actions

---

## 3️⃣ Start an Experiment

### Option A: Simple (runs in background, survives SSH disconnect)
```bash
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml
```

### Option B: With Auto-Restart (recommended for reliability)
```bash
tmux new -s experiment
./scripts/supervise_experiment.sh experiments/configs/baseline_100_clients_optimized.yaml
# Press Ctrl+B, then D to detach
```

---

## 4️⃣ Monitor Experiments

### Check what's running
```bash
python scripts/monitor_experiment.py --list
```

### Watch specific experiment (real-time)
```bash
python scripts/monitor_experiment.py baseline_100_clients_optimized --show-logs
```

### View raw logs
```bash
tail -f logs/experiments/baseline_100_clients_optimized_*.log
```

---

## 5️⃣ Web Dashboard (Optional but Recommended)

### On VM:
```bash
# Start API server
tmux new -s api
python scripts/monitoring_api.py
# Press Ctrl+B, then D to detach
```

### On Your Local Machine:
```bash
# Setup port forwarding
ssh -L 5000:localhost:5000 miraahanafee@<YOUR_VM_EXTERNAL_IP>

# Open browser: http://localhost:5000
```

---

## 6️⃣ Stop an Experiment

### Find and kill
```bash
# Find PID
cat logs/experiments/baseline_100_clients_optimized.pid

# Kill it
kill <PID>
```

### Or kill all experiments
```bash
pkill -f simulation_runner
```

---

## 7️⃣ Troubleshooting

### Check for OOM (Out of Memory) kills
```bash
./scripts/check_system_resources.sh
```

### Check system logs for crashes
```bash
sudo journalctl --since "1 hour ago" | grep -i "oom\|killed\|error"
```

### Clean up Ray if stuck
```bash
pkill -9 -f ray
# Then restart your experiment
```

### Check memory usage
```bash
watch -n 5 free -h
```

---

## 8️⃣ Using tmux (Keeps Processes Alive)

### Create session
```bash
tmux new -s experiment_name
# Run your commands here
```

### Detach (leave it running)
```
Press: Ctrl+B, then D
```

### List sessions
```bash
tmux ls
```

### Reattach to session
```bash
tmux attach -t experiment_name
```

### Kill session
```bash
tmux kill-session -t experiment_name
```

---

## 9️⃣ Quick Status Checks

### See all running processes
```bash
./scripts/what_is_running.sh
```

### Memory status
```bash
free -h
```

### Disk space
```bash
df -h
```

### System load
```bash
uptime
```

---

## 🔟 Emergency: System Too Slow / OOM

### Add swap space (32GB)
```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Kill all Python processes
```bash
pkill -9 python
```

### Clean Ray and restart
```bash
pkill -9 -f ray
rm -rf /tmp/ray
```

---

## 📚 Documentation Files

- `VM_QUICK_START.md` - Quick reference
- `PRODUCTION_RUNNING_GUIDE.md` - Complete guide
- `README.md` - Project overview

---

## 🎯 Typical Daily Workflow

```bash
# 1. SSH into VM
ssh miraahanafee@<VM_IP>

# 2. Check what's running
cd FL_CognitiveDefence
./scripts/what_is_running.sh

# 3. Start new experiment (if needed)
tmux new -s exp1
./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml
# Ctrl+B, D to detach

# 4. Monitor progress
python scripts/monitor_experiment.py --list

# 5. Check back later
tmux attach -t exp1

# 6. View final results
tail -100 logs/experiments/baseline_100_clients_optimized_*.log
```

---

## ⚠️ Important Notes

1. **Always use tmux** to prevent SSH disconnection issues
2. **Use optimized config** (`baseline_100_clients_optimized.yaml`) to prevent OOM
3. **Monitor memory** during experiments: `watch -n 5 free -h`
4. **Check logs regularly** to catch failures early
5. **Clean up Ray** if experiments get stuck

---

## 🆘 Getting Help

If something isn't working:
1. Run `./scripts/what_is_running.sh` to see current state
2. Check `./scripts/check_system_resources.sh` for OOM kills
3. View logs: `tail -100 logs/experiments/<experiment>_*.log`
4. Check Ray: `ps aux | grep ray`
5. Review documentation: `cat PRODUCTION_RUNNING_GUIDE.md`
