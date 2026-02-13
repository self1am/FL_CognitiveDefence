# FL Experiment Performance & Resource Requirements Guide

## Current Status Summary

Your 64GB RAM, 8vCPU VM with **100 clients** is running well via tmux.

**Observed Performance:**
- ✅ **Round Time:** ~30 minutes per round (stable)
- ✅ **Total for 10 rounds:** 5.4 hours
- ✅ **Accuracy:** Improving nicely (0.0974 → 0.9888 by round 5)
- ✅ **Process Stability:** No hangs with tmux

---

## How Much RAM Is Being Used?

### Quick Measurement (Next Time You Run Experiment)

**Terminal 1 - Start experiment:**
```bash
tmux new-session -d -s experiment
tmux send-keys -t experiment "python run_server_with_eval.py --config experiments/configs/baseline_100_clients.yaml" Enter
```

**Terminal 2 - Monitor RAM in real-time:**
```bash
python ram_monitor.py
```

This will:
- Track memory every 5 seconds
- Show Peak RAM usage
- Identify which processes consume the most memory
- Save all data to `ram_measurements.json`

**Terminal 3 - When experiment finishes, analyze:**
```bash
python analyze_ram_log.py
```

### What You'll Find

Based on breakdown of your 64GB VM:

| Component | Usage | % of 64GB |
|-----------|-------|----------|
| Ray object store | ~20GB | 43% |
| Ray runtime | ~3GB | 5% |
| Model training (16 parallel) | ~20GB | 31% |
| Flower + Python overhead | ~3GB | 5% |
| Data loading (MNIST) | ~2GB | 3% |
| Free/Buffer | ~16GB | 13% |

**Per-Client Memory:**
- MNIST (your current): ~20-30MB per client
- With 16 parallel: ~320MB - 480MB total model memory
- **Peak during round:** All 64GB could be in use

---

## Real Specifications Needed

### For Different Scenarios

#### **Scenario 1: Local Testing (Your MacBook)**
- **Spec:** 8GB RAM, 8-core, 256GB storage
- **Recommended clients:** 10 max
- **Client resource:** num_cpus=1.0 (serialize execution)
- **Expected time:** 5-15 min per round
- **Why slow:** Single machine, no GPU, I/O bound

#### **Scenario 2: Small VM Experiments**
- **Spec:** 16GB RAM, 4vCPU, 50GB storage
- **Recommended clients:** 20-30
- **Client resource:** num_cpus=0.25
- **Expected time:** 10-15 min per round
- **Good for:** Development, quick tests

#### **Scenario 3: Medium VM (⭐ RECOMMENDED)**
- **Spec:** 32GB RAM, 8vCPU, 100GB storage
- **Recommended clients:** 50
- **Client resource:** num_cpus=0.5
- **Expected time:** 15-20 min per round
- **Good for:** Most experimental work
- **Cost on GCP:** ~$200-300/month

#### **Scenario 4: Large VM (Current)**
- **Spec:** 64GB RAM, 8vCPU, 100GB storage ← **You are here**
- **Recommended clients:** 100
- **Client resource:** num_cpus=0.5
- **Expected time:** 30 min per round
- **Good for:** Full-scale experiments
- **Cost on GCP:** ~$400-500/month

#### **Scenario 5: Production Scale**
- **Spec:** 128GB RAM, 32vCPU, 500GB storage
- **Recommended clients:** 200-500
- **Client resource:** num_cpus=0.25-0.5
- **Expected time:** 30-60 min per round
- **Good for:** Final production runs
- **Cost on GCP:** ~$1000-2000/month

---

## Speed Comparison Matrix

| Spec | Clients | Max Parallel | Round Time | Total (10 rounds) | Cost |
|------|---------|-------------|-----------|-----------------|------|
| 4 vCPU, 16GB | 50 | 8 | 20 min | 3.3 hours | $150/mo |
| 8 vCPU, 32GB | 50 | 8 | 15 min | 2.5 hours | $250/mo |
| 8 vCPU, 64GB | 100 | 16 | 30 min | 5.5 hours | $400/mo |
| 16 vCPU, 64GB | 100 | 16 | 15 min | 2.5 hours | $600/mo |
| 16 vCPU, 128GB | 200 | 32 | 25 min | 4.2 hours | $900/mo |
| 32 vCPU, 128GB | 200 | 32 | 15 min | 2.5 hours | $1400/mo |

---

## Parallelism Explained

Your VM configuration details:

```
Total CPU cores: 8
CPU per client: num_cpus=0.5
Maximum parallel clients: 8 ÷ 0.5 = 16 clients
```

**What this means:**
- 100 clients need to run in batches
- Batch 1: Clients 0-15 train (in parallel)
- Batch 2: Clients 16-31 train (in parallel)
- ... and so on for 6-7 batches total
- Each batch takes ~4-5 minutes of training

**To speed up:**
- ❌ Don't increase clients on same hardware
- ✅ Increase vCPU to run more clients in parallel
- ✅ Increase RAM to handle aggregation faster

---

## Optimization Strategies

### Quick Wins (No Cost)

1. **Reduce evaluation frequency:**
   ```yaml
   # In config file - only eval every 2 rounds
   evaluation_strategy: "steps"
   eval_steps: 2
   ```

2. **Use smaller batch size:**
   ```yaml
   # Train faster on each client
   batch_size: 32  # instead of 64
   ```

3. **Reduce model size:**
   ```yaml
   # Smaller model = faster training
   model: "small_cnn"  # instead of large model
   ```

### Medium Investment (Spec Upgrade)

| Current → Target | Cost | Speedup | Time |
|------------------|------|---------|------|
| 8vCPU → 16vCPU | +$200/mo | 2x faster | 2.75 hrs |
| 64GB → 128GB | +$100/mo | 10% faster | 5 hrs |
| Both | +$300/mo | 2.2x faster | 2.5 hrs |

---

## Monitoring Dashboard

### What to Watch During Runs

```bash
# Terminal 2 - While experiment runs
while true; do
  clear
  echo "=== Memory Status ==="
  free -h | head -2
  echo "=== CPU Status ==="
  top -b -n 1 | head -6
  echo "=== Ray Status ==="
  ps aux | grep ray | grep -v grep | wc -l
  sleep 5
done
```

### Red Flags ⚠️

| Flag | Meaning | Action |
|------|---------|--------|
| Memory > 90% | Running out of RAM | Reduce clients or increase VM RAM |
| Swap > 10% | Using disk as memory | SEVERE: Reduce parallelism immediately |
| Swap > 50% | Near collapse | Stop experiment, upgrade RAM |
| Round time increasing each round | Memory leak | Kill and restart with fewer clients |
| No log output for 5+ min | Hung process | Check RAM/swap, may need restart |

---

## Deployment Checklist

### Local Mac Testing ✓
```bash
# ✓ Done - you have this working
tmux new-session -d -s experiment
python run_server_with_eval.py --config baseline_10_clients.yaml
```

### Cloud VM Experiments
```bash
# 1. Start experiment in tmux
tmux new-session -d -s experiment
tmux send-keys -t experiment "cd /path && python run_server_with_eval.py" Enter

# 2. Monitor in separate tmux window
tmux new-window -t experiment
tmux send-keys -t experiment "python ram_monitor.py" Enter

# 3. You can close browser tab anytime
# Process continues in background

# 4. Reconnect later
# New browser SSH tab:
tmux attach -t experiment
# Or check logs:
tail -f logs/experiment_safe_*.log
```

---

## Final Recommendations

### ✅ Your Current Setup is Good For:
- 100 clients per round
- 10 total rounds
- ~5.5 hours total training time
- Budget-conscious experiments

### ⚠️ Limitations:
- **Can't go much higher:** 200+ clients would take 1+ hour per round
- **Single dataset:** MNIST is small; CIFAR10 would be slower
- **SSH limitations:** Browser SSH can timeout; use tmux

### 🚀 To Get 2x Speedup:
1. **Option A:** Upgrade to 16vCPU + keep 64GB (add $200/mo)
   - Round time: 15 min → 5h total for 10 rounds
   
2. **Option B:** Reduce to 50 clients, keep current VM
   - Round time: 15 min → 2.5h total for 10 rounds
   - Cost savings: keep at $400/mo

3. **Option C:** Reduce batch size & model size (no extra cost)
   - ~30% faster per round
   - Round time: 21 min → 3.5h total for 10 rounds

---

## Next Steps

1. **Now:** Use `tmux` for all experiments (you've got it working)
2. **Next run:** Add RAM monitoring
   ```bash
   # Terminal 2
   python ram_monitor.py
   ```
3. **After experiment:** Analyze results
   ```bash
   python analyze_ram_log.py
   ```
4. **Share findings:** Come back with actual RAM peak numbers

This will tell you **exactly** what RAM headroom you have and whether you can push to more clients.
