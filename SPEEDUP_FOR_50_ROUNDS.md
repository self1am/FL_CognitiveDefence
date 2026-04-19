# Speed Up Your 50-Round Experiments (From 16.7 hrs → 11 hrs single, or 3-5 hrs parallel)

## Your Current Bottleneck

```
CPU:  100% maxed
RAM:  11GB/64GB (only 17% used)
Rounds: 100 clients × 20 min/round = 2000 min per 100 rounds
50 rounds: 1000 min = 16.7 hours
```

**Problem:** You're wasting 53GB of unused RAM because you're CPU-limited, not memory-limited.

---

## 🎯 Option 1: Faster Single Experiments (Simplest)

**Reduce clients from 100 → 50:**

```bash
python run_server_with_eval.py --config experiments/configs/baseline_50_clients.yaml
```

**Why this works:**
- 50 clients ÷ 32 max parallel = 1.56 batches (faster to complete)
- Estimated round time: 12-14 min (vs 20 min)
- **50 rounds × 13 min = 650 min = 10.8 hours** (32% faster)
- RAM usage: ~6-8GB (still very safe)

**Trade-off:**
- 50 fewer clients per round (but still meaningful federated learning)
- Accuracy might be slightly different (but same training dynamics)

**When to use:**
- Quick ablation studies
- Testing attack/defense scenarios
- When absolute scale (100 clients) isn't critical

---

## 🚀 Option 2: Parallel Experiments (Best for Research)

Run **2 experiments with 50 clients each simultaneously**:

```bash
# Terminal 1 (or tmux window)
python run_server_with_eval.py --config experiments/configs/baseline_50_clients_parallel_a.yaml

# Terminal 2 (new tmux window)
python run_server_with_eval.py --config experiments/configs/baseline_50_clients_parallel_b.yaml
```

**Resource allocation:**
- Experiment A: 4 vCPU + 3GB RAM
- Experiment B: 4 vCPU + 3GB RAM
- Total: 8 vCPU + 6GB RAM used
- Remaining: 58GB RAM idle (but you don't need it)

**Results:**
- **Both experiments complete in ~11 hours** (not 22!)
- You get 2x results in similar time
- Ideal for: testing 2 attack scenarios, 2 models, 2 seeds simultaneously

**When to use:**
- A/B testing different defenses
- Running multiple seeds for statistical significance
- Ablation studies (e.g., with/without defense)

---

## ⚡ Option 3: Crazy Mode (3 Parallel)

Run 3 experiments with 35 clients each:

```bash
# Theory: 8 vCPU ÷ 3 = 2.67 vCPU per experiment
# Reality: More context switching, might be slower
```

**Pros:**
- 3x the results potentially
- Wall-clock time still ~11-13 hours per experiment

**Cons:**
- CPU contention (8 vCPU / 3 = not evenly divisible)
- Round times might increase to 15-20 min
- Worth testing but Option 2 is safer

---

## 📊 Time Comparison Matrix

| Approach | Clients | Round Time | 50 Rounds | Results | Wall-Clock |
|----------|---------|-----------|----------|---------|-----------|
| **Current** | 100 | 20 min | 16.7 hrs | 1 exp | 16.7 hrs |
| **Option 1** | 50 | 13 min | 10.8 hrs | 1 exp | 10.8 hrs |
| **Option 2** | 2×50 | 13 min each | 10.8 hrs each | 2 exp | 11 hrs total |
| **Option 3** | 3×35 | ~15 min each | 12.5 hrs each | 3 exp | 13 hrs total |

**ROI Analysis:**
- Option 1 saves 6 hours vs current (35% speedup, 0 cost, 1 result)
- Option 2 saves 6 hours + gets 2 results (saves 6 hrs vs 22 hrs sequential)
- Option 3 saves 3 hours + gets 3 results (but risk of CPU contention)

---

## 🔧 Quick Start: Option 1 (Recommended Now)

### Step 1: Test with 4 rounds
```bash
# Edit baseline_50_clients.yaml: change num_rounds from 50 → 4
vi experiments/configs/baseline_50_clients.yaml
# Change: num_rounds: 4

# Run quick test
python run_server_with_eval.py --config experiments/configs/baseline_50_clients.yaml

# Time how long 4 rounds take
# If ~54 min → round time is ~13.5 min ✓
# If ~40 min → round time is ~10 min ✓✓
```

### Step 2: If good, run full experiment
```bash
# Reset to num_rounds: 50
vi experiments/configs/baseline_50_clients.yaml
# Change: num_rounds: 50

# Run with monitoring
tmux new-session -d -s exp
tmux send-keys -t exp "python run_server_with_eval.py --config experiments/configs/baseline_50_clients.yaml" Enter

# Monitor in another terminal
python ram_monitor.py
python cpu_profiler.py
```

---

## 🚀 Quick Start: Option 2 (Parallel)

### Simple way (in tmux):
```bash
# Terminal 1
tmux new-session -d -s fl_experiments
tmux send-keys -t fl_experiments "cd ~/FL_CognitiveDefence && python run_server_with_eval.py --config experiments/configs/baseline_50_clients_parallel_a.yaml" Enter

# Terminal 2
tmux new-window -t fl_experiments
tmux send-keys -t fl_experiments "cd ~/FL_CognitiveDefence && python run_server_with_eval.py --config experiments/configs/baseline_50_clients_parallel_b.yaml" Enter

# Reconnect anytime
tmux attach -t fl_experiments
```

### Automated way (uses provided script):
```bash
bash run_parallel_experiments.sh 2
```

---

## 💰 Cost Analysis

| Scenario | Approach | Time | Cost (if $400/mo for VM) | Notes |
|----------|----------|------|---------|--------|
| 50 rounds, 1 exp | Current 100 clients | 16.7 hrs | $2.78 | Wasteful |
| 50 rounds, 1 exp | **50 clients** | 10.8 hrs | $1.80 | **31% savings** |
| 100 rounds, 2 exp | **2×50 parallel** | 21.6 hrs | $3.60 | Get 2x results |
| 150 rounds, 3 exp | **3×50 sequential** | 32.4 hrs | $5.40 | All for 3x results |

**Recommendation:**
- Use **Option 1 (50 clients)** for routine work
- Switch to **Option 2 (parallel)** when you need multiple scenarios
- Saves money AND gets you results faster

---

## 🎯 My Recommendation

**Phase 1 (Today):** Test Option 1
1. Update config to 50 clients
2. Run 4-round test, measure round time
3. If 12-15 min per round → proceed to full 50 rounds

**Phase 2 (If satisfied):** Try Option 2
1. Use parallel configs when you have multiple experiments
2. Get 2 results in basically same time as 1

**Phase 3 (Optional):** Upgrade VM
1. If you still need faster individual rounds, upgrade to 16 vCPU
2. Would give ~1.5x parallelism improvement
3. Costs ~$200/mo extra

---

## FAQ

**Q: Why not 100 clients with more vCPU?**
A: You'd need 16 vCPU to get true parallelism. Current overhead is from batching sequential training. Parallel training can't be parallelized per-client without GPU.

**Q: Won't 50 clients affect my results?**
A: Slightly different convergence pattern, but same dynamics. Good for ablation studies. For final paper, use 100+ clients.

**Q: What if I run 2 parallel and RAM spikes?**
A: Unlikely. Each experiment uses ~6-8GB peak, so 2×6 = 12GB max (still 50GB free).

**Q: Should I use GPU instead?**
A: For MNIST, GPU adds overhead (data transfer). Only helps with CIFAR10+ or larger models.

---

## Recommended Path Forward

```
Today:
  1. Create baseline_50_clients.yaml (done)
  2. Test 4 rounds: measure actual round time
  3. If good, scale to 50 rounds

Tomorrow:
  1. Run 50-client experiment with full monitoring
  2. Capture peak RAM and CPU numbers
  3. Decide if Option 2 (parallel) is worth trying

Next Week:
  1. If running complex attacks, use parallel (Option 2)
  2. Run 2-3 scenarios simultaneously
  3. Save weeks of experimental time
```

---

## Command Summary

```bash
# Test fast single experiments
python run_server_with_eval.py --config experiments/configs/baseline_50_clients.yaml

# Run 2 experiments in parallel
tmux new-session -d -s exp
tmux send-keys -t exp "python run_server_with_eval.py --config experiments/configs/baseline_50_clients_parallel_a.yaml" Enter
tmux new-window -t exp
tmux send-keys -t exp "python run_server_with_eval.py --config experiments/configs/baseline_50_clients_parallel_b.yaml" Enter

# Monitor
python ram_monitor.py
python cpu_profiler.py
```

Ready to test? Start with Option 1 and report back the round timings! 🚀
