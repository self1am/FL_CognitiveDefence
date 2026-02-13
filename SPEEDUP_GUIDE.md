# How to Make FL Experiments Faster (With Headroom)

## Your Situation
- **Current:** 100 clients, 30 min per round, 16.6% RAM usage
- **Bottleneck:** NOT memory → likely CPU parallelism efficiency
- **Opportunity:** Significant room to optimize

---

## Method 1: CPU Parallelism Tuning (Best ROI)

### Current Config Analysis
```yaml
num_clients: 100
client_resources:
  num_cpus: 0.5        # ← Key tuning parameter
```

**What this means:**
- Each client needs 0.5 CPU cores
- Max parallel clients: 8 cores ÷ 0.5 = **16 clients**
- 100 clients ÷ 16 parallel = **6.25 batches** = ~7 training rounds

### Test: Increase Parallelism

**Option A - Moderate (Safest):** `num_cpus: 0.25`
```python
num_clients: 100
client_resources:
  num_cpus: 0.25       # Changed from 0.5
  
# Results:
# Max parallel: 8 ÷ 0.25 = 32 clients in parallel
# 100 clients ÷ 32 = 3.125 batches
# Speedup: ~50% faster (15 min/round instead of 30)
```

**Option B - Aggressive:** `num_cpus: 0.125`
```python
num_clients: 100
client_resources:
  num_cpus: 0.125      # Changed from 0.5
  
# Results:
# Max parallel: 8 ÷ 0.125 = 64 clients in parallel
# 100 clients ÷ 64 = 1.56 batches  
# Speedup: ~70% faster (9 min/round instead of 30)
# Risk: Heavy context switching, might bottleneck
```

### Recommendation: Start with 0.25

**Why?**
- Proven safe with modern Python/Ray
- 50% speedup is significant
- Each client still has meaningful CPU resources
- Easy to rollback if issues arise

---

## Method 2: Reduce Evaluation Overhead

Evaluation takes ~6.7 minutes per round. You can optimize this:

### Option A: Skip Early Rounds
```yaml
evaluation_strategy: "steps"
eval_steps: 2  # Only evaluate every 2 rounds, not every round

# Results:
# 10 rounds: 5 evals instead of 10 = save ~33 min total
# But: Less insight into training progress
```

### Option B: Use Smaller Test Set
```yaml
evaluation:
  num_test_samples: 5000  # Instead of 10000
  
# Results:
# Faster evaluation: ~3 min instead of ~7 min
# Slightly less accurate metrics, but still meaningful
```

### Option C: Parallel Evaluation
```yaml
evaluation:
  num_workers: 8  # Use all cores for evaluation
  
# Results:
# Evaluation: ~3 min instead of ~7 min
```

**Best case:** Combine all three = **50-60% faster total!**

---

## Method 3: Optimize Client-Side Training

### Reduce Epochs
```yaml
client:
  epochs: 1  # Instead of 2-3
  
# Results:
# Per-client training: 50% faster
# Overall speedup: ~25% per round
```

### Batch Size Tuning
```yaml
client:
  batch_size: 64  # Instead of 32 (if currently 32)
  
# Results:
# Modern GPUs/CPUs prefer larger batches
# Training: ~20% faster per client
```

### Skip Validation
```yaml
client:
  do_validation: false  # During training
  
# Results:
# Per-client training: ~15% faster
```

---

## Method 4: Ray Optimization

### Tune Ray Object Store
```python
# In your Ray init:
ray.init(
    object_store_memory=30*1024**3,  # 30GB (was ~20GB)
    plasma_directory="/dev/shm",      # Use RAM disk if available
    _temp_dir="/mnt/nvme"             # Fast SSD for temp
)

# Results:
# Smoother operations under load
# ~10% faster aggregation
```

### Increase Actor Pool Size
```yaml
flower:
  ray_actor_options:
    virtual_clients_per_actor: 2  # More workers
    
# Results:
# Better task distribution
# ~5-15% speedup
```

---

## Strategy: Ranked by Effort vs. Benefit

| Strategy | Effort | Speedup | Total Time |
|----------|--------|---------|-----------|
| Original (baseline) | - | 1.0x | 5.5 hrs |
| Reduce num_cpus: 0.5 → 0.25 | 🟢 Easy | 1.5x | 3.7 hrs |
| + Skip every other eval | 🟡 Medium | 2.2x | 2.5 hrs |
| + Reduce epochs 2 → 1 | 🟢 Easy | 2.8x | 2.0 hrs |
| + Batch size tuning | 🟡 Medium | 3.1x | 1.8 hrs |
| All options combined | 🟡 Medium | 3-4x | 1.5-1.8 hrs |

---

## Hands-On: Step-by-Step Speed Test

### Quick Test (30 minutes)

**Step 1:** Find the bottleneck
```bash
# Terminal 1: Run experiment
python run_server_with_eval.py --config baseline_100_clients.yaml

# Terminal 2: Profile CPU
python cpu_profiler.py
```

**Watch for:**
- If CPU usage < 50%: You can add more parallelism safely
- If CPU usage 60-80%: Good zone
- If CPU usage > 90%: Already maxed out

**Step 2:** Test parallelism improvement
```bash
# Create test config
cp experiments/configs/baseline_100_clients.yaml \
   experiments/configs/baseline_100_clients_fast.yaml
```

Edit `baseline_100_clients_fast.yaml`:
```yaml
server:
  num_rounds: 2  # Just 2 rounds for quick test

federated:
  num_clients: 100
  client_resources:
    num_cpus: 0.25  # Changed from 0.5 (2x more parallel)
```

**Step 3:** Run timing comparison
```bash
# Original
time python run_server_with_eval.py --config baseline_100_clients.yaml
# Note the time for 2 rounds

# Test version  
time python run_server_with_eval.py --config baseline_100_clients_fast.yaml
# Compare timing
```

**Step 4:** Scale to full run if good
```bash
# Update to full 10 rounds
# Copy good config to production
# Re-monitor RAM with cpu_profiler for peak usage
```

---

## Decide Based on Results

### If CPU < 50% during training:
```
✅ You can definitely go to num_cpus=0.25
✅ Consider even 0.125 if feeling adventurous
```

### If CPU 50-75%:
```
✅ Go to num_cpus=0.25
⚠️  Avoid going lower than 0.125
```

### If CPU > 85%:
```
⚠️  You're already well-utilized
❌ Won't benefit much from more parallelism
✅ Try evaluation optimization instead
```

---

## Full Optimized Config (Aggressive)

If CPU profiling shows you have headroom:

```yaml
# experiments/configs/baseline_100_clients_optimized.yaml

server:
  num_rounds: 10
  aggregation_strategy: "fedavg"
  evaluation_strategy: "steps"
  eval_steps: 2  # Every 2 rounds

federated:
  num_clients: 100
  client_resources:
    num_cpus: 0.25    # ← From 0.5
    num_gpus: 0

client:
  epochs: 1           # ← From 2
  batch_size: 64      # ← From 32
  learning_rate: 0.01
  
evaluation:
  num_test_samples: 5000  # ← From 10000
  num_workers: 8
```

**Expected results:**
- **Training time:** 1.5-2 hours (from 5.5 hours)
- **Peak RAM:** ~25-35GB (from ~30GB)
- **Speedup:** 3x-4x

---

## Next Steps

1. **Run cpu_profiler.py** during your current experiment to see actual CPU usage
2. **Share the output** - I can then tell you exactly which strategy to use
3. **Test with 2-round config** to validate speedup
4. **Scale to 10 rounds** with best settings

```bash
# Right now on your VM:
python cpu_profiler.py
# Let it sample for 5-10 minutes during active training
# Ctrl+C when done
# Share the analysis output
```

Once I see your CPU profile, I can give you the exact tuning parameters to use! 🎯
