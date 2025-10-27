# How to Run Experiments with Centralized Evaluation

## 🎯 The Problem with Current Setup

The centralized evaluation function is not being called because Flower's default behavior only triggers `evaluate_fn` when clients don't have an `evaluate()` method. Since your `EnhancedFLClient` implements `evaluate()`, Flower prefers distributed evaluation.

## ✅ Solution: Two-Step Process

### **Option 1: Run Server and Clients Separately (RECOMMENDED)**

This gives you full control and clear centralized evaluation:

#### **Step 1: Start the Server (on VM)**
```bash
# On your server VM
python run_server_with_eval.py --config experiments/configs/attack_only.yaml
```

This will:
- Load the clean MNIST test set (10,000 samples)
- Start the Flower server with centralized evaluation enabled
- Wait for clients to connect

#### **Step 2: Start Clients (from your machine)**
```bash
# From your local machine
python -m src.clients.client_runner \
    --client-id 0 \
    --server-address 140.245.224.116:8080 \
    --experiment-name attack_only_no_defence \
    --seed 123 \
    --attack-type label_flip \
    --attack-intensity 0.1

# Repeat for other clients (1-9) with appropriate attack configs
```

Or use the orchestrator (but launch server separately first):
```bash
# Make sure server is already running on VM, then:
python -m src.clients.client_orchestrator --config experiments/configs/attack_only.yaml
```

### **Option 2: Quick Test with Makefile**

```bash
# On server VM:
make run-attack-only-server

# Then launch clients from your machine
```

## 📊 What You'll See

With the new server script, you'll see clear output like:

```
================================================================================
🎯 ROUND 1 - CENTRALIZED EVALUATION (Clean Test Set)
   Loss:     0.856234
   Accuracy: 0.7245 (7245/10000 correct)
================================================================================

================================================================================
🎯 ROUND 10 - CENTRALIZED EVALUATION (Clean Test Set)  
   Loss:     0.654321
   Accuracy: 0.7892 (7892/10000 correct)
================================================================================
```

## 🔍 Expected Results

### Attack-Only (No Defense)
- **Centralized Loss**: Should remain HIGH (0.5-0.8 range)
- **Centralized Accuracy**: Should be LOW (~75-80%)
- **Reason**: Poisoned model performs poorly on clean test data

### Baseline (No Attacks)
- **Centralized Loss**: Should be LOW (0.03-0.05)
- **Centralized Accuracy**: Should be HIGH (~98-99%)

### With Defense
- **Centralized Loss**: Should be MEDIUM (0.1-0.3)
- **Centralized Accuracy**: Should be IMPROVED (85-95%)
- **Reason**: Defense mitigates attack impact

## 🐛 Troubleshooting

### Issue: "Centralized evaluation not showing"

**Cause**: The current experiment runner starts server in a thread, but Flower might not be calling the evaluate_fn.

**Solution**: Use the standalone server script (`run_server_with_eval.py`)

### Issue: "Connection refused"

**Cause**: Server not started or firewall blocking

**Solution**: 
1. Ensure server VM is accessible
2. Check firewall rules for port 8080
3. Verify server address in config

### Issue: "Model parameters not loading"

**Cause**: Parameter mismatch between model and loaded weights

**Solution**: Script includes proper error handling and will show detailed error

## 📁 Files Modified/Created

1. **`run_server_with_eval.py`** - Standalone server with clear centralized evaluation
2. **`Makefile`** - Added `run-attack-only-server` target
3. **`src/orchestration/experiment_runner.py`** - Enhanced evaluation function
4. **`src/server/no_defence_server.py`** - Added evaluate_fn support
5. **`src/server/cognitive_server.py`** - Added evaluate_fn support

## 🚀 Complete Workflow

### Terminal 1 (Server VM):
```bash
cd /path/to/FL_CognitiveDefence
source fl_config/bin/activate
python run_server_with_eval.py --config experiments/configs/attack_only.yaml
```

### Terminal 2 (Local Machine):
```bash
# Option A: Launch all clients via orchestrator
# (Make sure to configure it to not start server)

# Option B: Use the existing experiment runner but manually start server first
make run-attack-only
```

## 📝 Key Differences

| Method | Centralized Eval | Distributed Eval | Server Control |
|--------|-----------------|------------------|----------------|
| **Old Setup** | ❌ Not working | ✅ Yes | Auto-started |
| **New run_server_with_eval.py** | ✅ **CLEAR** | ✅ Yes | Manual start |
| **Updated experiment_runner** | ⚠️ May not trigger | ✅ Yes | Auto-started |

## 🎓 Understanding the Metrics

```
DISTRIBUTED evaluation (what you saw before):
- Average of client-reported losses on their LOCAL test sets
- Benign clients: Loss ~0.02 (good performance on clean data)
- Attacked clients: Loss ~0.60 (poor performance on poisoned data)
- Average: ~0.04 ← MISLEADING because benign clients dominate!

CENTRALIZED evaluation (what you should use):
- Server evaluates global model on ONE clean test set
- Shows TRUE impact of attacks on the aggregated model
- Attack-only: Loss ~0.6-0.8 (model is poisoned)
- Baseline: Loss ~0.03-0.05 (model is clean)
```

---

**Next Steps:**
1. Run the server using `run_server_with_eval.py`
2. Launch clients (they will connect automatically)
3. Watch for the `🎯 CENTRALIZED EVALUATION` messages
4. Compare centralized loss across experiments (attack-only vs baseline vs with-defense)
