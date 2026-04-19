# 📚 Production FL Experiments - Complete Documentation Index

## 🎯 Where to Start?

Choose based on your needs:

### ⚡ **Want to Run Experiments RIGHT NOW?** (5-10 minutes)
→ Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
```bash
# Just run this:
source ~/.fl_optimization.sh
./scripts/run_production_experiments.sh --all
```

### 🏗️ **Want to Understand the Architecture?** (20 minutes)
→ Read: [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)
- System layout for 100 clients on 64GB
- Execution flow diagrams
- Memory and timing breakdowns

### 📋 **Want Step-by-Step Instructions?** (30-60 minutes)
→ Read: [EXECUTION_CHECKLIST.md](EXECUTION_CHECKLIST.md)
- Pre-experiment setup checklist
- Real-time monitoring commands
- Post-analysis workflow
- Troubleshooting guide

### 📖 **Want Complete Technical Details?** (1-2 hours)
→ Read: [PRODUCTION_EXPERIMENT_GUIDE.md](PRODUCTION_EXPERIMENT_GUIDE.md)
- Hardware requirements breakdown
- Resource estimation formulas
- Advanced configuration options
- Performance optimization tips

### 📝 **Just Want Summary of Changes?** (5 minutes)
→ Read: [SETUP_SUMMARY.md](SETUP_SUMMARY.md)
- What was created
- Quick start
- Expected results
- Common issues

---

## 📑 Documentation Files Reference

| File | Purpose | Read Time | Best For |
|------|---------|-----------|----------|
| **QUICK_REFERENCE.md** ⭐ | Commands, cheatsheet, quick tips | 5 min | Getting started ASAP |
| **SETUP_SUMMARY.md** | What was created, overview | 5 min | Understanding scope |
| **ARCHITECTURE_DIAGRAMS.md** | Visual system design | 20 min | Understanding architecture |
| **EXECUTION_CHECKLIST.md** | Complete step-by-step guide | 30-60 min | Detailed instructions |
| **PRODUCTION_EXPERIMENT_GUIDE.md** | Comprehensive technical guide | 1-2 hours | Deep understanding |
| **README.md** | Original project documentation | Variable | Project overview |
| **CENTRALIZED_EVAL_GUIDE.md** | Evaluation metrics explained | 20 min | Understanding results |
| **ANOMALY_SCORING_EXPLAINED.md** | Cognitive defence mechanism | 30 min | Defence details |

---

## 🛠️ Scripts & Configuration Files

### Automation Scripts

```
scripts/
├─ run_production_experiments.sh
│  └─ Automates running 1 or many experiments
│     • Usage: ./scripts/run_production_experiments.sh --all
│     • Features: Logging, cleanup, archiving, monitoring
│     • Duration: 4-20 hours (depending on configs)
│
└─ optimize_gcp_instance.sh
   └─ System optimization (one-time)
      • TCP settings, file descriptors, CPU governor
      • Usage: ./scripts/optimize_gcp_instance.sh
      • Duration: 2-3 minutes
```

### Python Scripts

```
analyze_experiments.py
├─ Post-experiment analysis
├─ Generates: Text report, CSV, console summary
├─ Usage: python analyze_experiments.py
└─ Output: experiment_analysis_report.txt, experiment_analysis.csv
```

### Configuration Files

```
experiments/configs/
├─ production_100_clients_cognitive.yaml
│  └─ 100 clients × 40 rounds (~4 hours)
│     • 20% attack rate (label flip + gradient noise)
│     • Cognitive Defence mechanism
│     • Recommended: First experiment to run
│
├─ production_100_clients_adaptive.yaml
│  └─ 100 clients × 50 rounds (~6-8 hours)
│     • All 4 adaptive attack types
│     • Cognitive Defence with anomaly detection
│     • Most comprehensive scenario
│
└─ production_100_clients_multidefence.yaml
   └─ 100 clients × 40 rounds (~4 hours)
      • Template for comparing defences
      • Run 3 times with different defence.strategy
      • For Krum, Trimmed Mean, Cognitive Defence
```

---

## 🚀 Quick Command Reference

### Setup (One-Time)
```bash
# Clone project
git clone <repo> && cd FL_CognitiveDefence

# Optimize system
./scripts/optimize_gcp_instance.sh

# Setup environment
python3 -m venv fl_env
source ~/.fl_optimization.sh
source fl_env/bin/activate

# Install dependencies
pip install -r requirements.txt -e .
```

### Run Experiments
```bash
# Single experiment (~4 hours)
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml

# All experiments in sequence (~12-15 hours)
./scripts/run_production_experiments.sh --all

# Monitor resources (in separate terminal)
./scripts/run_production_experiments.sh --monitor
```

### Monitor During Execution
```bash
# System resources
watch -n 2 'free -h; uptime; ps aux | grep python | wc -l'

# Live experiment logs
tail -f logs/*_complete.json

# CPU usage
top -c -u $USER
```

### Post-Experiment Analysis
```bash
# Generate analysis report
python analyze_experiments.py

# View results
cat experiment_analysis_report.txt

# Visualize
python experiments/visualize_results.py --config <config_file>

# Backup results
tar -czf results_$(date +%Y%m%d).tar.gz logs/ experiments/results/
```

---

## 📊 Expected Results

### Performance Benchmarks (100 clients, 40 rounds)

| Scenario | Final Accuracy | Final Loss | Duration |
|----------|----------------|-----------|----------|
| Baseline (no attack) | 97-99% | 0.05-0.08 | 4h |
| Cognitive Defence (20% attack) | 92-96% | 0.08-0.12 | 4h |
| No Defence (20% attack) | 65-70% | 1.2-1.5 | 4h |
| Adaptive Attacks | 88-94% | 0.10-0.15 | 6-8h |

### Resource Usage (100 clients, batch_size=8)

| Resource | Usage | Peak |
|----------|-------|------|
| Memory | 55-60 GB | 62 GB |
| CPU | 1-7 vCPU | 8 vCPU |
| Disk I/O | 50-150 MB/s | 300 MB/s |
| Network | 10-50 Mbps | 100 Mbps |

---

## 🎯 Recommended Experiment Sequence

### Quick Path (8-10 hours)
1. **Cognitive Defence** - 100 clients, 40 rounds (~4h)
2. **Adaptive Attacks** - 100 clients, 50 rounds (~6h)

### Comprehensive Path (16-20 hours)
1. **Cognitive Defence** - 100 clients, 40 rounds (~4h)
2. **Krum Defence** - 100 clients, 40 rounds (~4h)
3. **Trimmed Mean** - 100 clients, 40 rounds (~4h)
4. **Adaptive Attacks** - 100 clients, 50 rounds (~6h)

### Validation Path (4-5 hours, test setup)
1. **Single Cognitive** - 100 clients, 40 rounds
   - Verify system works
   - Check resource usage
   - Validate results
   - Then proceed to full campaign

---

## ⚙️ Configuration Customization Examples

### To Run Fewer Clients (reduce memory)
```yaml
# Edit production_100_clients_cognitive.yaml
orchestration:
  num_clients: 50        # Instead of 100
  batch_size: 4          # Instead of 8
  max_memory_mb: 35000   # Instead of 58000
```

### To Run More Rounds (longer convergence)
```yaml
experiment:
  num_rounds: 50         # Instead of 40
```

### To Test Higher Attack Rate
```yaml
attacks:
  - attack_type: "label_flip"
    target_clients: [0, 1, 2, ..., 34]  # 35 clients (35% attack rate)
```

### To Compare Defence Mechanisms
```yaml
# Run 3 times with different strategies:
defence:
  strategy: "cognitive_defence"   # First run
  strategy: "krum"                # Second run
  strategy: "trimmed_mean"        # Third run
```

---

## 📱 Real-Time Monitoring Setup

### Terminal 1: Run Experiment
```bash
cd FL_CognitiveDefence
source ~/.fl_optimization.sh
source fl_env/bin/activate
python -m src.orchestration.experiment_runner \
  --config experiments/configs/production_100_clients_cognitive.yaml
```

### Terminal 2: Monitor Resources
```bash
watch -n 2 'echo "=== Memory ==="; free -h | grep Mem; \
            echo "=== CPU ==="; uptime; \
            echo "=== Processes ==="; \
            ps aux | grep "python" | grep -v grep | wc -l'
```

### Terminal 3: Watch Logs
```bash
cd FL_CognitiveDefence
tail -f logs/*_complete.json | \
  grep -o '"centralized_accuracy":[^}]*' | tail -1
```

---

## 🔧 Troubleshooting Quick Lookup

| Problem | Solution | Read More |
|---------|----------|-----------|
| OOM Error | Reduce batch_size to 6 | [EXECUTION_CHECKLIST.md](EXECUTION_CHECKLIST.md#issue-cuda-out-of-memory) |
| Clients Disconnect | Increase timeout to 2400s | [EXECUTION_CHECKLIST.md](EXECUTION_CHECKLIST.md#issue-clients-disconnected) |
| Disk Full | Delete old logs | [EXECUTION_CHECKLIST.md](EXECUTION_CHECKLIST.md#issue-disk-space-low) |
| Experiment Hangs | Check resources with `top` | [PRODUCTION_EXPERIMENT_GUIDE.md](PRODUCTION_EXPERIMENT_GUIDE.md#troubleshooting) |
| Wrong Results | Verify attack/defence config | [ANOMALY_SCORING_EXPLAINED.md](ANOMALY_SCORING_EXPLAINED.md) |

---

## 💾 File Organization

```
FL_CognitiveDefence/
│
├─📚 Documentation (READ FIRST)
│  ├─ QUICK_REFERENCE.md ⭐ (Start here!)
│  ├─ SETUP_SUMMARY.md
│  ├─ EXECUTION_CHECKLIST.md
│  ├─ PRODUCTION_EXPERIMENT_GUIDE.md
│  ├─ ARCHITECTURE_DIAGRAMS.md
│  ├─ README.md
│  ├─ CENTRALIZED_EVAL_GUIDE.md
│  └─ ANOMALY_SCORING_EXPLAINED.md
│
├─🛠️ Scripts (AUTOMATION)
│  └─ scripts/
│     ├─ run_production_experiments.sh
│     └─ optimize_gcp_instance.sh
│
├─⚙️ Configurations (EXPERIMENTS)
│  └─ experiments/configs/
│     ├─ production_100_clients_cognitive.yaml
│     ├─ production_100_clients_adaptive.yaml
│     └─ production_100_clients_multidefence.yaml
│
├─📊 Analysis (POST-EXPERIMENT)
│  ├─ analyze_experiments.py
│  └─ experiments/visualize_results.py
│
├─📝 Logs (RESULTS)
│  └─ logs/
│     └─ (Generated during experiments)
│
└─📦 Source Code (EXISTING)
   └─ src/
      └─ (Orchestration, models, attacks, defences, etc.)
```

---

## 🎬 Recommended Learning Path

### For Quick Start (15 minutes total)
1. Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (5 min)
2. Run: `./scripts/optimize_gcp_instance.sh` (2 min)
3. Run: Single experiment (observe for 5 min to ensure it works)

### For Complete Understanding (2-3 hours total)
1. Read: [SETUP_SUMMARY.md](SETUP_SUMMARY.md) (5 min)
2. Read: [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) (20 min)
3. Read: [EXECUTION_CHECKLIST.md](EXECUTION_CHECKLIST.md) (30-45 min)
4. Run: Full experiment campaign (15+ hours over time)
5. Read: [ANOMALY_SCORING_EXPLAINED.md](ANOMALY_SCORING_EXPLAINED.md) (30 min)

### For Expert Mastery (4+ hours)
1. Complete "Complete Understanding" path
2. Read: [PRODUCTION_EXPERIMENT_GUIDE.md](PRODUCTION_EXPERIMENT_GUIDE.md) (60-90 min)
3. Customize configurations for specific scenarios
4. Run multiple experiment campaigns
5. Master troubleshooting and optimization

---

## ✅ Pre-Launch Checklist

- [ ] Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- [ ] GCP instance is running (64GB, 8vCPU)
- [ ] SSH access configured
- [ ] Project cloned
- [ ] System optimized: `./scripts/optimize_gcp_instance.sh`
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip install -r requirements.txt -e .`
- [ ] Configuration reviewed
- [ ] First experiment selected
- [ ] Monitoring setup planned
- [ ] Backup strategy planned

---

## 🚀 Ready to Start?

### **Option 1: Fastest Start** (4 hours)
```bash
./scripts/run_production_experiments.sh -c \
  experiments/configs/production_100_clients_cognitive.yaml
```

### **Option 2: Automated Full Campaign** (12-15 hours)
```bash
./scripts/run_production_experiments.sh --all
```

### **Option 3: Step-by-Step** (Follow EXECUTION_CHECKLIST.md)
```bash
# See detailed instructions in EXECUTION_CHECKLIST.md
```

---

**Choose your starting point above and dive in! 🎯**

For any questions, refer to the [Troubleshooting](#troubleshooting-quick-lookup) section or the comprehensive guides linked throughout this document.

