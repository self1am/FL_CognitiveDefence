# Production Experiments Architecture & Flow Diagrams

## 🏗️ System Architecture (100 Clients on 64GB Instance)

```
┌─────────────────────────────────────────────────────────────────┐
│                      GCP Instance (64GB, 8vCPU)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
         ┌──────▼────────┐ ┌─▼────────┐ ┌─▼────────────┐
         │ Orchestration │ │  Server  │ │   Clients    │
         │   Process     │ │ Process  │ │  (100 total) │
         └──────┬────────┘ └─┬────────┘ └─┬────────────┘
                │             │             │
        Manages CLI      Aggregates   8 Concurrent
        Configs &        Updates &    (Load Balanced)
        Resource         Evaluates    
        Monitor          Model
                │             │             │
        ┌───────┴────────┬────┴────┬───────┴────────┐
        │                │         │                │
     Memory          Memory    Memory             Memory
     3-4 GB          2-3 GB    40-45 GB          5-10 GB
     Python       Aggregation  (5.6 GB/client)  Data/Buffers
     + Orchest.   + Testing    × 8 clients
                   Model
                │                │             │                │
        ├─ RAM: 5-10 GB    ├─ RAM: 5 GB    ├─ RAM: 45 GB    ├─ RAM: 5 GB
        ├─ CPU: 0.5 vCPU   ├─ CPU: 1 vCPU  ├─ CPU: 6 vCPU   ├─ Disk: I/O
        └─ Disk: Minimal   └─ I/O: ~50MB/s └─ I/O: ~150MB/s └─ Network: ~50Mbps
```

## 🔄 Experiment Execution Flow

```
┌────────────────────────────────────────────────────────────────┐
│ START: run_production_experiments.sh OR experiment_runner      │
└────────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
        ┌──────────────────┐    ┌─────────────────┐
        │  Single Exp      │    │  Multiple Exps  │
        │  Mode (-c)       │    │  Mode (--all)   │
        └────┬─────────────┘    └────┬────────────┘
             │                       │
             └────────────┬──────────┘
                         ▼
            ┌──────────────────────────┐
            │  Load Configuration YAML │
            │  (Experiment params)     │
            └────────┬─────────────────┘
                     ▼
            ┌──────────────────────────┐
            │  Initialize Logging      │
            │  Create log directory    │
            └────────┬─────────────────┘
                     ▼
            ┌──────────────────────────┐
            │  Setup Deterministic Env │
            │  (Seeds, Device)         │
            └────────┬─────────────────┘
                     ▼
         ┌──────────────────────────────┐
         │  START SERVER (subprocess)   │
         │  ├─ Load model               │
         │  ├─ Load test data           │
         │  └─ Create aggregation       │
         │     strategy                 │
         └────────┬─────────────────────┘
                  ▼
        ┌─────────────────────────────┐
        │ CREATE CLIENT ORCHESTRATOR   │
        │ (Resource manager)           │
        └────────┬────────────────────┘
                 ▼
      ┌────────────────────────────────┐
      │ BATCH SPAWN CLIENTS            │
      │ (8 at a time)                  │
      │                                │
      │ Batch 1: Clients 0-7           │
      │ Batch 2: Clients 8-15          │
      │ ...                            │
      │ Batch 13: Clients 92-99        │
      └────────┬─────────────────────┘
               ▼
      ┌─────────────────────────────────┐
      │ START RESOURCE MONITORING       │
      │ (CPU, Memory, Processes)        │
      └────────┬────────────────────────┘
               ▼
        ┌─────────────────────────┐
        │ FEDERATED LEARNING LOOP │     For each round (40-50):
        │                         │     
        │ ┌─────────────────────┐ │     1. Server samples clients
        │ │ Round 1             │ │        (configurable %)
        │ ├─ Clients train      │ │     
        │ ├─ Send updates       │ │     2. Clients perform:
        │ ├─ Server aggregates  │ │        - Load parameters
        │ ├─ Server evaluates   │ │        - Train locally
        │ └─ Log metrics        │ │        - Apply attacks (if)
        │                       │ │        - Send gradients
        │ ┌─────────────────────┐ │     
        │ │ Round 2             │ │     3. Server:
        │ └─────────────────────┘ │        - Detect anomalies
        │         ...             │        - Aggregate updates
        │ ┌─────────────────────┐ │        - Update model
        │ │ Round 40-50         │ │        - Evaluate on test set
        │ └─────────────────────┘ │        - Log metrics
        └──────┬──────────────────┘
               ▼
        ┌──────────────────────┐
        │ Wait for Completion  │
        │ (Timeout: 30 min)    │
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ Collect Results      │
        │ (Logs, metrics)      │
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ Save Experiment Log  │
        │ (JSON format)        │
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ Archive Results      │
        │ (if --all mode)      │
        └──────┬───────────────┘
               ▼
        ┌──────────────────────┐
        │ Cleanup & Shutdown   │
        │ Kill processes       │
        └──────┬───────────────┘
               ▼
    ┌──────────────────────────────┐
    │ END: Print summary & exit    │
    │ Display total metrics        │
    │ Completion time: ~4-6 hours  │
    └──────────────────────────────┘
```

## 📊 Per-Round Timing Breakdown

```
           ┌─────────── One Round (~5 minutes) ───────────┐
           │                                               │
    ┌──────▼──────┐  ┌──────────────┐  ┌─────────────┐   
    │  Client     │  │ Communication│  │  Server     │
    │ Training    │  │ & Network    │  │ Evaluation  │
    │             │  │              │  │             │
    │ ~3.5 min    │  │ ~0.5 min     │  │ ~1 min      │
    │             │  │              │  │             │
    │ × 8         │  │ Aggregate    │  │ Centralized │
    │ clients     │  │ gradients    │  │ test set    │
    │ training    │  │ (Byzantine   │  │ evaluation  │
    │ in parallel │  │  detection)  │  │ & logging   │
    │             │  │              │  │             │
    └─────────────┘  └──────────────┘  └─────────────┘
```

## 🎯 Batch Spawning Timeline (100 clients, batch size 8)

```
Time  Batch 1  Batch 2  Batch 3  ... Batch 13
 0s   ████     
 2s   ████     ████
 4s   ████     ████     ████
 6s   ████     ████     ████     ████
      (8-14s all batches spawning in staggered manner)

~30s: All 100 clients connected to server
~2min: Clients synchronized and ready
~5min: First round begins
```

## 💾 Memory Timeline for 100 Clients

```
Memory Usage Over Experiment Duration

65 GB │                    ████████████████████████████
      │                   ██████████████████████████████
60 GB │              ██████████████████████████████████████
      │            ███████████████████████████████████████████
55 GB │          █████████████████████████████████████████████
      │       ███████ Client Spawning ███████████████████████
50 GB │      ████████Finalization████ Training & Evaluation
      │     ██████════════════════════════════════════════
45 GB │    ██
      │   ██
40 GB │  ██
      │ ██
35 GB │                                        (Cooling down)
      │
 0 min    10     20     30     40     50    150    160    170
```

## 🔧 Configuration Options Impact

```
                    Num Clients
                    (horizontal)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
      50 clients      100 clients     150 clients
        │                │                │
        │                │                │
      Memory         Memory            Memory
     ~30-35GB       ~60-65GB          ~90-95GB ❌
     ~2-3h/40r      ~4-5h/40r        Exceeds 64GB
        │                │                │
        ▼                ▼                ▼
      Safe          Optimal         Exceeds
      (Extra        (Perfect for      Capacity
       headroom)    64GB)
    
                  Batch Size
              (affects concurrency)
    
        ┌─────────────┬──────────┬──────────┐
        ▼             ▼          ▼          ▼
    Batch=4       Batch=6    Batch=8    Batch=10
    Slowest       Safe       Optimal    Risky
    (~40GB mem)   (~50GB)    (~60GB)    (~70GB) ❌
    (~8h/50r)     (~6h/50r)  (~5h/50r)
```

## 📈 Attack Rate vs Defence Effectiveness

```
Attack Rate    0%    10%    20%    35%    50%
│              
│
Cognitive    ╔════════════════════════╗
Defence      ║ ███████████████████ ║  93-96% accuracy
             ║ ███████████████████ ║  0.08-0.12 loss
             ║ ███████████████████ ║
             ╚════════════════════════╝
│
Krum         ╔════════════════════╗
             ║ █████████████████ ║  88-92% accuracy
             ║ █████████████████ ║
             ╚════════════════════╝
│
Trimmed      ╔════════════════════╗
Mean         ║ █████████████████ ║  85-90% accuracy
             ╚════════════════════╝
│
No Defence   ╔═════════════════════════════════╗
             ║ ██ ║  40-70% accuracy (degrades)
             ║ ██ ║  1.0-1.5 loss
             ╚═════════════════════════════════╝
```

## 🎬 Running Multiple Experiments (Campaign)

```
Experiment Campaign Timeline

Day 1:
├─ 00:00 - Start Exp 1: Cognitive Defence (100 clients, 40r)
│         └─ 04:30 - Completed ✓
│         └─ 05:00 - Analyze & backup
│
├─ 05:30 - Start Exp 2: Adaptive Attacks (100 clients, 50r)
│         └─ 11:30 - Completed ✓
│         └─ 12:00 - Analyze & backup
│
├─ 12:30 - Start Exp 3: Krum Defence (100 clients, 40r)
│         └─ 16:30 - Completed ✓
│         └─ 17:00 - Final analysis

Total Campaign Duration: ~17 hours
```

## 🚦 Monitoring Dashboard Layout

```
┌────────────────────────────────────────────────┐
│  FL Experiment Monitoring Dashboard            │
├────────────────────────────────────────────────┤
│                                                │
│  System Resources          │  Experiment Progress
│  ──────────────────        │  ──────────────────
│  Memory: ████████ 57/64GB  │  Round: 28/40
│  CPU:    ██████████ 85%    │  Accuracy: ↗ 92.3%
│  Disk:   ████ 45/100GB     │  Loss:     ↘ 0.095
│  Network: ████ 35 Mbps     │  
│                            │  Client Status
│  Processes                 │  ──────────────
│  ──────────                │  Connected: 95/100
│  Python:       89          │  Training:   8
│  Server:       1           │  Idle:      87
│  Clients:      88          │  Failed:     5
│
│  Top 3 Memory Hogs:
│  ─────────────────
│  1. client_runner_45  4.2 GB
│  2. client_runner_32  4.1 GB
│  3. server_process    2.8 GB
│
└────────────────────────────────────────────────┘
```

## 📊 Result Analysis Workflow

```
Experiment Completed
        │
        ▼
┌──────────────────────┐
│ Collect Log Files    │
│ (JSON format)        │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ analyze_experiments  │
│ .py runs:            │
│                      │
│ 1. Load all logs     │
│ 2. Compute metrics   │
│ 3. Generate report   │
│ 4. Export CSV        │
└──────────┬───────────┘
           ▼
      3 Outputs:
      ├─ experiment_analysis_report.txt
      ├─ experiment_analysis.csv
      └─ Console summary
           │
           ▼
    ┌──────────────────┐
    │ Visualize Results│
    │ (Python script)  │
    │ → Charts & plots │
    └──────┬───────────┘
           ▼
    ┌────────────────────────┐
    │ Archive & Backup       │
    │ → tar.gz + cloud store │
    └────────────────────────┘
```

---

**These diagrams show:**
- System architecture and resource allocation
- Complete execution flow from start to finish
- Timing breakdowns for optimization
- Memory usage patterns
- Configuration impact on performance
- Multi-experiment campaign timeline
- Monitoring and analysis workflow

