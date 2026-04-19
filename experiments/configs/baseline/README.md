# Baseline Experiment Configurations

## Standardized Parameters (Identical Across All Configs)

| Parameter | Value | Rationale |
|---|---|---|
| `seed` | 123 | Reproducibility |
| `num_rounds` | 30 | Sufficient for convergence and degradation trends |
| `num_clients` | 100 | Standard FL scale |
| `min_clients` | 20 | Minimum per-round participation |
| `malicious_clients` | 0–39 (40%) | 40% Byzantine — standard threat model |
| `num_cpus` (client) | 0.25 | 4 concurrent per CPU core |
| `num_cpus` (total) | 8 | Ray cluster resources |
| `epochs` | 1 | Single local epoch per round |
| `batch_size_client` | 64 | Client training batch size |
| `num_test_samples` | 10000 | Full MNIST test set |

## Experiment Matrix (26 Configs)

### 00 — Clean Baseline
| # | Config File | Attack | Defence |
|---|---|---|---|
| 0 | `00_clean_no_attack.yaml` | None | None (FedAvg) |

### 01 — Static Attacks (Label Flip, intensity=1.0)
| # | Config File | Attack | Defence |
|---|---|---|---|
| 1 | `01_static_label_flip_no_defence.yaml` | label_flip | None (FedAvg) |
| 2 | `01_static_label_flip_cognitive_defence.yaml` | label_flip | Cognitive (OODA+MAPE-K) |
| 3 | `01_static_label_flip_krum_defence.yaml` | label_flip | Multi-Krum (f=40) |
| 4 | `01_static_label_flip_trimmed_mean_defence.yaml` | label_flip | Trimmed Mean (β=0.2) |
| 5 | `01_static_label_flip_vert_defence.yaml` | label_flip | VERT (κ=5) |

### 02 — Adaptive Attacks: DnyOpt (Q-Learning RL, intensity=0.35)
| # | Config File | Attack | Defence |
|---|---|---|---|
| 6 | `02_adaptive_dny_opt_no_defence.yaml` | dny_opt | None (FedAvg) |
| 7 | `02_adaptive_dny_opt_cognitive_defence.yaml` | dny_opt | Cognitive (OODA+MAPE-K) |
| 8 | `02_adaptive_dny_opt_krum_defence.yaml` | dny_opt | Multi-Krum (f=40) |
| 9 | `02_adaptive_dny_opt_trimmed_mean_defence.yaml` | dny_opt | Trimmed Mean (β=0.2) |
| 10 | `02_adaptive_dny_opt_vert_defence.yaml` | dny_opt | VERT (κ=5) |

### 03 — Adaptive Attacks: StatOpt (Statistical Optimization, intensity=0.5)
| # | Config File | Attack | Defence |
|---|---|---|---|
| 11 | `03_adaptive_stat_opt_no_defence.yaml` | stat_opt | None (FedAvg) |
| 12 | `03_adaptive_stat_opt_cognitive_defence.yaml` | stat_opt | Cognitive (OODA+MAPE-K) |
| 13 | `03_adaptive_stat_opt_krum_defence.yaml` | stat_opt | Multi-Krum (f=40) |
| 14 | `03_adaptive_stat_opt_trimmed_mean_defence.yaml` | stat_opt | Trimmed Mean (β=0.2) |
| 15 | `03_adaptive_stat_opt_vert_defence.yaml` | stat_opt | VERT (κ=5) |

### 04 — Adaptive Attacks: Min-Max (Game-Theoretic, intensity=0.5)
| # | Config File | Attack | Defence |
|---|---|---|---|
| 16 | `04_adaptive_min_max_no_defence.yaml` | min_max | None (FedAvg) |
| 17 | `04_adaptive_min_max_cognitive_defence.yaml` | min_max | Cognitive (OODA+MAPE-K) |
| 18 | `04_adaptive_min_max_krum_defence.yaml` | min_max | Multi-Krum (f=40) |
| 19 | `04_adaptive_min_max_trimmed_mean_defence.yaml` | min_max | Trimmed Mean (β=0.2) |
| 20 | `04_adaptive_min_max_vert_defence.yaml` | min_max | VERT (κ=5) |

### 05 — Adaptive Attacks: Min-Sum (Game-Theoretic, intensity=0.5)
| # | Config File | Attack | Defence |
|---|---|---|---|
| 21 | `05_adaptive_min_sum_no_defence.yaml` | min_sum | None (FedAvg) |
| 22 | `05_adaptive_min_sum_cognitive_defence.yaml` | min_sum | Cognitive (OODA+MAPE-K) |
| 23 | `05_adaptive_min_sum_krum_defence.yaml` | min_sum | Multi-Krum (f=40) |
| 24 | `05_adaptive_min_sum_trimmed_mean_defence.yaml` | min_sum | Trimmed Mean (β=0.2) |
| 25 | `05_adaptive_min_sum_vert_defence.yaml` | min_sum | VERT (κ=5) |

## What Changed vs Old Configs

### Bugs Fixed in Existing Configs
1. **`static_attacks_no_defence.yaml`** — `intensity` changed from `0.5` → `1.0` (was unfairly easier than the defence configs)
2. **`static_attacks_vertical_defence.yaml`** — `strategy` changed from `"vertical"` → `"vert"` (code only matches `"vert"`; old config silently ran with **no defence at all**)
3. **`adaptive_attacks_vertical_defence.yaml`** — Same `"vertical"` → `"vert"` fix
4. **VERT config params** — Replaced wrong cognitive params (`anomaly_threshold`, `reputation_decay`) with correct VERT params (`kappa`, `history_size`, `projection_dim`, `learning_rate`, `min_history_rounds`)

### Standardisation Changes (Old → New Baseline Configs)
| Parameter | Old (Inconsistent) | New (Standardized) |
|---|---|---|
| `num_rounds` | 10 or 20 | **30** |
| `num_test_samples` | 5000 | **10000** (full MNIST test set) |
| `target_clients` | Some had 10 (10%), some had 40 (40%) | **40 (clients 0–39)** in all |
| `seed` | Some used 42, some 123 | **123** in all |
| `num_cpus` (client) | Some used 0.5 | **0.25** in all |
| Attack type per defence | Different attack per defence! | **Same attack for all 5 defences in each group** |

## Running Order

Run in this order (each group's no-defence experiment first to establish the attack baseline):

```bash
# Phase 1: Clean baseline (run first)
python run_server_with_eval.py --config experiments/configs/baseline/00_clean_no_attack.yaml

# Phase 2: Static attacks (5 experiments)
for defence in no_defence cognitive_defence krum_defence trimmed_mean_defence vert_defence; do
    python run_server_with_eval.py --config experiments/configs/baseline/01_static_label_flip_${defence}.yaml
done

# Phase 3: Adaptive attacks (20 experiments)
for attack_group in 02_adaptive_dny_opt 03_adaptive_stat_opt 04_adaptive_min_max 05_adaptive_min_sum; do
    for defence in no_defence cognitive_defence krum_defence trimmed_mean_defence vert_defence; do
        python run_server_with_eval.py --config experiments/configs/baseline/${attack_group}_${defence}.yaml
    done
done
```

## Expected Results Table (Fill In After Running)

| Attack | No Defence | Cognitive | Krum | Trimmed Mean | VERT |
|--------|-----------|-----------|------|-------------|------|
| No Attack | — | — | — | — | — |
| Label Flip | | | | | |
| DnyOpt | | | | | |
| StatOpt | | | | | |
| Min-Max | | | | | |
| Min-Sum | | | | | |
