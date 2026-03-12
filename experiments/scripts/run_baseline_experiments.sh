#!/bin/bash
# =============================================================================
# Run All Baseline Experiments
# Executes all 26 standardized configs sequentially with logging
# Usage: bash experiments/scripts/run_baseline_experiments.sh [--dry-run]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$PROJECT_DIR/experiments/configs/baseline"
LOG_DIR="$PROJECT_DIR/experiments/results/baseline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — no experiments will be executed ==="
fi

mkdir -p "$LOG_DIR"

# All configs in execution order
CONFIGS=(
    "00_clean_no_attack.yaml" // 
    "01_static_label_flip_no_defence.yaml" //
    "01_static_label_flip_cognitive_defence.yaml"
    "01_static_label_flip_krum_defence.yaml" //
    "01_static_label_flip_trimmed_mean_defence.yaml" //
    "01_static_label_flip_vert_defence.yaml" //
    "02_adaptive_dny_opt_no_defence.yaml" //
    "02_adaptive_dny_opt_cognitive_defence.yaml"
    "02_adaptive_dny_opt_krum_defence.yaml" //
    "02_adaptive_dny_opt_trimmed_mean_defence.yaml" //
    "02_adaptive_dny_opt_vert_defence.yaml" //
    "03_adaptive_stat_opt_no_defence.yaml" //
    "03_adaptive_stat_opt_cognitive_defence.yaml"
    "03_adaptive_stat_opt_krum_defence.yaml" //
    "03_adaptive_stat_opt_trimmed_mean_defence.yaml" //
    "03_adaptive_stat_opt_vert_defence.yaml" //
    "04_adaptive_min_max_no_defence.yaml" //
    "04_adaptive_min_max_cognitive_defence.yaml"
    "04_adaptive_min_max_krum_defence.yaml" //
    "04_adaptive_min_max_trimmed_mean_defence.yaml"
    "04_adaptive_min_max_vert_defence.yaml" //
    "05_adaptive_min_sum_no_defence.yaml" 
    "05_adaptive_min_sum_cognitive_defence.yaml"
    "05_adaptive_min_sum_krum_defence.yaml"
    "05_adaptive_min_sum_trimmed_mean_defence.yaml"
    "05_adaptive_min_sum_vert_defence.yaml"
)

TOTAL=${#CONFIGS[@]}
PASSED=0
FAILED=0
SKIPPED=0

echo "=============================================="
echo " Baseline Experiment Runner"
echo " Total experiments: $TOTAL"
echo " Config dir: $CONFIG_DIR"
echo " Log dir: $LOG_DIR"
echo " Started: $(date)"
echo "=============================================="
echo ""

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    EXP_NUM=$((i + 1))
    EXP_NAME="${CONFIG%.yaml}"
    LOG_FILE="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"

    echo "[$EXP_NUM/$TOTAL] Running: $CONFIG"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  → [DRY RUN] Would run: python run_server_with_eval.py --config $CONFIG_DIR/$CONFIG"
        echo "  → Log: $LOG_FILE"
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    fi

    # Check config exists
    if [[ ! -f "$CONFIG_DIR/$CONFIG" ]]; then
        echo "  → ERROR: Config file not found: $CONFIG_DIR/$CONFIG"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi

    START_TIME=$(date +%s)

    if python "$PROJECT_DIR/run_server_with_eval.py" --config "$CONFIG_DIR/$CONFIG" 2>&1 | tee "$LOG_FILE"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "  → PASSED in ${DURATION}s — Log: $LOG_FILE"
        PASSED=$((PASSED + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "  → FAILED after ${DURATION}s — Log: $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    echo ""

    # Brief cooldown between experiments to let Ray clean up
    sleep 5
done

echo "=============================================="
echo " Baseline Experiments Complete"
echo " Passed: $PASSED / $TOTAL"
echo " Failed: $FAILED / $TOTAL"
if [[ "$DRY_RUN" == true ]]; then
    echo " Skipped (dry run): $SKIPPED / $TOTAL"
fi
echo " Finished: $(date)"
echo "=============================================="
