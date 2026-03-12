#!/bin/bash
# Quick test commands for Phase 1 optimizations
# Run these in sequence to validate the improvements

set -e  # Exit on error

echo "============================================================"
echo "Phase 1 Optimization Test Suite"
echo "============================================================"
echo ""

# Activate environment
source /Users/hanafemira/development/FL_CognitiveDefence/fl_env/bin/activate

# Test 1: Validation test (1 minute)
echo "Test 1: Running validation tests..."
python test_phase1_optimizations.py
echo "✅ Test 1 complete"
echo ""

# Test 2: Quick smoke test with 10 rounds (5-10 minutes)
echo "Test 2: Running quick smoke test (10 rounds)..."
python experiments/scripts/run_single_experiment.py \
    --config experiments/configs/static_attacks_cognitive_defence.yaml \
    --output-dir results/phase1_quick_test \
    --seed 42 2>&1 | tee results/phase1_quick_test.log

echo "✅ Test 2 complete"
echo ""

# Check results
echo "Test 2 Results:"
grep "CENTRALIZED EVALUATION" results/phase1_quick_test.log | tail -5
echo ""

# Test 3: Full static attack test (30-45 minutes) - OPTIONAL
read -p "Run full 30-round test? This will take 30-45 minutes. (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Test 3: Running full static attack test (30 rounds)..."
    python experiments/scripts/run_single_experiment.py \
        --config experiments/configs/static_attacks_cognitive_defence.yaml \
        --output-dir results/phase1_static_full \
        --seed 42 2>&1 | tee results/phase1_static_full.log
    
    echo "✅ Test 3 complete"
    echo ""
    
    # Compare to baseline
    echo "============================================================"
    echo "Comparison: Baseline vs Phase 1 Optimized"
    echo "============================================================"
    
    echo ""
    echo "BASELINE (from existing logs):"
    grep "Round.*CENTRALIZED EVALUATION.*Accuracy:" important_results/baseline/static_label_flip_cognitive_defence.log | tail -3 || echo "Baseline logs not found"
    
    echo ""
    echo "PHASE 1 OPTIMIZED:"
    grep "Round.*CENTRALIZED EVALUATION.*Accuracy:" results/phase1_static_full.log | tail -3
    
    echo ""
    echo "Multi-Krum Isolation Stats (first 5 rounds):"
    grep "Multi-Krum:" results/phase1_static_full.log | head -5
    
    echo ""
    echo "SAC Training Stability (first 10 updates):"
    grep "SAC update" results/phase1_static_full.log | head -10
fi

echo ""
echo "============================================================"
echo "Testing Complete!"
echo "============================================================"
echo ""
echo "Results saved in:"
echo "  - results/phase1_quick_test/"
echo "  - results/phase1_static_full/ (if full test was run)"
echo ""
echo "Next steps:"
echo "  1. Check accuracy improvement vs baseline"
echo "  2. Verify Multi-Krum is isolating ~40% of clients"
echo "  3. Confirm SAC updates are stable"
echo "  4. If successful (>70% accuracy), proceed to Phase 2"
echo ""
