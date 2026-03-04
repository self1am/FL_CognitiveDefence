#!/bin/bash
# Run multiple FL experiments in parallel using tmux
# Usage: bash run_parallel_experiments.sh [num_experiments]

set -e

NUM_EXPERIMENTS=${1:-2}

echo "=================================="
echo "Parallel FL Experiment Runner"
echo "=================================="
echo "Starting $NUM_EXPERIMENTS experiments in parallel..."
echo ""

# Check if tmux session exists, create if not
if ! tmux has-session -t fl_parallel 2>/dev/null; then
    tmux new-session -d -s fl_parallel -x 250 -y 50
    echo "✓ Created tmux session: fl_parallel"
fi

case $NUM_EXPERIMENTS in
    2)
        echo "Starting 2 parallel experiments (50 clients each)..."
        echo "Each will use 4 vCPU + 3GB RAM"
        echo ""
        
        # Window 1: Experiment A
        tmux new-window -t fl_parallel -n exp_a
        tmux send-keys -t fl_parallel:exp_a "echo '=== Experiment A ===' && python run_server_with_eval.py --config experiments/configs/baseline_50_clients_parallel_a.yaml" Enter
        echo "✓ Started Experiment A in window 'exp_a'"
        
        # Window 2: Experiment B
        tmux new-window -t fl_parallel -n exp_b
        tmux send-keys -t fl_parallel:exp_b "echo '=== Experiment B ===' && python run_server_with_eval.py --config experiments/configs/baseline_50_clients_parallel_b.yaml" Enter
        echo "✓ Started Experiment B in window 'exp_b'"
        
        echo ""
        echo "Expected timing:"
        echo "  - Round time: ~13 min each"
        echo "  - 50 rounds: ~11 hours per experiment"
        echo "  - Both complete in parallel: ~11 hours total (vs 22 hours sequential)"
        echo ""
        echo "Monitor with:"
        echo "  tmux attach -t fl_parallel"
        echo ""
        ;;
        
    3)
        echo "Starting 3 parallel experiments (35 clients each)..."
        echo "Each will use 2.6 vCPU + 2GB RAM"
        echo "(May see CPU contention on 8 vCPU machine)"
        echo ""
        
        for i in 1 2 3; do
            tmux new-window -t fl_parallel -n exp_$i
            echo "✓ Created window for experiment $i"
        done
        
        echo ""
        echo "Note: For 3 experiments, you'll need to create additional configs:"
        echo "  - baseline_35_clients_parallel_a.yaml (35 clients, 2.6 vCPU)"
        echo "  - baseline_35_clients_parallel_b.yaml (35 clients, 2.6 vCPU)"
        echo "  - baseline_35_clients_parallel_c.yaml (35 clients, 2.6 vCPU)"
        echo ""
        ;;
        
    *)
        echo "Usage: bash run_parallel_experiments.sh [2|3]"
        echo "  2 - Run 2 experiments with 50 clients each (recommended)"
        echo "  3 - Run 3 experiments with 35 clients each (experimental)"
        exit 1
        ;;
esac

echo "=================================="
echo "All experiments started!"
echo "=================================="
