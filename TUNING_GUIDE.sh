#!/usr/bin/env bash
# Test different parallelism settings to find optimal speed
# Run this to benchmark various num_cpus values

echo "===================================================="
echo "FL EXPERIMENT SPEED TUNING"
echo "===================================================="
echo ""
echo "STEP 1: Identify Current Bottleneck"
echo "Run in terminal 1: python run_server_with_eval.py --config baseline_100_clients.yaml"
echo "Run in terminal 2: python cpu_profiler.py"
echo ""
echo "Watch for:"
echo "  - If CPU < 50%: You have parallelism headroom"
echo "  - If CPU 50-90%: Good utilization"  
echo "  - If CPU > 90%: CPU is maxed out"
echo ""
echo "===================================================="
echo "STEP 2: Calculate Speedup Options"
echo "===================================================="
echo ""
echo "Your current config:"
echo "  - Clients: 100"
echo "  - num_cpus per client: 0.5"
echo "  - Max parallel: 8 cores ÷ 0.5 = 16 clients"
echo "  - Round time: ~30 minutes"
echo ""
echo "Option A: Reduce num_cpus to 0.25"
echo "  - Max parallel: 8 ÷ 0.25 = 32 clients"
echo "  - 100 clients in 4 batches (instead of 6.25)"
echo "  - Expected speedup: ~33% faster (~20 min/round)"
echo ""
echo "Option B: Reduce num_cpus to 0.125"
echo "  - Max parallel: 8 ÷ 0.125 = 64 clients"
echo "  - 100 clients in 2 batches"
echo "  - Expected speedup: ~60% faster (~12 min/round)"
echo "  - Risk: Each client gets thin CPU slice, might bottleneck"
echo ""
echo "Option C: Double clients, reduce num_cpus to 0.25"
echo "  - 200 clients, 0.25 num_cpus"
echo "  - Max parallel: 32 clients"
echo "  - 200 clients in 7 batches"
echo "  - Expected: Same time but 2x the work done"
echo ""
echo "===================================================="
echo "STEP 3: Quick Test with Modified Config"
echo "===================================================="
echo ""
echo "Create a test config (baseline_100_clients_fast.yaml):"
echo ""
cat << 'EOF'
# Copy from baseline_100_clients.yaml but change:

# CHANGE THIS:
server:
  num_rounds: 2  # Just 2 rounds for testing
  aggregation_strategy: "fedavg"
  
federated:
  num_clients: 100
  client_resources:
    num_cpus: 0.25  # <-- CHANGE FROM 0.5
    num_gpus: 0

# Also consider:
client:
  epochs: 1  # Faster training per client
  batch_size: 32  # Smaller batches
EOF
echo ""
echo "Then run:"
echo "  time python run_server_with_eval.py --config baseline_100_clients_fast.yaml"
echo ""
echo "Compare:"
echo "  - Original (0.5): ~30min for 2 rounds"
echo "  - Test (0.25): ??? for 2 rounds"
echo ""
echo "===================================================="
echo "STEP 4: Production Tuning"
echo "===================================================="
echo ""
echo "If testing shows improvement, use in full config:"
echo "  1. Update baseline_100_clients.yaml with new num_cpus"
echo "  2. Re-run with full 10 rounds"
echo "  3. Measure actual peak RAM (might increase)"
echo "  4. Finalize and commit config"
echo ""
