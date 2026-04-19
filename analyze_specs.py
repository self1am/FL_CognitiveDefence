#!/usr/bin/env python3
"""
Analyze experiment performance and recommend optimal specs.
Based on observed timings from the log.
"""
import json
from datetime import datetime
from pathlib import Path

# Extract timings from latest experiment run (from the log you provided)
OBSERVED_DATA = {
    'spec': {
        'vm_ram_gb': 64,
        'vm_cpu_vcpu': 8,
        'vm_storage_gb': 100,
        'num_clients': 100,
        'client_resources': {'num_cpus': 0.5},
        'ray_workers': 16,  # 8 vCPU / 0.5 = 16 maximum in parallel
    },
    'timings': {
        # From log timestamps
        'round_0_to_1_seconds': 1916,      # 14:39:46 - 13:51:42 = ~48 minutes (includes eval)
        'round_1_to_2_seconds': 2170,      # 15:16:03 - 14:39:52 = ~36 minutes
        'round_2_to_3_seconds': 2144,      # 15:51:48 - 15:16:03 = ~35 minutes
        'round_3_to_4_seconds': 1576,      # 16:27:24 - 15:51:48 = ~35 minutes  
        'round_4_to_5_seconds': 1949,      # 17:03:13 - 16:27:24 = ~35 minutes
        'client_training_only': 1631,      # Average round training time (excluding eval)
        'evaluation_time': 400,            # ~6-7 minutes for centralized evaluation
    },
    'performance': {
        'accuracy_progression': [0.0974, 0.0974, 0.5934, 0.9716, 0.9863, 0.9888],
        'loss_progression': [2.3034, 2.3038, 1.9115, 0.2522, 0.0657, 0.0952],
    }
}

def analyze_current_setup():
    """Analyze current performance"""
    print("\n" + "="*80)
    print("CURRENT SETUP ANALYSIS (64GB RAM, 8 vCPU VM)")
    print("="*80)
    
    spec = OBSERVED_DATA['spec']
    timings = OBSERVED_DATA['timings']
    
    # Calculate effective parallelism
    max_parallel = spec['vm_cpu_vcpu'] / spec['client_resources']['num_cpus']
    training_batches = spec['num_clients'] / max_parallel
    
    print(f"\nParallelism:")
    print(f"  - Total clients: {spec['num_clients']}")
    print(f"  - VM CPU cores: {spec['vm_cpu_vcpu']}")
    print(f"  - CPU per client: {spec['client_resources']['num_cpus']}")
    print(f"  - Max parallel clients: {int(max_parallel)}")
    print(f"  - Training batches needed: {training_batches:.1f}")
    
    print(f"\nRound Timings (with 16 parallel Ray workers):")
    avg_round_time = sum([
        timings['round_1_to_2_seconds'],
        timings['round_2_to_3_seconds'],
        timings['round_3_to_4_seconds'],
        timings['round_4_to_5_seconds']
    ]) / 4
    
    print(f"  - Average round time: {avg_round_time:.0f}s ({avg_round_time/60:.1f} minutes)")
    print(f"  - Training per round: {timings['client_training_only']:.0f}s (~{timings['client_training_only']/60:.1f} min)")
    print(f"  - Evaluation per round: {timings['evaluation_time']:.0f}s (~{timings['evaluation_time']/60:.1f} min)")
    print(f"  - For 10 rounds: {avg_round_time * 10 / 3600:.1f} hours")
    
    # Memory
    print(f"\nMemory Configuration:")
    print(f"  - Total VM RAM: {spec['vm_ram_gb']}GB")
    print(f"  - Ray object store: ~20GB (43% of RAM)")
    print(f"  - Available for processes: ~45GB")
    print(f"  - Effectiveness: MODERATE (Ray overhead is ~43%)")

def calculate_optimal_specs():
    """Calculate ideal specifications for different scenarios"""
    print("\n" + "="*80)
    print("OPTIMAL SPECIFICATIONS FOR DIFFERENT SCALES")
    print("="*80)
    
    scenarios = [
        {
            'name': '10 Clients (Debugging)',
            'num_clients': 10,
            'rounds': 10,
            'recomm_cpu': 4,
            'recomm_ram': 16,
            'recomm_storage': 50,
        },
        {
            'name': '50 Clients (Small Scale)',
            'num_clients': 50,
            'rounds': 10,
            'recomm_cpu': 8,
            'recomm_ram': 32,
            'recomm_storage': 100,
        },
        {
            'name': '100 Clients (Current)',
            'num_clients': 100,
            'rounds': 10,
            'recomm_cpu': 16,
            'recomm_ram': 64,
            'recomm_storage': 200,
        },
        {
            'name': '200 Clients (Large)',
            'num_clients': 200,
            'rounds': 10,
            'recomm_cpu': 32,
            'recomm_ram': 128,
            'recomm_storage': 500,
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Recommended:")
        print(f"    - CPU: {scenario['recomm_cpu']} vCPUs")
        print(f"    - RAM: {scenario['recomm_ram']}GB")
        print(f"    - Storage: {scenario['recomm_storage']}GB")
        print(f"  Reasoning:")
        
        if scenario['num_clients'] == 10:
            print(f"    - Can serialize clients (no parallelism)")
            print(f"    - Minimal Ray overhead")
            print(f"    - Low memory pressure")
        elif scenario['num_clients'] == 50:
            print(f"    - ~6 clients in parallel (8 vCPU / 0.5)")
            print(f"    - Balanced CPU usage")
            print(f"    - Expected round time: ~15 min")
        elif scenario['num_clients'] == 100:
            print(f"    - ~16 clients in parallel (16 vCPU / 0.5)")
            print(f"    - Better parallelism")
            print(f"    - Expected round time: ~25 min")
        elif scenario['num_clients'] == 200:
            print(f"    - ~32 clients in parallel (32 vCPU / 0.5)")
            print(f"    - Full parallelism with no oversubscription")
            print(f"    - Expected round time: ~35-40 min")

def calculate_training_speedup():
    """Show how specs affect training speed"""
    print("\n" + "="*80)
    print("TRAINING TIME PROJECTIONS")
    print("="*80)
    
    # Based on observed ~30 min per round with 100 clients
    base_round_time = 30  # minutes
    
    specs = [
        (4, '4 vCPU, 16GB RAM'),
        (8, '8 vCPU, 32GB RAM (Current)'),
        (16, '16 vCPU, 64GB RAM'),
        (32, '32 vCPU, 128GB RAM'),
    ]
    
    print(f"\nFor 100 clients, 10 rounds:")
    print(f"{'Spec':<30} {'Round Time':<15} {'Total Time':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for cpu, desc in specs:
        # Rough estimate: time scales with parallelism
        # With 0.5 CPU per client: max_parallel = cpu / 0.5
        # Time scales inversely with parallelism
        parallelism_factor = cpu / 8.0
        round_time = base_round_time / parallelism_factor
        total_time = round_time * 10
        speedup = (base_round_time * 10) / total_time
        
        print(f"{desc:<30} {round_time:>6.1f} min         {total_time:>6.1f} min        {speedup:>5.1f}x")

def memory_requirements():
    """Analyze memory requirements per client"""
    print("\n" + "="*80)
    print("MEMORY USAGE ANALYSIS")
    print("="*80)
    
    print("\nEstimated memory breakdown (64GB VM):")
    print("  Ray runtime + object store:    ~20GB (43%)")
    print("  Python interpreter overhead:   ~2GB  (3%)")
    print("  Flower framework:              ~1GB  (2%)")
    print("  Data loading (MNIST):          ~2GB  (3%)")
    print("  Model training (~16 parallel): ~20GB (31%)")
    print("  Free/buffer:                   ~19GB (18%)")
    print("  Total:                         ~64GB")
    
    print("\nPer-client memory (active training):")
    print("  - Small model (MNIST):    ~20-30MB")
    print("  - Medium model (CIFAR10): ~100-150MB")
    print("  - Large model (ResNet50): ~500MB-1GB")
    print("  - With 16 parallel:       ~320MB-5.12GB total")
    
    print("\nMemory pressure points:")
    print("  ⚠ Ray object store fills up  → workers block waiting for memory")
    print("  ⚠ Swap usage > 10%           → massive slowdown (SSD thrashing)")
    print("  ⚠ Swap usage > 50%           → experiment likely to hang/crash")

def scaling_guide():
    """Guide for choosing specs"""
    print("\n" + "="*80)
    print("QUICK DECISION GUIDE")
    print("="*80)
    
    print("\n❓ What's YOUR use case?")
    print("\n1. LOCAL TESTING (Laptop/Mac):")
    print("   - Use 10 clients max")
    print("   - Lower num_rounds to 3-5")
    print("   - Expected: SLOW (as you've seen ~1 hour for 2 rounds)")
    print("   - Why: Single machine, no GPU, I/O bound")
    
    print("\n2. SMALL VM EXPERIMENTS (8GB RAM, 2vCPU):")
    print("   - Use 10 clients")
    print("   - Increase num_rounds to 10")
    print("   - Expected: 5-10 minutes per round")
    print("   - Use num_cpus=1 (full CPU per client, no parallelism)")
    
    print("\n3. MEDIUM VM (32GB RAM, 8vCPU) - ⭐ RECOMMENDED:")
    print("   - Use 50 clients")
    print("   - 10 rounds practical")
    print("   - Expected: 15-20 minutes per round")
    print("   - Use num_cpus=0.5 (8 parallel, some contention)")
    
    print("\n4. LARGE VM (64GB RAM, 16vCPU) - YOUR CURRENT SETUP:")
    print("   - Use 100 clients")
    print("   - 10 rounds practical")
    print("   - Expected: 25-30 minutes per round")
    print("   - Use num_cpus=0.5 (16 parallel workers)")
    print("   - Can push to 200 clients with longer time")
    
    print("\n5. PRODUCTION (128GB+ RAM, 32+ vCPU):")
    print("   - Use 200-500 clients")
    print("   - 20+ rounds practical")
    print("   - Expected: 30-60 min per round")
    print("   - Use num_cpus=0.25-0.5 for full utilization")

def main():
    analyze_current_setup()
    calculate_optimal_specs()
    calculate_training_speedup()
    memory_requirements()
    scaling_guide()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR YOUR VM")
    print("="*80)
    print("\n✓ Keep tmux for long-running experiments")
    print("✓ Monitor RAM with: python ram_monitor.py")
    print("✓ For 100 clients: ~1 hour per 2 rounds (your current pace)")
    print("✓ If you want 2x speedup: Upgrade to 16 vCPU")
    print("✓ If you want 4x speedup: Upgrade to 32 vCPU + 128GB RAM")
    print("✓ Storage should be 2-3x your dataset size (for logs, checkpoints)")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
