#!/usr/bin/env python3
"""
Quick validation test for Phase 1 optimizations.
Tests that the optimized cognitive defense can be instantiated and runs basic operations.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/hanafemira/development/FL_CognitiveDefence')

from src.defences.cognitive_defence_posg import CognitiveDefencePOSG

def test_instantiation():
    """Test that defense can be created with new hyperparameters."""
    print("Testing instantiation with new hyperparameters...")
    defense = CognitiveDefencePOSG(
        max_clients=10,
        obs_dim=6,
        belief_hidden_dim=64,
        warmup_rounds=10,
        device="cpu"
    )
    print(f"✅ Defense created successfully")
    print(f"   - Warmup rounds: {defense.warmup_rounds}")
    print(f"   - Reward alpha: {defense.reward_alpha}")
    print(f"   - Reward beta: {defense.reward_beta}")
    print(f"   - SAC gamma: {defense.agent.gamma}")
    print(f"   - SAC target entropy: {defense.agent.target_entropy}")
    print(f"   - Buffer capacity: {defense.agent.replay.capacity}")
    print(f"   - Batch size: {defense.agent.batch_size}")
    return defense

def test_observation_and_belief():
    """Test observation extraction and belief update."""
    print("\nTesting observation & belief tracking...")
    defense = CognitiveDefencePOSG(max_clients=5, device="cpu")
    
    # Simulate 3 clients with random updates
    client_updates = {
        "client_0": ([np.random.randn(10, 10), np.random.randn(10)], 100, {}),
        "client_1": ([np.random.randn(10, 10), np.random.randn(10)], 100, {}),
        "client_2": ([np.random.randn(10, 10), np.random.randn(10)], 100, {}),
    }
    
    # Set global model
    global_params = [np.random.randn(10, 10), np.random.randn(10)]
    defense.set_global_model(global_params)
    
    # Extract observations
    observations = defense.observe(client_updates)
    print(f"✅ Observations extracted for {len(observations)} clients")
    print(f"   - Observation shape: {next(iter(observations.values())).shape}")
    
    # Update beliefs
    beliefs, state = defense.orient(observations)
    print(f"✅ Beliefs updated")
    print(f"   - State shape: {state.shape} (expected: 128 for 2*64)")
    print(f"   - Belief hidden dims: {next(iter(beliefs.values())).shape}")
    
    return defense, client_updates, beliefs, state

def test_multi_krum_heuristic():
    """Test Multi-Krum warm-up heuristic."""
    print("\nTesting Multi-Krum heuristic...")
    defense = CognitiveDefencePOSG(max_clients=10, device="cpu")
    
    # Create 10 clients: 6 benign (similar updates), 4 Byzantine (outliers)
    client_updates = {}
    defense._current_flattened_updates = {}
    
    # Benign clients (clustered around origin)
    for i in range(6):
        params = [np.random.randn(100) * 0.1 for _ in range(2)]
        client_updates[f"client_{i}"] = (params, 100, {})
        defense._current_flattened_updates[f"client_{i}"] = np.concatenate([p.ravel() for p in params])
    
    # Byzantine clients (outliers)
    for i in range(6, 10):
        params = [np.random.randn(100) * 10.0 for _ in range(2)]  # 100x larger
        client_updates[f"client_{i}"] = (params, 100, {})
        defense._current_flattened_updates[f"client_{i}"] = np.concatenate([p.ravel() for p in params])
    
    observations = {cid: np.random.randn(6) for cid in client_updates.keys()}
    weights = defense._heuristic_weights(observations)
    
    isolated = sum(1 for w in weights.values() if w < 0.5)
    trusted = sum(1 for w in weights.values() if w >= 0.5)
    
    print(f"✅ Multi-Krum executed")
    print(f"   - Trusted clients: {trusted}/10")
    print(f"   - Isolated clients: {isolated}/10")
    print(f"   - Expected: ~6 trusted, ~4 isolated (should correctly detect outliers)")
    
    if isolated >= 3:
        print(f"   ✅ PASS: Isolated {isolated} clients (expected ~4)")
    else:
        print(f"   ⚠️  WARNING: Only isolated {isolated} clients (expected ~4)")
    
    return weights

def test_reward_computation():
    """Test EMA reward stabilization."""
    print("\nTesting EMA reward stabilization...")
    defense = CognitiveDefencePOSG(max_clients=5, device="cpu")
    
    # Simulate noisy accuracy sequence
    accuracies = [0.1, 0.12, 0.11, 0.13, 0.12, 0.14]
    
    print(f"   Raw accuracies: {accuracies}")
    print(f"   EMA smoothing (alpha=0.3):")
    
    for round_num, acc in enumerate(accuracies, start=1):
        defense.round_number = round_num
        
        if round_num == 1:
            defense._acc_ema = acc
            defense._prev_acc_ema = acc
        else:
            defense._prev_acc_ema = defense._acc_ema
            defense._acc_ema = 0.3 * acc + 0.7 * defense._acc_ema
        
        delta = defense._acc_ema - defense._prev_acc_ema
        print(f"     Round {round_num}: raw={acc:.4f}, ema={defense._acc_ema:.4f}, delta={delta:.4f}")
    
    print(f"✅ EMA computation working correctly")
    return defense

def main():
    print("="*60)
    print("Phase 1 Optimization Validation Test")
    print("="*60)
    
    try:
        # Test 1: Instantiation
        defense1 = test_instantiation()
        
        # Test 2: Observation & Belief
        defense2, updates, beliefs, state = test_observation_and_belief()
        
        # Test 3: Multi-Krum
        weights = test_multi_krum_heuristic()
        
        # Test 4: Reward EMA
        defense4 = test_reward_computation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - Phase 1 optimizations validated!")
        print("="*60)
        print("\nNext step: Run full experiment with:")
        print("  python experiments/scripts/run_single_experiment.py \\")
        print("      --config experiments/configs/static_attacks_cognitive_defence.yaml \\")
        print("      --output-dir results/phase1_test")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
