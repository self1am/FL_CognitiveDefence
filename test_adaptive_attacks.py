#!/usr/bin/env python3
"""
Test script for adaptive attacks

Tests instantiation and basic functionality of all adaptive attacks.
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.attacks import (
    StatOptAttack, DnyOptAttack, MinMaxAttack, MinSumAttack,
    LabelFlipAttack, GradientNoiseAttack
)


def test_attack_instantiation():
    """Test that all attacks can be instantiated"""
    print("Testing attack instantiation...")
    
    # Test static attacks
    label_flip = LabelFlipAttack(intensity=0.1)
    print(f"✓ {label_flip.get_attack_description()}")
    
    grad_noise = GradientNoiseAttack(intensity=0.1)
    print(f"✓ {grad_noise.get_attack_description()}")
    
    # Test adaptive attacks
    stat_opt = StatOptAttack(intensity=0.2, constraint_factor=1.5)
    print(f"✓ {stat_opt.get_attack_description()}")
    
    dny_opt = DnyOptAttack(intensity=0.15, learning_rate=0.1)
    print(f"✓ {dny_opt.get_attack_description()}")
    
    min_max = MinMaxAttack(intensity=0.2, defense_models=['krum', 'trimmed_mean'])
    print(f"✓ {min_max.get_attack_description()}")
    
    min_sum = MinSumAttack(intensity=0.2, distance_weight=0.7)
    print(f"✓ {min_sum.get_attack_description()}")
    
    print("\n✓ All attacks instantiated successfully!\n")
    return True


def test_attack_parameters():
    """Test that attacks can modify parameters"""
    print("Testing parameter attacks...")
    
    # Create dummy parameters
    params = [
        np.random.randn(10, 5).astype(np.float32),
        np.random.randn(5).astype(np.float32),
        np.random.randn(5, 2).astype(np.float32),
    ]
    
    # Test each attack
    attacks = [
        StatOptAttack(intensity=0.2),
        DnyOptAttack(intensity=0.15),
        MinMaxAttack(intensity=0.2),
        MinSumAttack(intensity=0.2),
    ]
    
    for attack in attacks:
        attacked_params = attack.attack_parameters(params, client_id=0)
        
        # Verify output format
        assert len(attacked_params) == len(params), "Parameter count mismatch"
        for i, (original, attacked) in enumerate(zip(params, attacked_params)):
            assert original.shape == attacked.shape, f"Shape mismatch in param {i}"
            assert attacked.dtype == original.dtype, f"Dtype mismatch in param {i}"
        
        print(f"✓ {attack.__class__.__name__} successfully modified parameters")
    
    print("\n✓ All attacks can modify parameters!\n")
    return True


def test_feedback_mechanism():
    """Test adaptive feedback mechanism"""
    print("Testing feedback mechanism...")
    
    attacks = [
        StatOptAttack(intensity=0.2),
        DnyOptAttack(intensity=0.15),
        MinMaxAttack(intensity=0.2),
        MinSumAttack(intensity=0.2),
    ]
    
    for attack in attacks:
        # Simulate feedback over multiple rounds
        for round_num in range(5):
            was_accepted = round_num % 2 == 0  # Alternate acceptance/rejection
            attack.update_feedback(
                round_num=round_num,
                was_accepted=was_accepted,
                global_accuracy=0.9 - round_num * 0.01,
                anomaly_score=0.5 + round_num * 0.05
            )
        
        # Check feedback was recorded
        assert len(attack.feedback_history) == 5, "Feedback history length mismatch"
        assert attack.round_number == 4, "Round number not updated"
        
        # Check adaptation summary
        summary = attack.get_adaptation_summary()
        assert summary['total_rounds'] == 4
        assert 0 <= summary['detection_rate'] <= 1
        assert 0 <= summary['acceptance_rate'] <= 1
        
        print(f"✓ {attack.__class__.__name__} feedback mechanism working")
    
    print("\n✓ Feedback mechanism working for all adaptive attacks!\n")
    return True


def test_benign_statistics():
    """Test benign statistics update for stat-opt and min-sum"""
    print("Testing benign statistics...")
    
    # Create dummy benign parameters
    benign_params = [
        [np.random.randn(10, 5).astype(np.float32), np.random.randn(5).astype(np.float32)]
        for _ in range(5)
    ]
    
    # Test stat-opt
    stat_opt = StatOptAttack(intensity=0.2)
    stat_opt.update_benign_statistics(benign_params)
    assert stat_opt.benign_stats, "Benign stats not updated"
    print(f"✓ StatOptAttack benign statistics: mean={stat_opt.benign_stats['mean']:.4f}, std={stat_opt.benign_stats['std']:.4f}")
    
    # Test min-sum
    min_sum = MinSumAttack(intensity=0.2)
    min_sum.update_benign_estimates(benign_params)
    assert min_sum.benign_centroid is not None, "Benign centroid not computed"
    assert len(min_sum.benign_updates) == 5, "Benign updates not stored"
    print(f"✓ MinSumAttack benign estimates: centroid shape={min_sum.benign_centroid.shape}, num_updates={len(min_sum.benign_updates)}")
    
    print("\n✓ Benign statistics working!\n")
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("ADAPTIVE ATTACKS TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_attack_instantiation,
        test_attack_parameters,
        test_feedback_mechanism,
        test_benign_statistics,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}\n")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
