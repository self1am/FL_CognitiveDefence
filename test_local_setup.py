# test_local_setup.py
"""Debug script to test local setup"""
import sys
import traceback
from pathlib import Path
import torch

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        import flwr as fl
        print("✅ Flower imported successfully")
    except ImportError as e:
        print(f"❌ Flower import failed: {e}")
        return False
    
    try:
        from src.utils.config import ExperimentConfig, DeterministicEnvironment
        print("✅ Utils imported successfully")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    try:
        from src.attacks.label_flip import LabelFlipAttack
        from src.attacks import StatOptAttack, DnyOptAttack, MinMaxAttack, MinSumAttack
        print("✅ Attacks (including adaptive attacks) imported successfully")
    except ImportError as e:
        print(f"❌ Attacks import failed: {e}")
        return False
    
    try:
        from src.defences.cognitive_defence import CognitivedefenceStrategy
        print("✅ defences imported successfully")
    except ImportError as e:
        print(f"❌ defences import failed: {e}")
        return False
    
    try:
        from src.models.cnn_mnist import MNISTNet
        print("✅ Models imported successfully")
    except ImportError as e:
        print(f"❌ Models import failed: {e}")
        return False
    
    try:
        from src.datasets.mnist_handler import MNISTDataHandler
        print("✅ Dataset handlers imported successfully")
    except ImportError as e:
        print(f"❌ Dataset handlers import failed: {e}")
        return False
    
    return True

def test_device_setup():
    """Test device configuration"""
    print("\nTesting device setup...")
    
    from src.utils.config import DeterministicEnvironment
    
    device = DeterministicEnvironment.get_device()
    print(f"✅ Device detected: {device}")
    
    if device.type == "mps":
        print("✅ Apple Silicon MPS acceleration available")
    elif device.type == "cuda":
        print("✅ CUDA acceleration available")
    else:
        print("ℹ️  Using CPU (normal for testing)")
    
    return True

def test_data_loading():
    """Test MNIST data loading"""
    print("\nTesting data loading...")
    
    try:
        from src.datasets.mnist_handler import MNISTDataHandler
        
        handler = MNISTDataHandler(batch_size=32)
        client_loaders, test_loader = handler.create_client_dataloaders(num_clients=3, alpha=0.5)
        
        print(f"✅ Created {len(client_loaders)} client dataloaders")
        print(f"✅ Test loader has {len(test_loader.dataset)} samples")
        
        # Test a batch
        for i, (data, target) in enumerate(client_loaders[0]):
            print(f"✅ Client 0 batch shape: {data.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation and basic forward pass"""
    print("\nTesting model creation...")
    
    try:
        from src.models.cnn_mnist import MNISTNet
        from src.utils.config import DeterministicEnvironment
        
        device = DeterministicEnvironment.get_device()
        model = MNISTNet()
        model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        output = model(dummy_input)
        
        print(f"✅ Model created and moved to {device}")
        print(f"✅ Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_cognitive_defence():
    """Test cognitive aggregation strategy creation"""
    print("\nTesting cognitive aggregation strategy...")
    
    try:
        from src.server.cognitive_server import CognitiveAggregationStrategy
        from src.utils.config import ExperimentConfig
        import numpy as np
        
        # Create test config
        config = ExperimentConfig(
            experiment_name="test",
            seed=42,
            num_rounds=5,
            min_clients=2,
            min_available_clients=2,
            server_address="0.0.0.0:8080"
        )
        
        strategy = CognitiveAggregationStrategy(
            config=config,
            anomaly_threshold=0.7,
            reputation_decay=0.8,
            history_size=100
        )
        
        print(f"✅ Cognitive aggregation strategy created")
        print(f"✅ Anomaly threshold: {strategy.anomaly_threshold}")
        print(f"✅ Reputation decay: {strategy.reputation_decay}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cognitive aggregation strategy failed: {e}")
        traceback.print_exc()
        return False

def test_krum_defence():
    """Test Krum defence strategy creation"""
    print("\nTesting Krum defence strategy...")
    
    try:
        from src.defences.krum_defence import KrumDefenceStrategy
        from src.server.krum_server import KrumAggregationStrategy
        from src.utils.config import ExperimentConfig
        
        # Test Krum defence
        krum_defence = KrumDefenceStrategy(num_byzantine=2, multi_krum=False)
        print(f"✅ Krum defence strategy created")
        print(f"✅ Description: {krum_defence.get_defence_description()}")
        
        # Test Krum server
        config = ExperimentConfig(
            experiment_name="test",
            seed=42,
            num_rounds=5,
            min_clients=2,
            min_available_clients=2,
            server_address="0.0.0.0:8080"
        )
        
        strategy = KrumAggregationStrategy(
            config=config,
            num_byzantine=2,
            multi_krum=False
        )
        
        print(f"✅ Krum aggregation strategy created")
        
        return True
        
    except Exception as e:
        print(f"❌ Krum defence strategy failed: {e}")
        traceback.print_exc()
        return False

def test_trimmed_mean_defence():
    """Test Trimmed Mean defence strategy creation"""
    print("\nTesting Trimmed Mean defence strategy...")
    
    try:
        from src.defences.trimmed_mean_defence import TrimmedMeanDefenceStrategy
        from src.server.trimmed_mean_server import TrimmedMeanAggregationStrategy
        from src.utils.config import ExperimentConfig
        
        # Test Trimmed Mean defence
        trimmed_mean_defence = TrimmedMeanDefenceStrategy(beta=0.2)
        print(f"✅ Trimmed Mean defence strategy created")
        print(f"✅ Description: {trimmed_mean_defence.get_defence_description()}")
        
        # Test Trimmed Mean server
        config = ExperimentConfig(
            experiment_name="test",
            seed=42,
            num_rounds=5,
            min_clients=2,
            min_available_clients=2,
            server_address="0.0.0.0:8080"
        )
        
        strategy = TrimmedMeanAggregationStrategy(
            config=config,
            beta=0.2
        )
        
        print(f"✅ Trimmed Mean aggregation strategy created")
        
        return True
        
    except Exception as e:
        print(f"❌ Trimmed Mean defence strategy failed: {e}")
        traceback.print_exc()
        return False

def test_vert_defence():
    """Test VERT defence strategy creation"""
    print("\nTesting VERT defence strategy...")
    
    try:
        from src.defences.vert_defence import VERTDefenceStrategy
        from src.server.vert_server import VERTAggregationStrategy
        from src.utils.config import ExperimentConfig
        import numpy as np
        
        # Test VERT defence
        vert_defence = VERTDefenceStrategy(kappa=5, history_size=10, projection_dim=100)
        print(f"✅ VERT defence strategy created")
        print(f"✅ Description: {vert_defence.get_defence_description()}")
        
        # Test VERT aggregation with simulated client updates
        # Create mock client updates
        np.random.seed(42)
        client_updates = {}
        for i in range(8):
            # Simulate 2 parameter layers
            params = [
                np.random.randn(10, 5).astype(np.float32),
                np.random.randn(5).astype(np.float32)
            ]
            client_updates[f"client_{i}"] = (params, 100, {'loss': 0.5})
        
        # Run aggregation (first few rounds will use fallback)
        for round_num in range(4):
            aggregated, decisions = vert_defence.aggregate_updates(client_updates)
            if aggregated is not None:
                print(f"✅ Round {round_num}: Aggregation successful, {len(decisions)} decisions made")
        
        # Test VERT server
        config = ExperimentConfig(
            experiment_name="test",
            seed=42,
            num_rounds=5,
            min_clients=2,
            min_available_clients=2,
            server_address="0.0.0.0:8080"
        )
        
        strategy = VERTAggregationStrategy(
            config=config,
            kappa=5,
            history_size=10,
            projection_dim=100
        )
        
        print(f"✅ VERT aggregation strategy created")
        
        return True
        
    except Exception as e:
        print(f"❌ VERT defence strategy failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🚀 Testing Federated Cognitive defence Setup\n")
    
    tests = [
        test_imports,
        test_device_setup,
        test_data_loading,
        test_model_creation,
        test_cognitive_defence,
        test_krum_defence,
        test_trimmed_mean_defence,
        test_vert_defence,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            traceback.print_exc()
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Ready to run experiments.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)