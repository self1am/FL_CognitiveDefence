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
        print("✅ Attacks imported successfully")
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
    """Test cognitive defence creation"""
    print("\nTesting cognitive defence...")
    
    try:
        from src.defences.cognitive_defence import CognitivedefenceStrategy
        import numpy as np
        
        defence = CognitivedefenceStrategy()
        
        # Create dummy client updates
        client_updates = {
            "client_0": ([np.random.randn(10), np.random.randn(5)], 100, {"loss": 0.5}),
            "client_1": ([np.random.randn(10), np.random.randn(5)], 80, {"loss": 0.6}),
        }
        
        aggregated_params, decisions = defence.aggregate_updates(client_updates)
        
        print(f"✅ Cognitive defence created")
        print(f"✅ Processed {len(client_updates)} client updates")
        print(f"✅ Generated {len(decisions)} decisions")
        
        return True
        
    except Exception as e:
        print(f"❌ Cognitive defence failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🚀 Testing Federated Cognitive defence Setup\n")
    
    tests = [
        test_imports,
        test_device_setup,
        test_data_loading,
        test_model_creation,
        test_cognitive_defence
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