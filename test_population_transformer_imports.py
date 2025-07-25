#!/usr/bin/env python3
"""
Test PopulationTransformer integration imports and basic functionality.
"""

import sys
import os
from unittest.mock import Mock, patch
import traceback

def test_basic_imports():
    """Test if we can import basic components."""
    print("Testing basic imports...")
    
    try:
        import torch
        import numpy
        import yaml
        print("✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing core dependency: {e}")
        return False

def test_config_loading():
    """Test loading our PopulationTransformer config."""
    print("Testing config loading...")
    
    try:
        import yaml
        
        config_path = "configs/population_transformer/population_transformer_cpu.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded: {config['trial_name']}")
        print(f"   Model: {config['model_constructor_name']}")
        print(f"   Device: {config['data_params']['preprocessor_params']['device']}")
        print(f"   Batch size: {config['data_params']['preprocessor_params']['batch_size']}")
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_registry_imports():
    """Test if our registry functions can be imported (with mocking)."""
    print("Testing registry imports...")
    
    try:
        # Mock MNE since we don't have it yet
        sys.modules['mne'] = Mock()
        sys.modules['mne.io'] = Mock()
        sys.modules['mne.io.Raw'] = Mock()
        
        # Mock registry
        mock_registry = Mock()
        mock_registry.register_model_constructor = lambda: lambda func: func
        mock_registry.register_data_preprocessor = lambda: lambda func: func  
        mock_registry.register_config_setter = lambda name: lambda func: func
        
        sys.modules['registry'] = mock_registry
        
        # Mock config
        sys.modules['config'] = Mock()
        sys.modules['config'].ExperimentConfig = Mock()
        
        # Now try to import our module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "population_transformer_utils",
            "population_transformer_module/population_transformer_utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print("✅ Registry functions imported successfully")
        
        # Check if our functions exist
        functions = ['population_transformer_mlp', 
                    'population_transformer_preprocessing_fn',
                    'population_transformer_config_setter']
        
        for func_name in functions:
            if hasattr(module, func_name):
                print(f"   ✅ Found: {func_name}")
            else:
                print(f"   ❌ Missing: {func_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Registry import failed: {e}")
        traceback.print_exc()
        return False

def test_model_constructor():
    """Test our model constructor with mock parameters."""
    print("Testing model constructor...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Mock the dependencies and import our module
        sys.modules['mne'] = Mock()
        sys.modules['mne.io'] = Mock()
        sys.modules['registry'] = Mock()
        sys.modules['config'] = Mock()
        
        # Mock PopulationTransformer modules  
        mock_pt_build_model = Mock()
        mock_pt_build_preprocessor = Mock()
        
        sys.modules['models'] = Mock()
        sys.modules['models'].build_model = mock_pt_build_model
        sys.modules['preprocessors'] = Mock()
        sys.modules['preprocessors'].build_preprocessor = mock_pt_build_preprocessor
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "population_transformer_utils",
            "population_transformer_module/population_transformer_utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Test our decoder
        model_params = {
            'input_dim': 512,
            'output_dim': 50,
            'hidden_dims': [128, 64]
        }
        
        decoder = module.PopulationTransformerDecoder(**model_params)
        
        # Test forward pass
        test_input = torch.randn(4, 512)  # Batch of 4, 512 features
        output = decoder(test_input)
        
        print(f"✅ Model constructor works!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output dim: {model_params['output_dim']}")
        
        if output.shape == (4, 50):
            print("   ✅ Output shape correct!")
            return True
        else:
            print("   ❌ Output shape incorrect")
            return False
        
    except Exception as e:
        print(f"❌ Model constructor test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PopulationTransformer Integration Test")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_config_loading, 
        test_registry_imports,
        test_model_constructor
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 PopulationTransformer integration is working!")
        print("\n📋 Next steps:")
        print("1. Install MNE: pip install mne")
        print("2. Download PopulationTransformer weights")
        print("3. Run: make population-transformer-cpu")
    else:
        print("❌ Some tests failed - but this might be expected without full dependencies")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 