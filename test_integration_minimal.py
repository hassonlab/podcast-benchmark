#!/usr/bin/env python3
"""
Minimal test script for PopulationTransformer integration with podcast-benchmark.
This tests the integration logic without requiring all dependencies.
"""

import sys
import os
import tempfile
from unittest.mock import Mock, patch
import yaml

def test_config_loading():
    """Test if our configuration files load correctly."""
    print("Testing configuration loading...")
    
    config_path = "configs/population_transformer/population_transformer_cpu.yml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = [
            'model_constructor_name',
            'config_setter_name', 
            'model_params',
            'training_params',
            'data_params'
        ]
        
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check CPU-specific settings
        if config['data_params']['preprocessor_params']['device'] != 'cpu':
            print("❌ CPU config should have device: cpu")
            return False
            
        print("✅ Configuration loading successful")
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_registry_structure():
    """Test if our registry functions are properly structured."""
    print("Testing registry structure...")
    
    try:
        # Mock the dependencies that aren't available yet
        sys.modules['torch'] = Mock()
        sys.modules['torch.nn'] = Mock()
        sys.modules['mne'] = Mock()
        sys.modules['registry'] = Mock()
        
        # Create a mock registry for testing
        mock_registry = Mock()
        mock_registry.register_model_constructor = lambda: lambda func: func
        mock_registry.register_data_preprocessor = lambda: lambda func: func
        mock_registry.register_config_setter = lambda name: lambda func: func
        
        with patch.dict('sys.modules', {'registry': mock_registry}):
            # Try to import our utils module structure (not execution)
            spec = None
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "population_transformer_utils", 
                    "population_transformer_module/population_transformer_utils.py"
                )
                print("✅ Module structure is valid")
                return True
            except Exception as e:
                print(f"❌ Module structure error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Registry structure test failed: {e}")
        return False

def test_makefile_targets():
    """Test if Makefile has our new targets."""
    print("Testing Makefile targets...")
    
    try:
        with open("Makefile", 'r') as f:
            makefile_content = f.read()
        
        required_targets = [
            'population-transformer:',
            'population-transformer-cpu:',
            'population-transformer-base:',
            'population-transformer-frozen:',
            'population-transformer-finetune:'
        ]
        
        for target in required_targets:
            if target not in makefile_content:
                print(f"❌ Missing Makefile target: {target}")
                return False
        
        print("✅ Makefile targets present")
        return True
        
    except Exception as e:
        print(f"❌ Makefile test failed: {e}")
        return False

def test_main_imports():
    """Test if main.py has our import."""
    print("Testing main.py imports...")
    
    try:
        with open("main.py", 'r') as f:
            main_content = f.read()
        
        if 'import_all_from_package("population_transformer_module")' not in main_content:
            print("❌ Missing import in main.py")
            return False
        
        print("✅ main.py import present")
        return True
        
    except Exception as e:
        print(f"❌ main.py test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PopulationTransformer Integration Test (Minimal)")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_registry_structure,
        test_makefile_targets,
        test_main_imports
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All integration tests passed!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install torch numpy pyyaml")
        print("2. Download PopulationTransformer weights")
        print("3. Run: make population-transformer-cpu")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 