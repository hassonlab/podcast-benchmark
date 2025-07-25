#!/usr/bin/env python3
"""
Basic test script for PopulationTransformer integration structure.
No external dependencies required.
"""

import os
import sys

def test_file_structure():
    """Test if all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "population_transformer_module/__init__.py",
        "population_transformer_module/population_transformer_utils.py",
        "configs/population_transformer/population_transformer_base.yml",
        "configs/population_transformer/population_transformer_cpu.yml",
        "configs/population_transformer/population_transformer_frozen.yml",
        "configs/population_transformer/population_transformer_finetune.yml",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def test_main_py_import():
    """Test if main.py has our import."""
    print("Testing main.py import...")
    
    try:
        with open("main.py", 'r') as f:
            content = f.read()
        
        if 'import_all_from_package("population_transformer_module")' in content:
            print("✅ main.py import found")
            return True
        else:
            print("❌ main.py import missing")
            return False
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")
        return False

def test_makefile_targets():
    """Test if Makefile has our targets."""
    print("Testing Makefile targets...")
    
    try:
        with open("Makefile", 'r') as f:
            content = f.read()
        
        targets = [
            "population-transformer:",
            "population-transformer-cpu:",
        ]
        
        missing_targets = []
        for target in targets:
            if target not in content:
                missing_targets.append(target)
        
        if missing_targets:
            print(f"❌ Missing Makefile targets: {missing_targets}")
            return False
        
        print("✅ Makefile targets found")
        return True
    except Exception as e:
        print(f"❌ Error reading Makefile: {e}")
        return False

def test_config_content():
    """Test if config files have required content."""
    print("Testing config file content...")
    
    try:
        with open("configs/population_transformer/population_transformer_cpu.yml", 'r') as f:
            content = f.read()
        
        required_strings = [
            "population_transformer_mlp",
            "population_transformer_preprocessing_fn", 
            "device: cpu",
            "batch_size: 4"
        ]
        
        missing_content = []
        for req_str in required_strings:
            if req_str not in content:
                missing_content.append(req_str)
        
        if missing_content:
            print(f"❌ Missing config content: {missing_content}")
            return False
        
        print("✅ Config content looks good")
        return True
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def test_utils_functions():
    """Test if utils file has required function signatures."""
    print("Testing utils file functions...")
    
    try:
        with open("population_transformer_module/population_transformer_utils.py", 'r') as f:
            content = f.read()
        
        required_functions = [
            "def population_transformer_mlp(",
            "def population_transformer_preprocessing_fn(",
            "def population_transformer_config_setter(",
            "@registry.register_model_constructor()",
            "@registry.register_data_preprocessor()",
            "@registry.register_config_setter("
        ]
        
        missing_functions = []
        for func in required_functions:
            if func not in content:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing function signatures: {missing_functions}")
            return False
        
        print("✅ Function signatures found")
        return True
    except Exception as e:
        print(f"❌ Error reading utils file: {e}")
        return False

def main():
    """Run all basic tests."""
    print("=" * 60)
    print("PopulationTransformer Integration Basic Test")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_main_py_import,
        test_makefile_targets,
        test_config_content,
        test_utils_functions
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 Integration structure is complete!")
        print("\n📋 To run PopulationTransformer on CPU:")
        print("1. Install: pip install torch numpy pyyaml mne pandas scikit-learn")
        print("2. Download PopulationTransformer weights to:")
        print("   population_transformer/pretrained_weights/popt_brainbert_stft.pth")
        print("3. Run: make population-transformer-cpu")
        print("\n💡 CPU Performance Tips:")
        print("- Use smaller batch sizes (already set to 4 in CPU config)")
        print("- Freeze PopulationTransformer weights (enabled in CPU config)")
        print("- Consider fewer folds and smaller lag ranges for testing")
    else:
        print("❌ Some structure tests failed. Please fix the issues above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 