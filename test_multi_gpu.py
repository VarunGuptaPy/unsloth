#!/usr/bin/env python3
"""
Test script for multi-GPU functionality in Unsloth.

This script tests the multi-GPU implementation without actually requiring GPUs.
It can be run on any system to verify that the code structure is correct.

Usage:
    python test_multi_gpu.py
"""

import os
import sys

def test_imports():
    """Test that all multi-GPU related imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test distributed utils import
        from unsloth.distributed_utils import (
            is_distributed_available,
            is_distributed_initialized,
            get_world_size,
            get_rank,
            is_main_process,
            setup_distributed_training,
            print_distributed_info,
        )
        print("✓ Distributed utils imported successfully")
        
        # Test trainer imports
        from unsloth.trainer import UnslothTrainingArguments, UnslothTrainer
        print("✓ UnslothTrainer imported successfully")
        
        # Test that distributed utils are exported from main module
        import unsloth
        assert hasattr(unsloth, 'is_distributed_available')
        assert hasattr(unsloth, 'UnslothTrainingArguments')
        assert hasattr(unsloth, 'UnslothTrainer')
        print("✓ Multi-GPU functions exported from main module")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_distributed_utils():
    """Test distributed utility functions."""
    print("\nTesting distributed utility functions...")
    
    try:
        from unsloth.distributed_utils import (
            is_distributed_available,
            is_distributed_initialized,
            get_world_size,
            get_rank,
            is_main_process,
        )
        
        # These should work even without GPUs
        world_size = get_world_size()
        rank = get_rank()
        is_main = is_main_process()
        is_init = is_distributed_initialized()
        
        print(f"✓ World size: {world_size}")
        print(f"✓ Rank: {rank}")
        print(f"✓ Is main process: {is_main}")
        print(f"✓ Is distributed initialized: {is_init}")
        
        # Test device map functions
        from unsloth.distributed_utils import get_device_map_for_distributed, get_optimal_device_map
        
        device_map = get_device_map_for_distributed("test_model")
        optimal_map = get_optimal_device_map({})
        
        print(f"✓ Device map for distributed: {device_map}")
        print(f"✓ Optimal device map: {optimal_map}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing distributed utils: {e}")
        return False

def test_training_arguments():
    """Test UnslothTrainingArguments creation."""
    print("\nTesting UnslothTrainingArguments...")
    
    try:
        from unsloth.trainer import UnslothTrainingArguments
        
        # Test basic creation
        args = UnslothTrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            enable_distributed_training=False,  # Disable for testing
            distributed_backend="nccl",
        )
        
        print("✓ UnslothTrainingArguments created successfully")
        print(f"✓ Enable distributed training: {args.enable_distributed_training}")
        print(f"✓ Distributed backend: {args.distributed_backend}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing training arguments: {e}")
        return False

def test_trainer_creation():
    """Test UnslothTrainer creation (without actual model)."""
    print("\nTesting UnslothTrainer creation...")
    
    try:
        from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
        
        # Create mock training arguments
        args = UnslothTrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            max_steps=1,
            enable_distributed_training=False,  # Disable for testing
        )
        
        print("✓ Training arguments created for trainer test")
        
        # Note: We can't actually create the trainer without a model,
        # but we can verify the class exists and is importable
        assert hasattr(UnslothTrainer, '__init__')
        assert hasattr(UnslothTrainer, 'training_step')
        assert hasattr(UnslothTrainer, 'compute_loss')
        
        print("✓ UnslothTrainer class structure verified")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing trainer creation: {e}")
        return False

def test_examples_exist():
    """Test that example files exist and are readable."""
    print("\nTesting example files...")
    
    examples = [
        "examples/multi_gpu_training.py",
        "examples/multi_gpu_inference.py",
        "docs/MULTI_GPU_TRAINING.md",
    ]
    
    all_exist = True
    for example in examples:
        if os.path.exists(example):
            print(f"✓ {example} exists")
        else:
            print(f"✗ {example} missing")
            all_exist = False
    
    return all_exist

def test_cli_updates():
    """Test that CLI script has been updated."""
    print("\nTesting CLI updates...")
    
    try:
        with open("unsloth-cli.py", "r") as f:
            content = f.read()
        
        # Check for multi-GPU related imports and usage
        checks = [
            "UnslothTrainingArguments" in content,
            "UnslothTrainer" in content,
            "device_map" in content,
            "enable_distributed_training" in content,
        ]
        
        if all(checks):
            print("✓ CLI script updated with multi-GPU support")
            return True
        else:
            print("✗ CLI script missing some multi-GPU features")
            return False
            
    except Exception as e:
        print(f"✗ Error checking CLI updates: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Unsloth Multi-GPU Implementation Test")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Distributed Utils Tests", test_distributed_utils),
        ("Training Arguments Tests", test_training_arguments),
        ("Trainer Tests", test_trainer_creation),
        ("Example Files Tests", test_examples_exist),
        ("CLI Updates Tests", test_cli_updates),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Multi-GPU implementation looks good.")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
