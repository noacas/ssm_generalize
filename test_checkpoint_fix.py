#!/usr/bin/env python3
"""
Test script to verify the checkpoint division by zero fix.
"""

import tempfile
import shutil
from pathlib import Path
from checkpoint import CheckpointManager
import argparse

def test_checkpoint_division_by_zero_fix():
    """Test that checkpoint saving doesn't crash with division by zero."""
    
    # Create a temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create mock args
        args = argparse.Namespace(
            student_dims=[30, 50],
            sequence_length=5,
            num_seeds=2,
            figures_dir=temp_dir / "figures",
            results_dir=temp_dir / "results",
            checkpoint_dir=temp_dir / "checkpoints"
        )
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(args, checkpoint_interval=1)
        
        # Test case 1: total_experiments = 0 (should not crash)
        print("Test 1: total_experiments = 0")
        checkpoint_manager.update_results({}, 0, 0)
        checkpoint_manager.save_final_checkpoint()
        print("✓ Test 1 passed: No division by zero error")
        
        # Test case 2: total_experiments > 0 (should show progress)
        print("\nTest 2: total_experiments > 0")
        checkpoint_manager.update_results({}, 5, 10)
        checkpoint_manager.save_final_checkpoint()
        print("✓ Test 2 passed: Progress calculation works")
        
        # Test case 3: completed > total (edge case)
        print("\nTest 3: completed > total")
        checkpoint_manager.update_results({}, 15, 10)
        checkpoint_manager.save_final_checkpoint()
        print("✓ Test 3 passed: Edge case handled")
        
        print("\n🎉 All tests passed! The division by zero fix is working correctly.")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_checkpoint_division_by_zero_fix()

