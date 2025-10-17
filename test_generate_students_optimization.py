#!/usr/bin/env python3
"""
Test script to verify the generate_students optimization works correctly
and measure performance improvement.
"""

import torch
import time
import numpy as np
from generator import generate_students

def original_generate_students(student_dim: int, bs: int, device: torch.device):
    """Original implementation for comparison"""
    return torch.normal(mean=0, std=1/(student_dim**0.5), size=(bs, student_dim), device=device)

def test_correctness():
    """Test that the optimized version produces the same statistical properties"""
    print("Testing correctness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_dim = 200
    bs = 10000
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    original_result = original_generate_students(student_dim, bs, device)
    
    torch.manual_seed(42)
    optimized_result = generate_students(student_dim, bs, device)
    
    # Check that results are identical
    assert torch.allclose(original_result, optimized_result, atol=1e-6), "Results don't match!"
    
    # Check statistical properties
    expected_std = 1.0 / (student_dim ** 0.5)
    original_std = original_result.std().item()
    optimized_std = optimized_result.std().item()
    
    print(f"Expected std: {expected_std:.6f}")
    print(f"Original std: {original_std:.6f}")
    print(f"Optimized std: {optimized_std:.6f}")
    print("✓ Correctness test passed!")
    return True

def benchmark_performance():
    """Benchmark performance improvement"""
    print("\nBenchmarking performance...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters similar to your use case
    test_cases = [
        (200, 10000),    # Small batch
        (200, 100000),   # Medium batch  
        (200, 500000),   # Large batch (your typical case)
        (275, 500000),   # Large batch with max student_dim
    ]
    
    for student_dim, bs in test_cases:
        print(f"\nTesting student_dim={student_dim}, batch_size={bs}")
        
        # Warm up
        for _ in range(3):
            _ = generate_students(student_dim, bs, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark original
        torch.manual_seed(42)
        start_time = time.time()
        for _ in range(5):
            _ = original_generate_students(student_dim, bs, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        original_time = (time.time() - start_time) / 5
        
        # Benchmark optimized
        torch.manual_seed(42)
        start_time = time.time()
        for _ in range(5):
            _ = generate_students(student_dim, bs, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        optimized_time = (time.time() - start_time) / 5
        
        speedup = original_time / optimized_time
        print(f"  Original time: {original_time:.4f}s")
        print(f"  Optimized time: {optimized_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Memory usage comparison
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = generate_students(student_dim, bs, device)
            optimized_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = original_generate_students(student_dim, bs, device)
            original_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            print(f"  Original memory: {original_memory:.1f} MB")
            print(f"  Optimized memory: {optimized_memory:.1f} MB")

def test_chunked_generation():
    """Test that chunked generation works correctly for large batches"""
    print("\nTesting chunked generation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_dim = 200
    bs = 600000  # Large batch to trigger chunking
    
    torch.manual_seed(42)
    result = generate_students(student_dim, bs, device)
    
    assert result.shape == (bs, student_dim), f"Wrong shape: {result.shape}"
    assert result.device == device, f"Wrong device: {result.device}"
    
    # Check statistical properties
    expected_std = 1.0 / (student_dim ** 0.5)
    actual_std = result.std().item()
    print(f"Expected std: {expected_std:.6f}")
    print(f"Actual std: {actual_std:.6f}")
    print("✓ Chunked generation test passed!")

if __name__ == "__main__":
    print("Testing generate_students optimization...")
    
    try:
        test_correctness()
        test_chunked_generation()
        benchmark_performance()
        print("\n🎉 All tests passed! The optimization is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
