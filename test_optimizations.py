#!/usr/bin/env python3
"""
Test script to verify CPU bottleneck optimizations and measure performance improvements.
"""

import time
import torch
import numpy as np
from performance_config import optimize_environment, perf_config
from generator import generate_teacher, generate_dataset, generate_students
from ssm_forward import ssm_forward
from loss import get_y_teacher, gnc_sensing_loss
from training import train_gnc, train_gd

def test_data_generation_performance():
    """Test the performance of optimized data generation."""
    print("Testing data generation performance...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    num_measurements = 10
    sequence_length = 5
    teacher_rank = 1
    student_dim = 500
    batch_size = 1000
    
    # Test teacher generation
    start_time = time.time()
    for _ in range(10):
        teacher = generate_teacher(teacher_rank, student_dim, device)
    teacher_time = time.time() - start_time
    print(f"Teacher generation: {teacher_time:.3f}s for 10 iterations")
    
    # Test dataset generation
    start_time = time.time()
    for _ in range(10):
        dataset = generate_dataset(num_measurements, sequence_length, False, device)
    dataset_time = time.time() - start_time
    print(f"Dataset generation: {dataset_time:.3f}s for 10 iterations")
    
    # Test student generation
    start_time = time.time()
    for _ in range(10):
        students = generate_students(student_dim, batch_size, sequence_length, device)
    student_time = time.time() - start_time
    print(f"Student generation: {student_time:.3f}s for 10 iterations")
    
    return teacher_time, dataset_time, student_time

def test_ssm_forward_performance():
    """Test the performance of optimized SSM forward pass."""
    print("\nTesting SSM forward pass performance...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    batch_size = 100
    student_dim = 500
    num_measurements = 10
    sequence_length = 5
    
    # Generate test data
    A_diag = torch.normal(mean=0, std=1/student_dim**(1/2), size=(batch_size, student_dim), device=device)
    B = torch.ones(batch_size, 1, student_dim, device=device)
    C = torch.ones(batch_size, student_dim, 1, device=device)
    x = torch.normal(mean=0, std=1, size=(num_measurements, sequence_length, 1), device=device)
    
    # Warm up
    for _ in range(5):
        _ = ssm_forward(A_diag, B, C, x)
    
    # Test performance
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(50):
        output = ssm_forward(A_diag, B, C, x)
    
    torch.cuda.synchronize()
    forward_time = time.time() - start_time
    
    print(f"SSM forward pass: {forward_time:.3f}s for 50 iterations")
    print(f"Output shape: {output.shape}")
    
    return forward_time

def test_training_performance():
    """Test the performance of optimized training."""
    print("\nTesting training performance...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    seed = 42
    student_dim = 500
    num_measurements = 5
    sequence_length = 5
    teacher_rank = 1
    
    # Generate test data
    teacher = generate_teacher(teacher_rank, student_dim, device)
    dataset = generate_dataset(num_measurements, sequence_length, False, device)
    y_teacher = get_y_teacher(teacher, dataset)
    
    # Test GNC training
    print("Testing GNC training...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    mean_prior, gnc_gen_loss = train_gnc(
        seed, student_dim, device, y_teacher, dataset,
        eps_train=1e-5, num_samples=1000, batch_size=100,
        sequence_length=sequence_length, calc_loss_only_on_last_output=True
    )
    
    torch.cuda.synchronize()
    gnc_time = time.time() - start_time
    
    print(f"GNC training: {gnc_time:.3f}s")
    print(f"GNC results: mean_prior={mean_prior:.6f}, gen_loss={gnc_gen_loss:.6f}")
    
    return gnc_time

def test_memory_usage():
    """Test memory usage and management."""
    print("\nTesting memory usage...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        # Get initial memory stats
        initial_allocated = torch.cuda.memory_allocated(device) / 1024**3
        initial_reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f"Initial memory - Allocated: {initial_allocated:.3f}GB, Reserved: {initial_reserved:.3f}GB")
        
        # Create some tensors
        large_tensor = torch.randn(1000, 1000, device=device)
        
        # Get memory stats after allocation
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f"After allocation - Allocated: {allocated:.3f}GB, Reserved: {reserved:.3f}GB")
        
        # Clear memory
        del large_tensor
        torch.cuda.empty_cache()
        
        # Get memory stats after clearing
        final_allocated = torch.cuda.memory_allocated(device) / 1024**3
        final_reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f"After clearing - Allocated: {final_allocated:.3f}GB, Reserved: {final_reserved:.3f}GB")
        
        return allocated - initial_allocated
    else:
        print("CUDA not available, skipping memory test")
        return 0

def main():
    """Run all performance tests."""
    print("=== CPU Bottleneck Optimization Test ===")
    
    # Apply performance optimizations
    optimize_environment()
    
    # Test data generation
    teacher_time, dataset_time, student_time = test_data_generation_performance()
    
    # Test SSM forward pass
    forward_time = test_ssm_forward_performance()
    
    # Test training
    gnc_time = test_training_performance()
    
    # Test memory usage
    memory_used = test_memory_usage()
    
    # Summary
    print("\n=== Performance Summary ===")
    print(f"Data generation total time: {teacher_time + dataset_time + student_time:.3f}s")
    print(f"SSM forward pass time: {forward_time:.3f}s")
    print(f"GNC training time: {gnc_time:.3f}s")
    if torch.cuda.is_available():
        print(f"Peak memory usage: {memory_used:.3f}GB")
    
    print("\nOptimizations applied:")
    print(f"  CUDA benchmark: {perf_config.cudnn_benchmark}")
    print(f"  Max GPU load: {perf_config.max_gpu_load}")
    print(f"  Max GPU memory: {perf_config.max_gpu_memory}")
    print(f"  Max processes: {perf_config.max_processes}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
