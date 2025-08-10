#!/usr/bin/env python3
"""
Test script to verify multi-GPU functionality
"""

import torch
import GPUtil
from utils import get_available_gpus, get_available_device

def test_gpu_detection():
    print("=== Testing GPU Detection ===")
    
    print(f"Total CUDA devices: {torch.cuda.device_count()}")
    
    # Test single GPU selection (original function)
    single_device = get_available_device(max_load=0.3, max_memory=0.3)
    print(f"Single available device: {single_device}")
    
    # Test multiple GPU selection (new function)
    multiple_gpus = get_available_gpus(max_load=0.3, max_memory=0.3)
    print(f"All available GPUs: {multiple_gpus}")
    
    # Test with limit
    limited_gpus = get_available_gpus(max_load=0.3, max_memory=0.3, max_gpus=2)
    print(f"Limited GPUs (max 2): {limited_gpus}")
    
    # Show GPU info
    print("\n=== GPU Information ===")
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            print(f"  Load: {gpu.load:.1%}")
            print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal}MB ({gpu.memoryUtil:.1%})")
            print(f"  Temperature: {gpu.temperature}°C")
            print()
    except Exception as e:
        print(f"Could not get GPU info: {e}")

if __name__ == "__main__":
    test_gpu_detection()
