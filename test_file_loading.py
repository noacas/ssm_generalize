#!/usr/bin/env python3
"""
Test script for paired file loading functionality
"""

import torch
from pathlib import Path
from generator import load_alpha_w_pairs_from_file, get_alpha_w_pair

def test_paired_file_loading():
    """Test the paired file loading functionality"""
    device = torch.device("cpu")
    
    print("Testing paired file loading functionality...")
    
    # Test paired data loading
    data_file = Path("example_alpha_w_pairs.json")
    if data_file.exists():
        print(f"\n1. Testing paired data loading from {data_file}")
        try:
            pairs = load_alpha_w_pairs_from_file(data_file, device)
            print(f"   Loaded {len(pairs)} alpha-W pairs:")
            for i, (alpha, w_sequences) in enumerate(pairs):
                print(f"   Pair {i}: alpha={alpha.item():.3f}, {len(w_sequences)} sequences")
                for j, seq in enumerate(w_sequences):
                    print(f"     Sequence {j}: {seq}")
        except Exception as e:
            print(f"   Error loading paired data: {e}")
    else:
        print(f"   Data file {data_file} not found")
    
    # Test get_alpha_w_pair with file (using seed to select pair)
    print(f"\n2. Testing get_alpha_w_pair with file (seed-based selection)")
    try:
        alpha, w_sequences = get_alpha_w_pair(data_file, device, seed=0)
        print(f"   Seed 0 -> alpha={alpha.item():.3f}, {len(w_sequences)} sequences")
        for i, seq in enumerate(w_sequences):
            print(f"     Sequence {i}: {seq}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test get_alpha_w_pair with file (using different seed)
    print(f"\n3. Testing get_alpha_w_pair with file (seed=1)")
    try:
        alpha, w_sequences = get_alpha_w_pair(data_file, device, seed=1)
        print(f"   Seed 1 -> alpha={alpha.item():.3f}, {len(w_sequences)} sequences")
        for i, seq in enumerate(w_sequences):
            print(f"     Sequence {i}: {seq}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test get_alpha_w_pair with file (using pair index)
    print(f"\n4. Testing get_alpha_w_pair with file (pair_index=2)")
    try:
        alpha, w_sequences = get_alpha_w_pair(data_file, device, pair_index=2)
        print(f"   Pair index 2 -> alpha={alpha.item():.3f}, {len(w_sequences)} sequences")
        for i, seq in enumerate(w_sequences):
            print(f"     Sequence {i}: {seq}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test get_alpha_w_pair without file (seed-based generation)
    print(f"\n5. Testing get_alpha_w_pair without file (seed-based generation)")
    try:
        alpha, w_sequences = get_alpha_w_pair(None, device, seed=42)
        print(f"   Seed-based -> alpha={alpha.item():.3f}, w_sequences={w_sequences}")
        print("   (w_sequences will be generated separately in the main code)")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nPaired file loading test completed!")

if __name__ == "__main__":
    test_paired_file_loading()
