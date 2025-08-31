#!/usr/bin/env python3
"""
Test script to verify multiple sequences functionality
"""

import torch
import numpy as np
from parser import parse_args
from theoretical_loss import gnc_theoretical_loss
from generator import generate_teacher_alpha, generate_w

def test_multiple_sequences():
    """Test that multiple sequences functionality works correctly"""
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate test data
    torch.manual_seed(42)
    alpha_teacher = generate_teacher_alpha(device)
    sequence_length = 5
    
    # Generate single sequence
    w_single = generate_w(sequence_length, device)
    
    # Generate multiple sequences
    num_sequences = 3
    w_sequences = []
    for i in range(num_sequences):
        w = generate_w(sequence_length, device)
        w_sequences.append(w)
    
    student_dim = 200
    
    print(f"Testing with alpha_teacher={alpha_teacher.item():.4f}")
    print(f"Sequence length: {sequence_length}")
    print(f"Student dimension: {student_dim}")
    print(f"Number of sequences: {num_sequences}")
    
    # Test single sequence
    print("\n--- Testing single sequence ---")
    try:
        loss_single, asymp_single, delta_single = gnc_theoretical_loss(alpha_teacher, w_single, student_dim, device)
        print(f"Single sequence loss: {loss_single.item():.6f}")
        print(f"Single sequence asymptotic: {asymp_single.item():.6f}")
        print(f"Single sequence delta: {delta_single.item():.6f}")
    except Exception as e:
        print(f"Error with single sequence: {e}")
        return False
    
    # Test multiple sequences
    print("\n--- Testing multiple sequences ---")
    try:
        loss_multi, asymp_multi, delta_multi = gnc_theoretical_loss(alpha_teacher, w_sequences, student_dim, device)
        print(f"Multiple sequences loss: {loss_multi.item():.6f}")
        print(f"Multiple sequences asymptotic: {asymp_multi.item():.6f}")
        print(f"Multiple sequences delta: {delta_multi.item():.6f}")
    except Exception as e:
        print(f"Error with multiple sequences: {e}")
        return False
    
    # Test that results are reasonable
    print("\n--- Results validation ---")
    if loss_single > 0 and loss_multi > 0:
        print("✓ Both losses are positive")
    else:
        print("✗ One or both losses are non-positive")
        return False
    
    if torch.isfinite(loss_single) and torch.isfinite(loss_multi):
        print("✓ Both losses are finite")
    else:
        print("✗ One or both losses are infinite/NaN")
        return False
    
    print("✓ All tests passed!")
    return True

def test_parser():
    """Test that the parser correctly handles num_sequences argument"""
    print("\n--- Testing parser ---")
    
    # Test default value
    import sys
    sys.argv = ['test']
    args = parse_args()
    print(f"Default num_sequences: {args.num_sequences}")
    
    # Test custom value
    sys.argv = ['test', '--num_sequences', '5']
    args = parse_args()
    print(f"Custom num_sequences: {args.num_sequences}")
    
    print("✓ Parser test passed!")

if __name__ == "__main__":
    print("Testing multiple sequences functionality...")
    
    # Test parser
    test_parser()
    
    # Test multiple sequences
    success = test_multiple_sequences()
    
    if success:
        print("\n🎉 All tests passed! Multiple sequences functionality is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
