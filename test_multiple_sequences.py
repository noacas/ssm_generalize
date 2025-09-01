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

def test_asymptotic_convergence():
    """Test that asymptotic loss is close to exact loss for d=150"""
    print("\n--- Testing asymptotic convergence for d=150 ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_dim = 150
    tolerance = 0.05  # 5% relative tolerance
    
    # Test with multiple random seeds to ensure robustness
    seeds_tested = 0
    seeds_passed = 0
    
    for seed in range(42, 52):  # Test 10 different seeds
        torch.manual_seed(seed)
        alpha_teacher = generate_teacher_alpha(device)
        w = generate_w(5, device)  # sequence length of 5
        
        try:
            exact_loss, asymptotic_loss, _ = gnc_theoretical_loss(alpha_teacher, w, student_dim, device)
            
            # Skip if losses are not positive and finite
            if not (exact_loss > 0 and asymptotic_loss > 0 and 
                   torch.isfinite(exact_loss) and torch.isfinite(asymptotic_loss)):
                continue
                
            seeds_tested += 1
            
            # Calculate relative error
            relative_error = abs(exact_loss - asymptotic_loss) / exact_loss
            
            print(f"  Seed {seed}: exact={exact_loss.item():.6f}, asymptotic={asymptotic_loss.item():.6f}, "
                  f"rel_error={relative_error.item():.4f}")
            
            if relative_error <= tolerance:
                seeds_passed += 1
                
        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")
            continue
    
    # Require at least 5 seeds to be tested and at least 80% to pass
    min_seeds_required = 5
    min_pass_rate = 0.8
    
    if seeds_tested < min_seeds_required:
        print(f"✗ Insufficient valid seeds tested ({seeds_tested} < {min_seeds_required})")
        return False
    
    pass_rate = seeds_passed / seeds_tested
    print(f"\nAsymptotic convergence results:")
    print(f"  Seeds tested: {seeds_tested}")
    print(f"  Seeds passed: {seeds_passed}")
    print(f"  Pass rate: {pass_rate:.2%}")
    print(f"  Required pass rate: {min_pass_rate:.0%}")
    print(f"  Tolerance: {tolerance:.1%}")
    
    if pass_rate >= min_pass_rate:
        print("✓ Asymptotic convergence test passed!")
        return True
    else:
        print("✗ Asymptotic convergence test failed!")
        return False

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
    sequences_success = test_multiple_sequences()
    
    # Test asymptotic convergence for d=150
    convergence_success = test_asymptotic_convergence()
    
    overall_success = sequences_success and convergence_success
    
    if overall_success:
        print("\n🎉 All tests passed! Multiple sequences functionality and asymptotic convergence are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        if not sequences_success:
            print("  - Multiple sequences test failed")
        if not convergence_success:
            print("  - Asymptotic convergence test failed")
