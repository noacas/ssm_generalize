#!/usr/bin/env python3

import torch
import logging
import numpy as np
from theoretical_loss import gnc_theoretical_loss

# Set up logging to see warnings and errors
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_theoretical_stability():
    """Test the stability of theoretical loss calculations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test parameters that might cause instability
    test_cases = [
        # (alpha_teacher, w, student_dim)
        (0.5, torch.tensor([1.0, 0.1, 0.01, 0.001], device=device), 100),
        (0.5, torch.tensor([0.001, 0.01, 0.1, 1.0], device=device), 100),
        (0.9, torch.tensor([1.0, 0.1, 0.01, 0.001], device=device), 100),
        (0.1, torch.tensor([1.0, 0.1, 0.01, 0.001], device=device), 100),
        (0.5, torch.tensor([1e-6, 1e-5, 1e-4, 1e-3], device=device), 100),
        (0.5, torch.tensor([1e3, 1e2, 1e1, 1e0], device=device), 100),
    ]
    
    print("Testing theoretical loss stability...")
    print("=" * 60)
    
    for i, (alpha_teacher, w, student_dim) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"  alpha_teacher: {alpha_teacher}")
        print(f"  w: {w}")
        print(f"  student_dim: {student_dim}")
        
        try:
            exact_loss, asymptotic_loss = gnc_theoretical_loss(alpha_teacher, w, student_dim, device)
            
            if torch.isnan(exact_loss) or torch.isnan(asymptotic_loss):
                print(f"  ❌ Result: NaN detected")
                print(f"    Exact: {exact_loss}")
                print(f"    Asymptotic: {asymptotic_loss}")
            elif torch.isinf(exact_loss) or torch.isinf(asymptotic_loss):
                print(f"  ❌ Result: Inf detected")
                print(f"    Exact: {exact_loss}")
                print(f"    Asymptotic: {asymptotic_loss}")
            elif abs(exact_loss) > 1e6 or abs(asymptotic_loss) > 1e6:
                print(f"  ⚠️  Result: Very large values")
                print(f"    Exact: {exact_loss:.6f}")
                print(f"    Asymptotic: {asymptotic_loss:.6f}")
            else:
                print(f"  ✅ Result: Stable")
                print(f"    Exact: {exact_loss:.6f}")
                print(f"    Asymptotic: {asymptotic_loss:.6f}")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("Testing with varying student dimensions...")
    
    # Test with a fixed set of parameters across different student dimensions
    alpha_teacher = 0.5
    w = torch.tensor([1.0, 0.1, 0.01, 0.001], device=device)
    
    for student_dim in [100, 125, 150, 175, 200, 225, 250, 275]:
        try:
            exact_loss, asymptotic_loss = gnc_theoretical_loss(alpha_teacher, w, student_dim, device)
            
            if torch.isnan(exact_loss) or torch.isnan(asymptotic_loss):
                print(f"d={student_dim}: ❌ NaN")
            elif torch.isinf(exact_loss) or torch.isinf(asymptotic_loss):
                print(f"d={student_dim}: ❌ Inf")
            elif abs(exact_loss) > 1e6 or abs(asymptotic_loss) > 1e6:
                print(f"d={student_dim}: ⚠️  Large values - Exact: {exact_loss:.2f}, Asymptotic: {asymptotic_loss:.2f}")
            else:
                print(f"d={student_dim}: ✅ Stable - Exact: {exact_loss:.6f}, Asymptotic: {asymptotic_loss:.6f}")
                
        except Exception as e:
            print(f"d={student_dim}: ❌ Exception: {e}")

if __name__ == "__main__":
    test_theoretical_stability()
