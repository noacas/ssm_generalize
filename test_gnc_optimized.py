#!/usr/bin/env python3
"""
Simple test to verify the optimized GNC prediction works well with 2 sequences.
"""

import torch
import time
import logging
from generator import generate_teacher_alpha, generate_w
from training import train_gnc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_gnc_optimized():
    """Test GNC with prediction optimization using 2 sequences."""
    
    # Test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_dim = 200
    eps_train = 1e-5
    num_samples = 10000
    batch_size = 2000
    
    # Generate test data with 2 sequences
    torch.manual_seed(42)
    alpha_teacher = generate_teacher_alpha(device)
    w_sequences = [generate_w(5, device), generate_w(5, device)]  # 2 sequences
    
    logging.info(f"Testing optimized GNC with 2 sequences")
    logging.info(f"Device: {device}")
    logging.info(f"Student dim: {student_dim}, Samples: {num_samples}, Batch size: {batch_size}")
    logging.info(f"Epsilon: {eps_train}")
    
    # Test without prediction
    logging.info("\n=== Testing WITHOUT prediction ===")
    start_time = time.time()
    mean_prior_no_pred, mean_gnc_no_pred = train_gnc(
        seed=42, student_dim=student_dim, device=device, alpha_teacher=alpha_teacher,
        w_sequences=w_sequences, eps_train=eps_train, num_samples=num_samples,
        batch_size=batch_size, use_prediction=False
    )
    time_no_pred = time.time() - start_time
    
    # Test with prediction
    logging.info("\n=== Testing WITH prediction ===")
    start_time = time.time()
    mean_prior_with_pred, mean_gnc_with_pred = train_gnc(
        seed=42, student_dim=student_dim, device=device, alpha_teacher=alpha_teacher,
        w_sequences=w_sequences, eps_train=eps_train, num_samples=num_samples,
        batch_size=batch_size, use_prediction=True
    )
    time_with_pred = time.time() - start_time
    
    # Compare results
    logging.info("\n=== Results Comparison ===")
    logging.info(f"Without prediction: mean_prior={mean_prior_no_pred:.6f}, mean_gnc={mean_gnc_no_pred:.6f}, time={time_no_pred:.2f}s")
    logging.info(f"With prediction:    mean_prior={mean_prior_with_pred:.6f}, mean_gnc={mean_gnc_with_pred:.6f}, time={time_with_pred:.2f}s")
    
    # Check accuracy
    prior_diff = abs(mean_prior_no_pred - mean_prior_with_pred)
    gnc_diff = abs(mean_gnc_no_pred - mean_gnc_with_pred)
    
    logging.info(f"Differences: prior={prior_diff:.6f}, gnc={gnc_diff:.6f}")
    
    # Check speedup
    if time_no_pred > 0:
        speedup = time_no_pred / time_with_pred
        logging.info(f"Speedup: {speedup:.2f}x")
    
    # Determine if test passed
    accuracy_threshold = 1e-3  # Allow small differences due to randomness
    if prior_diff < accuracy_threshold and gnc_diff < accuracy_threshold:
        logging.info("✅ Test PASSED: Results are within acceptable accuracy threshold")
        return True
    else:
        logging.warning("⚠️  Test FAILED: Results differ beyond acceptable threshold")
        return False

if __name__ == "__main__":
    success = test_gnc_optimized()
    exit(0 if success else 1)
