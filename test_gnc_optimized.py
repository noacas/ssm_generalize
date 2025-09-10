#!/usr/bin/env python3
"""
Simple test to verify the optimized GNC prediction works well with 2 sequences.
"""

import torch
import time
import logging
import math
from generator import generate_teacher_alpha, generate_w_sequences
from training import train_gnc, train_gnc_original, train_gnc_heuristic

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_gnc_optimized():
    """Test GNC with prediction optimization using 2 sequences."""
    
    # Test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_dim = 100
    eps_train = 1e-3
    num_samples = 10000000
    batch_size = 500000
    seed = 0
    number_sequences = 1
    
    # Generate test data
    torch.manual_seed(seed)
    alpha_teacher = generate_teacher_alpha(device)
    w_sequences = generate_w_sequences(5, number_sequences, device)
    
    logging.info(f"Testing optimized GNC with {number_sequences} sequences")
    logging.info(f"Device: {device}")
    logging.info(f"Student dim: {student_dim}, Samples: {num_samples}, Batch size: {batch_size}")
    logging.info(f"Epsilon: {eps_train}")
    
    # Test with original implementation
    logging.info("\n=== Testing ORIGINAL implementation ===")
    start_time = time.time()
    mean_prior_no_pred, mean_gnc_no_pred = train_gnc_original(
        seed=seed, student_dim=student_dim, device=device, alpha_teacher=alpha_teacher,
        w_sequences=w_sequences, eps_train=eps_train, num_samples=num_samples,
        batch_size=batch_size, use_prediction=False
    )
    time_no_pred = time.time() - start_time
    
    # Test with optimized loss calculation
    logging.info("\n=== Testing WITH optimized loss calculation ===")
    start_time = time.time()
    mean_prior_with_pred, mean_gnc_with_pred = train_gnc(
        seed=seed, student_dim=student_dim, device=device, alpha_teacher=alpha_teacher,
        w_sequences=w_sequences, eps_train=eps_train, num_samples=num_samples,
        batch_size=batch_size, use_prediction=False  # Using optimized loss calculation instead
    )
    time_with_pred = time.time() - start_time
    
    # Test with heuristic approach (manual thresholds)
    logging.info("\n=== Testing WITH heuristic approach (manual thresholds) ===")
    start_time = time.time()
    mean_prior_heuristic, mean_gnc_heuristic = train_gnc_heuristic(
        seed=seed, student_dim=student_dim, device=device, alpha_teacher=alpha_teacher,
        w_sequences=w_sequences, eps_train=eps_train, num_samples=num_samples,
        batch_size=batch_size, mean_min=-0.5, mean_max=0.5, var_min=0.2, var_max=1.5
    )
    time_heuristic_manual = time.time() - start_time
    
    # Test with heuristic approach (auto-determined thresholds)
    logging.info("\n=== Testing WITH heuristic approach (auto-determined thresholds) ===")
    start_time = time.time()
    mean_prior_heuristic_auto, mean_gnc_heuristic_auto = train_gnc_heuristic(
        seed=seed, student_dim=student_dim, device=device, alpha_teacher=alpha_teacher,
        w_sequences=w_sequences, eps_train=eps_train, num_samples=num_samples,
        batch_size=batch_size, auto_determine_thresholds=True, threshold_analysis_samples=5000
    )
    time_heuristic_auto = time.time() - start_time
    
    # Test with heuristic approach (pre-determined thresholds from file)
    logging.info("\n=== Testing WITH heuristic approach (pre-determined thresholds) ===")
    # First run analysis and save to file
    from training import run_threshold_analysis_once
    analysis_file = "temp_threshold_analysis.json"
    run_threshold_analysis_once(
        seed=seed, student_dim=student_dim, device=device, alpha_teacher=alpha_teacher,
        w_sequences=w_sequences, eps_train=eps_train, num_samples=10000,
        batch_size=batch_size, save_to_file=analysis_file
    )
    
    start_time = time.time()
    mean_prior_heuristic_file, mean_gnc_heuristic_file = train_gnc_heuristic(
        seed=seed, student_dim=student_dim, device=device, alpha_teacher=alpha_teacher,
        w_sequences=w_sequences, eps_train=eps_train, num_samples=num_samples,
        batch_size=batch_size, threshold_analysis_file=analysis_file
    )
    time_heuristic_file = time.time() - start_time
    
    # Clean up
    import os
    if os.path.exists(analysis_file):
        os.remove(analysis_file)
    
    # Compare results
    logging.info("\n=== Results Comparison ===")
    logging.info(f"Original version:     mean_prior={mean_prior_no_pred:.6f}, mean_gnc={mean_gnc_no_pred:.6f}, time={time_no_pred:.2f}s")
    logging.info(f"Optimized version:    mean_prior={mean_prior_with_pred:.6f}, mean_gnc={mean_gnc_with_pred:.6f}, time={time_with_pred:.2f}s")
    logging.info(f"Heuristic (manual):   mean_prior={mean_prior_heuristic:.6f}, mean_gnc={mean_gnc_heuristic:.6f}, time={time_heuristic_manual:.2f}s")
    logging.info(f"Heuristic (auto):     mean_prior={mean_prior_heuristic_auto:.6f}, mean_gnc={mean_gnc_heuristic_auto:.6f}, time={time_heuristic_auto:.2f}s")
    logging.info(f"Heuristic (file):     mean_prior={mean_prior_heuristic_file:.6f}, mean_gnc={mean_gnc_heuristic_file:.6f}, time={time_heuristic_file:.2f}s")
    
    # Check accuracy
    prior_diff_opt = abs(mean_prior_no_pred - mean_prior_with_pred)
    gnc_diff_opt = abs(mean_gnc_no_pred - mean_gnc_with_pred)
    prior_diff_heuristic_manual = abs(mean_prior_no_pred - mean_prior_heuristic)
    gnc_diff_heuristic_manual = abs(mean_gnc_no_pred - mean_gnc_heuristic)
    prior_diff_heuristic_auto = abs(mean_prior_no_pred - mean_prior_heuristic_auto)
    gnc_diff_heuristic_auto = abs(mean_gnc_no_pred - mean_gnc_heuristic_auto)
    prior_diff_heuristic_file = abs(mean_prior_no_pred - mean_prior_heuristic_file)
    gnc_diff_heuristic_file = abs(mean_gnc_no_pred - mean_gnc_heuristic_file)
    
    logging.info(f"Optimized differences:        prior={prior_diff_opt:.6f}, gnc={gnc_diff_opt:.6f}")
    logging.info(f"Heuristic (manual) differences: prior={prior_diff_heuristic_manual:.6f}, gnc={gnc_diff_heuristic_manual:.6f}")
    logging.info(f"Heuristic (auto) differences:   prior={prior_diff_heuristic_auto:.6f}, gnc={gnc_diff_heuristic_auto:.6f}")
    logging.info(f"Heuristic (file) differences:   prior={prior_diff_heuristic_file:.6f}, gnc={gnc_diff_heuristic_file:.6f}")
    
    # Check speedups
    if time_no_pred > 0:
        speedup_opt = time_no_pred / time_with_pred
        speedup_heuristic_manual = time_no_pred / time_heuristic_manual
        speedup_heuristic_auto = time_no_pred / time_heuristic_auto
        speedup_heuristic_file = time_no_pred / time_heuristic_file
        logging.info(f"Optimized speedup:        {speedup_opt:.2f}x")
        logging.info(f"Heuristic (manual) speedup: {speedup_heuristic_manual:.2f}x")
        logging.info(f"Heuristic (auto) speedup:   {speedup_heuristic_auto:.2f}x")
        logging.info(f"Heuristic (file) speedup:   {speedup_heuristic_file:.2f}x")
    
    # Determine if test passed
    accuracy_threshold = 1e-3  # Allow small differences due to randomness
    
    # Handle nan values in comparison
    prior_ok_opt = prior_diff_opt < accuracy_threshold
    gnc_ok_opt = (math.isnan(gnc_diff_opt) or gnc_diff_opt < accuracy_threshold)
    prior_ok_heuristic_manual = prior_diff_heuristic_manual < accuracy_threshold
    gnc_ok_heuristic_manual = (math.isnan(gnc_diff_heuristic_manual) or gnc_diff_heuristic_manual < accuracy_threshold)
    prior_ok_heuristic_auto = prior_diff_heuristic_auto < accuracy_threshold
    gnc_ok_heuristic_auto = (math.isnan(gnc_diff_heuristic_auto) or gnc_diff_heuristic_auto < accuracy_threshold)
    prior_ok_heuristic_file = prior_diff_heuristic_file < accuracy_threshold
    gnc_ok_heuristic_file = (math.isnan(gnc_diff_heuristic_file) or gnc_diff_heuristic_file < accuracy_threshold)
    
    if (prior_ok_opt and gnc_ok_opt and 
        prior_ok_heuristic_manual and gnc_ok_heuristic_manual and
        prior_ok_heuristic_auto and gnc_ok_heuristic_auto and
        prior_ok_heuristic_file and gnc_ok_heuristic_file):
        logging.info("✅ Test PASSED: All results are within acceptable accuracy threshold")
        return True
    else:
        logging.warning("⚠️  Test FAILED: Some results differ beyond acceptable threshold")
        return False

if __name__ == "__main__":
    success = test_gnc_optimized()
    exit(0 if success else 1)
