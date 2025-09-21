#!/usr/bin/env python3
"""
Script to generate a histogram of training losses for the Guess and Check (G&C) method.
"""

import argparse
import logging
import time
import torch
import numpy as np
import os

from training import train_gnc
from generator import generate_w_sequences
from plotting import plot_training_loss_histogram
from utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Generate G&C training loss histogram')
    
    # Model parameters
    parser.add_argument('--student_dim', type=int, default=100, help='Student dimension')
    parser.add_argument('--alpha_teacher', type=float, default=0.5, help='Teacher parameter')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length')
    parser.add_argument('--num_sequences', type=int, default=1, help='Number of sequences')
    
    # G&C parameters
    parser.add_argument('--eps_train', type=float, default=0.01, help='Training epsilon threshold')
    parser.add_argument('--num_samples', type=int, default=10000000, help='Number of student samples')
    parser.add_argument('--batch_size', type=int, default=100000, help='Batch size for processing')
    
    # Experiment parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7], help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    # Output parameters
    parser.add_argument('--figures_dir', type=str, default='./figures', help='Directory to save figures')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = setup_logging(args.log_dir, timestamp)
    logging.info(f'Args: {args}')
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logging.info(f'Using device: {device}')
    
    training_losses_all = []
    
    # Set random seed
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        logging.info(f'Random seeds set to: {seed}')
    
        # Generate w sequences
        logging.info(f'Generating {args.num_sequences} w sequences of length {args.sequence_length}')
        w_sequences = generate_w_sequences(args.sequence_length, args.num_sequences, device)
        
        # Run G&C training with loss collection
        logging.info(f'Running G&C training with {args.num_samples} samples...')
        logging.info(f'Parameters: student_dim={args.student_dim}, eps_train={args.eps_train}, '
                    f'alpha_teacher={args.alpha_teacher}')
        
        start_time = time.time()
        result = train_gnc(
            seed=seed,
            student_dim=args.student_dim,
            device=device,
            alpha_teacher=args.alpha_teacher,
            w_sequences=w_sequences,
            eps_train=args.eps_train,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            collect_training_losses=True
        )
        
        if len(result) == 3:
            mean_prior, mean_gnc, training_losses = result
        else:
            mean_prior, mean_gnc = result
            training_losses = None
        
        end_time = time.time()
        logging.info(f'G&C training completed in {end_time - start_time:.2f} seconds')
        logging.info(f'Results: mean_prior={mean_prior:.6f}, mean_gnc={mean_gnc:.6f}')
        
        if training_losses is None:
            logging.error("No training losses collected. Check the train_gnc function.")
            return
    
        logging.info(f'Collected {len(training_losses)} training loss values')
        training_losses_all.append(training_losses)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.figures_dir, exist_ok=True)
    
    # Generate plot filename
    seeds_str = '-'.join(map(str, args.seeds))
    plot_filename = (f'gnc_training_loss_histogram_'
                    f'dim={args.student_dim}_'
                    f'seq_len={args.sequence_length}_'
                    f'eps={args.eps_train}_'
                    f'samples={args.num_samples}_'
                    f'seeds={seeds_str}_'
                    f'time={timestamp}')
    
    # Generate histogram
    logging.info('Generating training loss histogram...')
    plot_training_loss_histogram(
        training_losses_all=training_losses_all,
        eps_train=args.eps_train,
        student_dim=args.student_dim,
        seeds=args.seeds,
        sequence_length=args.sequence_length,
        plot_filename=plot_filename,
        figures_dir=args.figures_dir
    )
    
    logging.info(f'Histogram saved to {args.figures_dir}/{plot_filename}.png')
    print(f'Histogram saved to {args.figures_dir}/{plot_filename}.png')
    
    # Print summary statistics
    for seed_idx, seed in enumerate(args.seeds):
        training_losses_array = np.array(training_losses_all[seed_idx])
        below_epsilon = np.sum(training_losses_array < args.eps_train)
        success_rate = (below_epsilon / len(training_losses_array)) * 100
    
        print(f'\nSummary Statistics: Seed {seed}')
        print(f'Total samples: {len(training_losses_array)}')
        print(f'Below ε_train: {below_epsilon} ({success_rate:.1f}%)')
        print(f'Mean training loss: {np.mean(training_losses_array):.6f}')
        print(f'Median training loss: {np.median(training_losses_array):.6f}')
        print(f'Std training loss: {np.std(training_losses_array):.6f}')
        print(f'Min training loss: {np.min(training_losses_array):.6f}')
        print(f'Max training loss: {np.max(training_losses_array):.6f}')


if __name__ == "__main__":
    main()
