#!/usr/bin/env python3
"""
Script to recreate training loss histograms from saved JSON data.
This allows you to modify plots without rerunning the expensive experiments.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from plotting import plot_training_loss_histogram


def parse_args():
    parser = argparse.ArgumentParser(description='Recreate G&C training loss histogram from saved data')
    
    parser.add_argument('--data_file', type=str, required=True, 
                       help='Path to the JSON data file (e.g., *_data.json)')
    parser.add_argument('--figures_dir', type=str, default='./figures', 
                       help='Directory to save the recreated plot')
    parser.add_argument('--plot_suffix', type=str, default='_recreated', 
                       help='Suffix to add to the plot filename')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load the data
    print(f"Loading data from {args.data_file}")
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    # Extract data
    training_losses_all = data['training_losses_all']
    seeds = data['seeds']
    eps_train = data['eps_train']
    student_dim = data['student_dim']
    sequence_length = data['sequence_length']
    statistics = data['statistics']
    
    print(f"Loaded data for seeds: {seeds}")
    print(f"Student dim: {student_dim}, Sequence length: {sequence_length}")
    print(f"Epsilon train: {eps_train}")
    print(f"Total samples: {statistics['total_samples']}")
    print(f"Success rate: {statistics['success_rate']:.1f}%")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.figures_dir, exist_ok=True)
    
    # Generate new plot filename
    seeds_str = '-'.join(map(str, seeds))
    plot_filename = (f'gnc_training_loss_histogram_'
                    f'dim={student_dim}_'
                    f'seq_len={sequence_length}_'
                    f'eps={eps_train}_'
                    f'seeds={seeds_str}'
                    f'{args.plot_suffix}')
    
    # Recreate the histogram
    print("Recreating histogram...")
    plot_training_loss_histogram(
        training_losses_all=training_losses_all,
        eps_train=eps_train,
        student_dim=student_dim,
        seeds=seeds,
        sequence_length=sequence_length,
        plot_filename=plot_filename,
        figures_dir=args.figures_dir
    )
    
    print(f"Recreated histogram saved to {args.figures_dir}/{plot_filename}.png")


if __name__ == "__main__":
    main()
