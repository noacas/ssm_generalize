#!/usr/bin/env python3
"""
Script to create customized training loss histograms from saved data.
Allows you to modify plot parameters like x-axis limits, bins, colors, etc.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Create customized G&C training loss histogram from saved data')
    
    parser.add_argument('--data_file', type=str, required=True, 
                       help='Path to the JSON data file (e.g., *_data.json)')
    parser.add_argument('--figures_dir', type=str, default='./figures', 
                       help='Directory to save the customized plot')
    parser.add_argument('--plot_suffix', type=str, default='_customized', 
                       help='Suffix to add to the plot filename')
    
    # Plot customization options
    parser.add_argument('--clip_min', type=float, default=0.0, 
                       help='Minimum value to clip data (values below this are excluded)')
    parser.add_argument('--clip_max', type=float, default=1.0, 
                       help='Maximum value to clip data (values above this are excluded)')
    parser.add_argument('--bins', type=int, default=50, 
                       help='Number of histogram bins')
    parser.add_argument('--figsize_width', type=int, default=20, 
                       help='Figure width')
    parser.add_argument('--figsize_height', type=int, default=8, 
                       help='Figure height')
    parser.add_argument('--alpha', type=float, default=0.6, 
                       help='Transparency level for histograms')
    
    return parser.parse_args()


def create_customized_histogram(training_losses_all, eps_train, student_dim, seeds, 
                               sequence_length, plot_filename, figures_dir, args):
    """
    Create a customized histogram with user-specified parameters.
    """
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    
    # Flatten all training losses for overall histogram
    all_training_losses = []
    for losses in training_losses_all:
        all_training_losses.extend(losses)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(args.figsize_width, args.figsize_height))
    
    # Plot 1: Overall histogram of all seeds combined
    n, bins, patches = ax1.hist(all_training_losses, bins=args.bins, alpha=0.7, color='skyblue', 
                               edgecolor='black', linewidth=0.5)
    
    # Add vertical line for epsilon threshold
    ax1.axvline(x=eps_train, color='red', linestyle='--', linewidth=2, 
                label=f'ε_train = {eps_train:.4f}')
    
    # Calculate overall statistics
    mean_loss = np.mean(all_training_losses)
    median_loss = np.median(all_training_losses)
    std_loss = np.std(all_training_losses)
    below_epsilon = np.sum(np.array(all_training_losses) < eps_train)
    total_samples = len(all_training_losses)
    success_rate = (below_epsilon / total_samples) * 100
    
    # Add statistics text
    stats_text = (f'Total samples: {total_samples}\n'
                 f'Below ε_train: {below_epsilon} ({success_rate:.1f}%)\n'
                 f'Mean: {mean_loss:.4f}\n'
                 f'Median: {median_loss:.4f}\n'
                 f'Std: {std_loss:.4f}')
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Training Loss', fontsize=16)
    ax1.set_ylabel('Frequency', fontsize=16)
    ax1.set_title(f'G&C Training Loss Distribution (All Seeds Combined)\n'
                 f'Student Dim: {student_dim}, Seeds: {seeds}, Seq Len: {sequence_length}', 
                 fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overlaid histograms for each seed (clipped data)
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    for i, (seed, losses) in enumerate(zip(seeds, training_losses_all)):
        # Clip values to specified range
        losses_array = np.array(losses)
        clipped_losses = losses_array[(losses_array >= args.clip_min) & (losses_array <= args.clip_max)]
        
        ax2.hist(clipped_losses, bins=args.bins, alpha=args.alpha, color=colors[i], 
                label=f'Seed {seed} (n={len(clipped_losses)})', density=True)
    
    ax2.axvline(x=eps_train, color='red', linestyle='--', linewidth=2, 
                label=f'ε_train = {eps_train:.4f}')
    
    # Set x-axis limits to match clipping range
    ax2.set_xlim(args.clip_min, args.clip_max)
    
    ax2.set_xlabel('Training Loss', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    ax2.set_title(f'G&C Training Loss Distribution by Seed\n'
                 f'Student Dim: {student_dim}, Seq Len: {sequence_length}', 
                 fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    outfile_base = os.path.join(figures_dir, plot_filename)
    fig.savefig(outfile_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outfile_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Customized histogram saved to {outfile_base}.png and {outfile_base}.pdf")
    print(f"Parameters used: clip=({args.clip_min}, {args.clip_max}), bins={args.bins}, alpha={args.alpha}")


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
    
    print(f"Loaded data for seeds: {seeds}")
    print(f"Student dim: {student_dim}, Sequence length: {sequence_length}")
    print(f"Epsilon train: {eps_train}")
    
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
    
    # Create customized histogram
    print("Creating customized histogram...")
    create_customized_histogram(
        training_losses_all=training_losses_all,
        eps_train=eps_train,
        student_dim=student_dim,
        seeds=seeds,
        sequence_length=sequence_length,
        plot_filename=plot_filename,
        figures_dir=args.figures_dir,
        args=args
    )


if __name__ == "__main__":
    main()
