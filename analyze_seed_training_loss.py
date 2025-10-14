#!/usr/bin/env python3
"""
Script to analyze training loss across 1000 seeds with 10,000,000 samples each.
For each seed, generates histograms of training loss (values between 0 and 1)
and identifies seeds with the lowest mean training loss.
"""

import argparse
import logging
import time
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import json

from training import train_gnc
from generator import generate_w_sequences
from utils import setup_logging


def process_seed_worker(process_id, seed_list, args_dict, results_queue, progress_queue, log_file):
    """
    Worker process that analyzes training loss for a specific list of seeds.
    
    Args:
        process_id: ID of this process
        seed_list: List of seeds to process
        args_dict: Experiment arguments
        results_queue: Queue to send results to main process
        progress_queue: Queue to send progress updates to main process
        log_file: Log file path
    """
    # Set up logging for this process
    setup_logging(log_file)
    
    # Setup device
    if args_dict['device'] == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args_dict['device'])
    
    logging.info(f"Process {process_id} started, processing seeds {seed_list}")
    
    for seed in seed_list:
        try:
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Generate w sequences (2 sequences as requested)
            w_sequences = generate_w_sequences(
                args_dict['sequence_length'], 
                args_dict['num_sequences'], 
                device
            )
            
            # Run G&C training with loss collection
            result = train_gnc(
                seed=seed,
                student_dim=args_dict['student_dim'],
                device=device,
                alpha_teacher=args_dict['alpha_teacher'],
                w_sequences=w_sequences,
                eps_train=args_dict['eps_train'],
                num_samples=args_dict['num_samples'],
                batch_size=args_dict['batch_size'],
                collect_training_losses=True
            )
            
            if len(result) == 3:
                mean_prior, mean_gnc, training_losses = result
            else:
                mean_prior, mean_gnc = result
                training_losses = None
            
            if training_losses is None:
                logging.error(f"No training losses collected for seed {seed}")
                continue
            
            # Filter training losses to values between 0 and 1
            training_losses_array = np.array(training_losses)
            filtered_losses = training_losses_array[(training_losses_array >= 0) & (training_losses_array <= 1)]
            
            if len(filtered_losses) == 0:
                logging.warning(f"No training losses in range [0,1] for seed {seed}")
                continue
            
            # Calculate statistics
            mean_loss = np.mean(filtered_losses)
            std_loss = np.std(filtered_losses)
            min_loss = np.min(filtered_losses)
            max_loss = np.max(filtered_losses)
            median_loss = np.median(filtered_losses)
            
            # Create histogram data
            hist, bin_edges = np.histogram(filtered_losses, bins=50, range=(0, 1))
            
            # Send result to main process
            result_data = {
                'seed': seed,
                'process_id': process_id,
                'mean_loss': mean_loss,
                'std_loss': std_loss,
                'min_loss': min_loss,
                'max_loss': max_loss,
                'median_loss': median_loss,
                'num_samples': len(filtered_losses),
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'training_losses': filtered_losses.tolist()
            }
            
            results_queue.put(result_data)
            
            # Send progress update
            progress_queue.put({
                'process_id': process_id,
                'seed': seed,
                'completed': True
            })
            
            logging.info(f"Process {process_id}: Completed seed {seed}, mean loss: {mean_loss:.6f}")
            
        except Exception as e:
            logging.error(f"Process {process_id}: Error processing seed {seed}: {e}")
            # Send error result
            error_result = {
                'seed': seed,
                'process_id': process_id,
                'error': str(e)
            }
            results_queue.put(error_result)
        
        # Clear cache to prevent memory issues
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logging.info(f"Process {process_id} completed")


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze training loss across 1000 seeds')
    
    # Model parameters
    parser.add_argument('--student_dim', type=int, default=100, help='Student dimension')
    parser.add_argument('--alpha_teacher', type=float, default=0.5, help='Teacher parameter')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length')
    parser.add_argument('--num_sequences', type=int, default=2, help='Number of sequences')
    
    # G&C parameters
    parser.add_argument('--eps_train', type=float, default=0.01, help='Training epsilon threshold')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of student samples')
    parser.add_argument('--batch_size', type=int, default=100000, help='Batch size for processing')
    
    # Experiment parameters
    parser.add_argument('--num_seeds', type=int, default=1000, help='Number of seeds to analyze')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./seed_analysis_results', help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    
    return parser.parse_args()


def create_histogram_plot(seed_data, output_dir, timestamp):
    """Create histogram plots for all seeds"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Loss Histograms - {len(seed_data)} Seeds', fontsize=16)
    
    # Plot 1: All histograms overlaid
    ax1 = axes[0, 0]
    for data in seed_data:
        if 'histogram' in data and 'bin_edges' in data:
            bin_centers = (np.array(data['bin_edges'][:-1]) + np.array(data['bin_edges'][1:])) / 2
            ax1.plot(bin_centers, data['histogram'], alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Training Loss')
    ax1.set_ylabel('Frequency')
    ax1.set_title('All Seeds Overlaid')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean loss distribution
    ax2 = axes[0, 1]
    mean_losses = [data['mean_loss'] for data in seed_data if 'mean_loss' in data]
    ax2.hist(mean_losses, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mean Training Loss')
    ax2.set_ylabel('Number of Seeds')
    ax2.set_title('Distribution of Mean Losses')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Top 10 seeds with lowest mean loss
    ax3 = axes[1, 0]
    sorted_data = sorted(seed_data, key=lambda x: x.get('mean_loss', float('inf')))
    top_10 = sorted_data[:10]
    
    for i, data in enumerate(top_10):
        if 'histogram' in data and 'bin_edges' in data:
            bin_centers = (np.array(data['bin_edges'][:-1]) + np.array(data['bin_edges'][1:])) / 2
            ax3.plot(bin_centers, data['histogram'], label=f"Seed {data['seed']}", alpha=0.7)
    ax3.set_xlabel('Training Loss')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Top 10 Seeds (Lowest Mean Loss)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean vs Std scatter plot
    ax4 = axes[1, 1]
    mean_losses = [data['mean_loss'] for data in seed_data if 'mean_loss' in data]
    std_losses = [data['std_loss'] for data in seed_data if 'std_loss' in data]
    ax4.scatter(mean_losses, std_losses, alpha=0.6)
    ax4.set_xlabel('Mean Training Loss')
    ax4.set_ylabel('Std Training Loss')
    ax4.set_title('Mean vs Standard Deviation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f'seed_analysis_histograms_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename


def main():
    args = parse_args()
    
    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = setup_logging(args.log_dir, timestamp)
    logging.info(f'Args: {args}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate seeds
    seeds = list(range(args.num_seeds))
    logging.info(f'Analyzing {len(seeds)} seeds')
    
    # Distribute seeds across processes
    seeds_per_process = len(seeds) // args.num_processes
    remaining_seeds = len(seeds) % args.num_processes
    
    seed_ranges = []
    start_idx = 0
    for i in range(args.num_processes):
        end_idx = start_idx + seeds_per_process + (1 if i < remaining_seeds else 0)
        process_seeds = seeds[start_idx:end_idx]
        seed_ranges.append(process_seeds)
        start_idx = end_idx
    
    # Convert args to dict for serialization
    args_dict = {
        'student_dim': args.student_dim,
        'alpha_teacher': args.alpha_teacher,
        'sequence_length': args.sequence_length,
        'num_sequences': args.num_sequences,
        'eps_train': args.eps_train,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'device': args.device
    }
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create queues for communication
    results_queue = Queue()
    progress_queue = Queue()
    
    # Create and start processes
    processes = []
    for i in range(args.num_processes):
        seed_list = seed_ranges[i]
        if len(seed_list) == 0:
            continue
            
        process = Process(
            target=process_seed_worker,
            args=(i, seed_list, args_dict, results_queue, progress_queue, log_file)
        )
        processes.append(process)
        process.start()
    
    # Monitor progress and collect results
    seed_data = []
    completed_seeds = 0
    total_seeds = len(seeds)
    
    with tqdm(total=total_seeds, desc="Processing seeds") as pbar:
        while completed_seeds < total_seeds:
            # Check for progress updates
            try:
                progress_data = progress_queue.get_nowait()
                completed_seeds += 1
                pbar.update(1)
            except:
                pass
            
            # Check for results
            try:
                result = results_queue.get_nowait()
                if 'error' in result:
                    logging.error(f"Error in seed {result['seed']}: {result['error']}")
                else:
                    seed_data.append(result)
            except:
                pass
            
            # Small sleep to prevent busy waiting
            time.sleep(0.1)
    
    # Wait for all processes to finish
    for process in processes:
        process.join()
    
    logging.info(f"Completed analysis of {len(seed_data)} seeds")
    
    # Sort seeds by mean loss
    seed_data.sort(key=lambda x: x.get('mean_loss', float('inf')))
    
    # Create summary statistics
    mean_losses = [data['mean_loss'] for data in seed_data if 'mean_loss' in data]
    std_losses = [data['std_loss'] for data in seed_data if 'std_loss' in data]
    
    summary_stats = {
        'total_seeds_analyzed': len(seed_data),
        'overall_mean_loss': np.mean(mean_losses),
        'overall_std_loss': np.std(mean_losses),
        'min_mean_loss': np.min(mean_losses),
        'max_mean_loss': np.max(mean_losses),
        'top_10_seeds': [data['seed'] for data in seed_data[:10]],
        'top_10_mean_losses': [data['mean_loss'] for data in seed_data[:10]]
    }
    
    # Save results
    results_filename = f'seed_analysis_results_{timestamp}.json'
    with open(os.path.join(args.output_dir, results_filename), 'w') as f:
        json.dump({
            'summary_stats': summary_stats,
            'seed_data': seed_data
        }, f, indent=2)
    
    # Create CSV with summary
    csv_data = []
    for data in seed_data:
        csv_data.append({
            'seed': data['seed'],
            'mean_loss': data['mean_loss'],
            'std_loss': data['std_loss'],
            'min_loss': data['min_loss'],
            'max_loss': data['max_loss'],
            'median_loss': data['median_loss'],
            'num_samples': data['num_samples']
        })
    
    df = pd.DataFrame(csv_data)
    csv_filename = f'seed_analysis_summary_{timestamp}.csv'
    df.to_csv(os.path.join(args.output_dir, csv_filename), index=False)
    
    # Create histogram plots
    plot_filename = create_histogram_plot(seed_data, args.output_dir, timestamp)
    
    # Print summary
    print(f"\n=== SEED ANALYSIS SUMMARY ===")
    print(f"Total seeds analyzed: {summary_stats['total_seeds_analyzed']}")
    print(f"Overall mean loss: {summary_stats['overall_mean_loss']:.6f}")
    print(f"Overall std loss: {summary_stats['overall_std_loss']:.6f}")
    print(f"Min mean loss: {summary_stats['min_mean_loss']:.6f}")
    print(f"Max mean loss: {summary_stats['max_mean_loss']:.6f}")
    print(f"\nTop 10 seeds with lowest mean loss:")
    for i, (seed, mean_loss) in enumerate(zip(summary_stats['top_10_seeds'], summary_stats['top_10_mean_losses'])):
        print(f"  {i+1:2d}. Seed {seed:4d}: {mean_loss:.6f}")
    
    print(f"\nResults saved to:")
    print(f"  - JSON: {os.path.join(args.output_dir, results_filename)}")
    print(f"  - CSV: {os.path.join(args.output_dir, csv_filename)}")
    print(f"  - Plot: {os.path.join(args.output_dir, plot_filename)}")
    
    logging.info(f"Analysis completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
