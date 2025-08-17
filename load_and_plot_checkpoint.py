import os
import ast
import numpy as np
from pathlib import Path
from checkpoint import CheckpointManager
from plotting import plot
from datetime import datetime
from save_results import save_results_to_csv


def print_checkpoint_summary(results):
    """Print a summary of the checkpoint data."""
    print("=== Checkpoint Summary ===")
    print(f"Completed experiments: {results.get('completed_experiments', 'N/A')}")
    print(f"Total experiments: {results.get('total_experiments', 'N/A')}")
    
    if 'completed_experiments' in results and 'total_experiments' in results:
        progress = (results['completed_experiments'] / results['total_experiments']) * 100
        print(f"Progress: {progress:.1f}%")
    
    print("\n=== Data Arrays ===")
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
            if value.size > 0:
                print(f"  - Non-zero elements: {np.count_nonzero(value)}")
                print(f"  - Min: {np.min(value):.6f}, Max: {np.max(value):.6f}")
                print(f"  - Mean: {np.mean(value):.6f}")


def main():
    """Load the latest checkpoint and create a plot."""
    # Define the checkpoint directory
    checkpoint_dir = Path("test_results/checkpoints")
    
    # Find the latest checkpoint
    results = CheckpointManager.load_latest_checkpoint(checkpoint_dir)
    
    if results is None:
        print("No checkpoint files found!")
        return
    
    # Print summary
    print_checkpoint_summary(results)
    
    # Extract the data arrays needed for plotting
    student_dims = results.get('student_dims')
    sequence_length = results.get('sequence_length')
    gnc_gen_losses = results.get('gnc_gen_losses')
    gd_gen_losses = results.get('gd_gen_losses')
    gnc_mean_priors = results.get('gnc_mean_priors')
    gnc_theoretical_losses = results.get('gnc_theoretical_losses')
    gnc_theoretical_asymptotic_losses = results.get('gnc_theoretical_asymptotic_losses')
    num_seeds = results.get('num_seeds')
    figures_dir = results.get('figures_dir')
    results_dir = results.get('results_dir')
    
    # Check if we have the required data and it's a numpy array
    if gnc_gen_losses is None or not isinstance(gnc_gen_losses, np.ndarray):
        print("Error: gnc_gen_losses not found or not a numpy array in checkpoint!")
        print(f"Type of gnc_gen_losses: {type(gnc_gen_losses)}")
        return
    
    # Generate plot filename based on checkpoint timestamp

    plot_filename = f"checkpoint_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\nCreating plot: {plot_filename}")
    print(f"Student dimensions: {student_dims}")
    print(f"Sequence length: {sequence_length}")
    
    # Create the plot
    try:
        plot(
            student_dims=student_dims,
            gnc_gen_losses=gnc_gen_losses,
            gd_gen_losses=gd_gen_losses if gd_gen_losses is not None and isinstance(gd_gen_losses, np.ndarray) else np.zeros_like(gnc_gen_losses),
            gnc_mean_priors=gnc_mean_priors if gnc_mean_priors is not None and isinstance(gnc_mean_priors, np.ndarray) else np.zeros_like(gnc_gen_losses),
            gnc_theoretical_losses=gnc_theoretical_losses if gnc_theoretical_losses is not None and isinstance(gnc_theoretical_losses, np.ndarray) else np.zeros_like(gnc_gen_losses),
            gnc_theoretical_asymptotic_losses=gnc_theoretical_asymptotic_losses if gnc_theoretical_asymptotic_losses is not None and isinstance(gnc_theoretical_asymptotic_losses, np.ndarray) else np.zeros_like(gnc_gen_losses),
            sequence_length=sequence_length,
            plot_filename=plot_filename,
            figures_dir=str(figures_dir),
            gnc=gnc_gen_losses is not None and isinstance(gnc_gen_losses, np.ndarray) and np.any(gnc_gen_losses != 0),
            gd=gd_gen_losses is not None and isinstance(gd_gen_losses, np.ndarray) and np.any(gd_gen_losses != 0)
        )
        
        print(f"Plot saved to: {figures_dir}/{plot_filename}.png")
        print(f"Plot saved to: {figures_dir}/{plot_filename}.pdf")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Save the results to a CSV file
    results_filename = f"checkpoint_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_path = results_dir / results_filename
    save_results_to_csv(
        gnc_gen_losses,
        gd_gen_losses,
        gnc_theoretical_losses,
        gnc_theoretical_asymptotic_losses,
        student_dims,
        num_seeds=num_seeds,
        results_filename=results_filename,
        results_dir=results_dir,
        gd=gd_gen_losses is not None and isinstance(gd_gen_losses, np.ndarray) and np.any(gd_gen_losses != 0),
        gnc=gnc_gen_losses is not None and isinstance(gnc_gen_losses, np.ndarray) and np.any(gnc_gen_losses != 0)
    )
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
