import os

import numpy as np
import matplotlib.pyplot as plt

from utils import median_iqr


def plot(student_dims: list,
         gnc_gen_losses: np.ndarray,
         gnc_variances: np.ndarray,
         gd_gen_losses: np.ndarray,
         gnc_mean_priors: np.ndarray,
         gnc_theoretical_losses: np.ndarray,
         gnc_theoretical_asymptotic_losses: np.ndarray,
         sequence_length: int,
         plot_filename: str,
         figures_dir: str = './figures',
         gnc: bool = True,
         gd: bool = True,
         seeds: list = None):

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    height = 10  # Height for side-by-side plots
    width = 30  # Increased width for side-by-side plots

    fig, (ax, ax2) = plt.subplots(figsize=(width, height), nrows=1, ncols=1, sharey=False)

    fig.suptitle(f'Sequence Length = {sequence_length}', fontsize=14)
    if gnc:
        gnc_med, gnc_iqr = median_iqr(gnc_gen_losses)
        gnc_mean_prior_med, gnc_mean_prior_iqr = median_iqr(gnc_mean_priors)
        ax.errorbar(
            student_dims, gnc_med, yerr=gnc_iqr,
            fmt="o-", capsize=3, label=f"G&C",
            linewidth=2.5, elinewidth=1.5
        )
        ax.errorbar(
            student_dims, gnc_mean_prior_med, yerr=gnc_mean_prior_iqr,
            fmt="o-", capsize=3, label=f"G&C Prior",
            linewidth=2.5, elinewidth=1.5
        )
        gnc_theoretical_med, gnc_theoretical_iqr = median_iqr(gnc_theoretical_losses)
        ax.errorbar(
            student_dims, gnc_theoretical_med, yerr=gnc_theoretical_iqr,
            fmt="o-", capsize=3, label=f"G&C Theoretical",
            linewidth=2.5, elinewidth=1.5
        )
        # gnc_theoretical_asymptotic_med, gnc_theoretical_asymptotic_iqr = median_iqr(gnc_theoretical_asymptotic_losses)
        # ax.errorbar(
        #     student_dims, gnc_theoretical_asymptotic_med, yerr=gnc_theoretical_asymptotic_iqr,
        #     fmt="o-", capsize=3, label=f"G&C Theoretical Asymptotic",
        #     linewidth=2.5, elinewidth=1.5
        # )
        
    if gd:
        gd_med, gd_iqr = median_iqr(gd_gen_losses)
        ax.errorbar(
            student_dims, gd_med, yerr=gd_iqr,
            fmt="s--", capsize=3, label=f'GD',
            linewidth=2.5, elinewidth=1.5
        )
            
    ax.set_ylabel("Generalization Loss", fontsize="xx-large")
    ax.set_xlabel("Student Dimension", fontsize="xx-large")
    ax.grid(True)
    ax.legend(fontsize="x-large", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    # plot for each seed to compare gnc and the theoretical loss and gd 
    # Define colors for each seed
    colors = plt.cm.tab10(np.linspace(0, 1, gnc_gen_losses.shape[-1]))
    
    # Use actual seed values if provided, otherwise use indices
    if seeds is None:
        seeds = list(range(gnc_gen_losses.shape[-1]))
    
    if gnc:
        # Plot actual G&C losses for all seeds
        for seed_idx, seed in enumerate(seeds):
            ax2.plot(student_dims, gnc_gen_losses[:, seed_idx], 
                    color=colors[seed_idx], marker='o', linewidth=1.5, markersize=4,
                    label=f"Seed {seed} (Actual)")
        
        # Plot theoretical G&C losses for all seeds
        for seed_idx, seed in enumerate(seeds):
            ax2.plot(student_dims, gnc_theoretical_losses[:, seed_idx], 
                    color=colors[seed_idx], marker='s', linestyle='--', linewidth=1.5, markersize=4,
                    label=f"Seed {seed} (Theoretical)")
    if gd:
        # Plot GD losses for all seeds
        for seed_idx, seed in enumerate(seeds):
            ax2.plot(student_dims, gd_gen_losses[:, seed_idx], 
                    color=colors[seed_idx], marker='^', linestyle=':', linewidth=2.0, markersize=6,
                    label=f"Seed {seed} (GD)")
    
    ax2.set_xlabel("Student Dimension", fontsize="xx-large")
    ax2.set_ylabel("Generalization Loss", fontsize="xx-large")
    ax2.grid(True)
    ax2.legend(fontsize="medium", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    plt.tight_layout()
    
    outfile_base = os.path.join(figures_dir, plot_filename)
    fig.savefig(outfile_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outfile_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if gnc:
        for seed_idx in range(gnc_gen_losses.shape[-1]):
            fig, ax = plt.subplots(figsize=(width, height), nrows=1, ncols=1, sharey=False)
            ax.plot(student_dims, gnc_gen_losses[:, seed_idx], 
                    color=colors[seed_idx], marker='o', linewidth=1.5, markersize=4,
                    label=f"Seed {seed_idx} (Actual)")
            ax.plot(student_dims, gnc_theoretical_losses[:, seed_idx], 
                    color=colors[seed_idx], marker='s', linestyle='--', linewidth=1.5, markersize=4,
                    label=f"Seed {seed_idx} (Theoretical)")
            ax.set_xlabel("Student Dimension", fontsize="xx-large")
            ax.set_ylabel("Generalization Loss", fontsize="xx-large")
            ax.grid(True)
            ax.legend(fontsize="medium", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
            plt.tight_layout()
            outfile_base = os.path.join(figures_dir, plot_filename)
            fig.savefig(outfile_base + f"_seed_{seed_idx}.png", dpi=300, bbox_inches="tight")
            fig.savefig(outfile_base + f"_seed_{seed_idx}.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    student_dims = [10, 20, 30, 40, 50]
    seeds = [0,1]
    gnc_gen_losses = np.random.rand(len(student_dims), len(seeds))
    gd_gen_losses = np.random.rand(len(student_dims), len(seeds))
    gnc_mean_priors = np.random.rand(len(student_dims), len(seeds))
    gnc_theoretical_losses = np.random.rand(len(student_dims), len(seeds))
    gnc_theoretical_asymptotic_losses = np.random.rand(len(student_dims), len(seeds))
    sequence_length = 5
    plot(student_dims, gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses, sequence_length, "test", gnc=True, gd=False)


def plot_training_loss_histogram(training_losses_all, eps_train, student_dim, seeds, 
                                sequence_length, plot_filename, figures_dir='./figures'):
    """
    Create a histogram of training losses for G&C method across multiple seeds.
    
    Args:
        training_losses_all: List of lists of training loss values (one list per seed)
        eps_train: Epsilon threshold for training loss
        student_dim: Student dimension
        seeds: List of random seeds used
        sequence_length: Sequence length
        plot_filename: Base filename for the plot
        figures_dir: Directory to save the plot
    """
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    
    # Flatten all training losses for overall histogram
    all_training_losses = []
    for losses in training_losses_all:
        all_training_losses.extend(losses)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Overall histogram of all seeds combined
    n, bins, patches = ax1.hist(all_training_losses, bins=50, alpha=0.7, color='skyblue', 
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
    
    # Plot 2: Overlaid histograms for each seed
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    for i, (seed, losses) in enumerate(zip(seeds, training_losses_all)):
        ax2.hist(losses, bins=100, alpha=0.6, color=colors[i], 
                label=f'Seed {seed} (n={len(losses)})', density=True)
    
    ax2.axvline(x=eps_train, color='red', linestyle='--', linewidth=2, 
                label=f'ε_train = {eps_train:.4f}')
    
    # Set x-axis limits to 0-1
    ax2.set_xlim(0, 1)
    
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
    
    # Save raw data
    import json
    data_filename = outfile_base + "_data.json"
    
    # Convert numpy arrays to Python lists for JSON serialization
    training_losses_all_serializable = []
    for losses in training_losses_all:
        if isinstance(losses, list):
            training_losses_all_serializable.append(losses)
        else:
            training_losses_all_serializable.append(losses.tolist())
    
    data_to_save = {
        'training_losses_all': training_losses_all_serializable,
        'seeds': seeds,
        'eps_train': float(eps_train),
        'student_dim': int(student_dim),
        'sequence_length': int(sequence_length),
        'statistics': {
            'total_samples': int(total_samples),
            'below_epsilon': int(below_epsilon),
            'success_rate': float(success_rate),
            'mean_loss': float(mean_loss),
            'median_loss': float(median_loss),
            'std_loss': float(std_loss)
        }
    }
    
    with open(data_filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Training loss histogram saved to {outfile_base}.png and {outfile_base}.pdf")
    print(f"Raw data saved to {data_filename}")
    print(f"Overall Statistics: {stats_text.replace(chr(10), ', ')}")