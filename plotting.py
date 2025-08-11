import os

import numpy as np
import matplotlib.pyplot as plt

from utils import median_iqr


def plot(student_dims: list,
         gnc_gen_losses: np.ndarray,
         gd_gen_losses: np.ndarray,
         gnc_mean_priors: np.ndarray,
         gnc_theoretical_losses: np.ndarray,
         gnc_theoretical_asymptotic_losses: np.ndarray,
         teacher_ranks: list,
         sequence_length: int,
         plot_filename: str,
         figures_dir: str = './figures',
         gnc: bool = True,
         gd: bool = True):

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    height = 10  # Height for side-by-side plots
    width = 30  # Increased width for side-by-side plots

    fig, (ax, ax2) = plt.subplots(figsize=(width, height), nrows=1, ncols=2, sharey=True)

    fig.suptitle(f'Sequence Length = {sequence_length}', fontsize=14)

    teacher_rank_labels = [f"Rank={rank}" for rank in teacher_ranks] if len(teacher_ranks) > 1 else [""]
    for t_idx, teacher_rank in enumerate(teacher_ranks):
        if gnc:
            gnc_med, gnc_iqr = median_iqr(gnc_gen_losses[t_idx])
            gnc_mean_prior_med, gnc_mean_prior_iqr = median_iqr(gnc_mean_priors[t_idx])
            ax.errorbar(
                student_dims, gnc_med, yerr=gnc_iqr,
                fmt="o-", capsize=3, label=f"G&C {teacher_rank_labels[t_idx]}",
                linewidth=2.5, elinewidth=1.5
            )
            ax.errorbar(
                student_dims, gnc_mean_prior_med, yerr=gnc_mean_prior_iqr,
                fmt="o-", capsize=3, label=f"G&C Prior {teacher_rank_labels[t_idx]}",
                linewidth=2.5, elinewidth=1.5
            )
            gnc_theoretical_med, gnc_theoretical_iqr = median_iqr(gnc_theoretical_losses[t_idx])
            ax.errorbar(
                student_dims, gnc_theoretical_med, yerr=gnc_theoretical_iqr,
                fmt="o-", capsize=3, label=f"G&C Theoretical {teacher_rank_labels[t_idx]}",
                linewidth=2.5, elinewidth=1.5
            )
            gnc_theoretical_asymptotic_med, gnc_theoretical_asymptotic_iqr = median_iqr(gnc_theoretical_asymptotic_losses[t_idx])
            ax.errorbar(
                student_dims, gnc_theoretical_asymptotic_med, yerr=gnc_theoretical_asymptotic_iqr,
                fmt="o-", capsize=3, label=f"G&C Theoretical Asymptotic (Rank={teacher_rank})",
                linewidth=2.5, elinewidth=1.5
            )
            
        if gd:
            gd_med, gd_iqr = median_iqr(gd_gen_losses[t_idx])
            ax.errorbar(
                student_dims, gd_med, yerr=gd_iqr,
                fmt="s--", capsize=3, label=f'GD (Rank={teacher_rank})',
                linewidth=2.5, elinewidth=1.5
            )
            
    ax.set_ylabel("Generalization Loss", fontsize="xx-large")
    ax.set_xlabel("Student Dimension", fontsize="xx-large")
    ax.grid(True)
    ax.legend(fontsize="x-large", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    # plot for each seed to compare gnc and the theoretical loss
    # Only plot for the first teacher rank to avoid overcrowding
    t_idx = 0  # Use only the first teacher rank
    teacher_rank = teacher_ranks[t_idx]
    
    # Define colors for each seed
    colors = plt.cm.tab10(np.linspace(0, 1, gnc_gen_losses.shape[-1]))
    
    # Plot actual G&C losses for all seeds
    for seed_idx in range(gnc_gen_losses.shape[-1]):
        ax2.plot(student_dims, gnc_gen_losses[t_idx, :, seed_idx], 
                color=colors[seed_idx], marker='o', linewidth=1.5, markersize=4,
                label=f"Seed {seed_idx} (Actual)")
    
    # Plot theoretical G&C losses for all seeds
    for seed_idx in range(gnc_theoretical_losses.shape[-1]):
        ax2.plot(student_dims, gnc_theoretical_losses[t_idx, :, seed_idx], 
                color=colors[seed_idx], marker='s', linestyle='--', linewidth=1.5, markersize=4,
                label=f"Seed {seed_idx} (Theoretical)")
    
    # # Add a text annotation to show which teacher rank is being plotted
    # ax2.text(0.02, 0.98, f'Teacher Rank = {teacher_rank}', 
    #          transform=ax2.transAxes, fontsize=12, 
    #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel("Student Dimension", fontsize="xx-large")
    ax2.set_ylabel("Generalization Loss", fontsize="xx-large")
    ax2.grid(True)
    ax2.legend(fontsize="medium", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    plt.tight_layout()
    
    outfile_base = os.path.join(figures_dir, plot_filename)
    fig.savefig(outfile_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outfile_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)