import os

import numpy as np
import matplotlib.pyplot as plt

from utils import filename_extensions, median_iqr


def plot(student_dims: list,
         gnc_gen_losses: np.ndarray,
         gd_gen_losses: np.ndarray,
         gnc_mean_priors: np.ndarray,
         gnc_theoretical_losses: np.ndarray,
         teacher_ranks: list,
         sequence_length: int,
         plot_filename: str,
         figures_dir: str = './figures',
         gnc: bool = True,
         gd: bool = True):

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    height = 3
    width = 6

    fig, ax = plt.subplots(figsize=(width, height))

    plt.title(f'Sequence Length = {sequence_length}', fontsize=12)

    for t_idx, teacher_rank in enumerate(teacher_ranks):
        if gnc:
            gnc_med, gnc_iqr = median_iqr(gnc_gen_losses[t_idx])
            gnc_mean_prior_med, gnc_mean_prior_iqr = median_iqr(gnc_mean_priors[t_idx])
            ax.errorbar(
                student_dims, gnc_med, yerr=gnc_iqr,
                fmt="o-", capsize=3, label=f"G&C (Rank={teacher_rank})",
                linewidth=2.5, elinewidth=1.5
            )
            ax.errorbar(
                student_dims, gnc_mean_prior_med, yerr=gnc_mean_prior_iqr,
                fmt="o-", capsize=3, label=f"G&C Prior (Rank={teacher_rank})",
                linewidth=2.5, elinewidth=1.5
            )
            gnc_theoretical_med, gnc_theoretical_iqr = median_iqr(gnc_theoretical_losses[t_idx])
            ax.errorbar(
                student_dims, gnc_theoretical_med, yerr=gnc_theoretical_iqr,
                fmt="o-", capsize=3, label=f"G&C Theoretical (Rank={teacher_rank})",
                linewidth=2.5, elinewidth=1.5
            )
            
        if gd:
            gd_med, gd_iqr = median_iqr(gd_gen_losses[t_idx])
            ax.errorbar(
                student_dims, gd_med, yerr=gd_iqr,
                fmt="s--", capsize=3, label=f'GD (Rank={teacher_rank})',
                linewidth=2.5, elinewidth=1.5
            )
            
    ax.set_xlabel("Student Dimension", fontsize="xx-large")
    ax.set_ylabel("Generalization Loss", fontsize="xx-large")
    ax.grid(True)

    ax.legend(fontsize="x-large", loc="upper right")

    outfile_base = os.path.join(figures_dir, plot_filename)
    fig.savefig(outfile_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outfile_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)