import os

import numpy as np
import matplotlib.pyplot as plt

from utils import filename_extensions, median_iqr


def plot(student_dims: list,
         gnc_gen_losses: np.ndarray,
         gd_gen_losses: np.ndarray,
         teacher_ranks: list,
         sequence_length: int,
         num_measurements: int,
         figures_dir: str = './figures'):
    os.makedirs(figures_dir, exist_ok=True)
    plot_filename = 'results_plot' + filename_extensions(sequence_length, num_measurements)

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    height = 3
    width = 6

    fig, ax = plt.subplots(figsize=(width, height))

    plt.suptitle('Generalization Loss', fontsize=20, y=1)
    plt.title(f'Sequence Length = {sequence_length}', fontsize=12)

    for t_idx, teacher_rank in enumerate(teacher_ranks):
        gnc_med, gnc_iqr = median_iqr(gnc_gen_losses[t_idx])
        gd_med, gd_iqr = median_iqr(gd_gen_losses[t_idx])
        ax.errorbar(
            student_dims, gnc_med, yerr=gnc_iqr,
            fmt="o-", capsize=3, label=f"G&C (Rank={teacher_rank})",
            linewidth=2.5, elinewidth=1.5
        )
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