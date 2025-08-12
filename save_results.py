import numpy as np
import pandas as pd

def save_results_to_csv(
    gnc_gen_losses: np.ndarray,
    gd_gen_losses: np.ndarray,
    gnc_theoretical_losses: np.ndarray,
    gnc_theoretical_asymptotic_losses: np.ndarray,
    student_dims,
    num_seeds,
    results_filename,
    results_dir,
    gd,
    gnc,
):
    """
    Save G&C and GD results to a CSV file.
    Each row: (student_dim, [gnc_gen_seed_0, ..., gnc_gen_seed_N, gd_gen_seed_0, ..., gd_gen_seed_N])
    """
    csv_path = results_dir / results_filename
    rows = []
    for s_idx, student_dim in enumerate(student_dims):
        row = {
            "student_dim": student_dim,
        }

        if gnc:
            # Add G&C results for all seeds
            for seed in range(num_seeds):
                row[f"gnc_gen_seed={seed}"] = gnc_gen_losses[s_idx, seed]
                row[f"gnc_gen_seed={seed}_theoretical"] = gnc_theoretical_losses[s_idx, seed]
                row[f"gnc_gen_seed={seed}_asymptotic"] = gnc_theoretical_asymptotic_losses[s_idx, seed]
        
        if gd:
            # Add GD results for all seeds
            for seed in range(num_seeds):
                row[f"gd_gen_seed={seed}"] = gd_gen_losses[s_idx, seed]
        
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
