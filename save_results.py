import numpy as np
import pandas as pd

def save_results_to_csv(
    gnc_gen_losses: np.ndarray,
    gnc_variances: np.ndarray,
    gd_gen_losses: np.ndarray,
    gnc_theoretical_losses: np.ndarray,
    gnc_theoretical_asymptotic_losses: np.ndarray,
    student_dims,
    seeds,
    results_filename,
    results_dir,
    gd,
    gnc,
):
    """
    Save G&C and GD results to a CSV file.
    Each row: (student_dim, [gnc_gen_seed_10, ..., gnc_gen_seed_17, gnc_gen_seed_10_variance, ..., gnc_gen_seed_17_variance, gd_gen_seed_10, ..., gd_gen_seed_17])
    """
    csv_path = results_dir / results_filename
    rows = []
    for s_idx, student_dim in enumerate(student_dims):
        row = {
            "student_dim": student_dim,
        }

        if gnc:
            # Add G&C results for all seeds
            for seed_idx, seed in enumerate(seeds):
                row[f"gnc_gen_seed={seed}"] = gnc_gen_losses[s_idx, seed_idx]
                row[f"gnc_gen_seed={seed}_variance"] = gnc_variances[s_idx, seed_idx]
                row[f"gnc_gen_seed={seed}_theoretical"] = gnc_theoretical_losses[s_idx, seed_idx]
                row[f"gnc_gen_seed={seed}_asymptotic"] = gnc_theoretical_asymptotic_losses[s_idx, seed_idx]
        
        if gd:
            # Add GD results for all seeds
            for seed_idx, seed in enumerate(seeds):
                row[f"gd_gen_seed={seed}"] = gd_gen_losses[s_idx, seed_idx]
        
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
