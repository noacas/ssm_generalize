import numpy as np
import torch
from torch.distributions import Uniform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# Create output directory
output_dir = ""
os.makedirs(output_dir, exist_ok=True)


# Parameters
d_values = list(range(3, 16))
n = 3
k = 3
alpha = 1
epsilon_test = 1e-2
c = 0.1
delta = (1 - c) / 4
epsilon_train = epsilon_test * c
num_tries_for_test_data = 1000 # amount of tries to fit specific test dat, we would have it * d**2
repeat_num = 10000


def generate_training_data(n_datasets=1000):
    """Generate training datasets"""
    # Each dataset has N sequences of length k
    # x_i^(n) ~ N(0,1)
    datasets = np.random.normal(0, 1, (n_datasets, n, k))
    return datasets


def generate_student_parameters(d, n_samples=1000):
    """Generate random student parameters"""
    # Student parameters: a_i ~ N(0,1)
    A_samples = np.random.normal(0, 1, (n_samples, d))
    return A_samples


def compute_power_sums(A_samples):
    """Compute S_1 and S_2 for each sample"""
    # S_m = sum(a_i^m) - alpha^m
    S_1 = np.sum(A_samples, axis=1) - alpha
    S_2 = np.sum(A_samples ** 2, axis=1) - alpha ** 2
    return S_1, S_2


def compute_empirical_statistics(datasets):
    """Compute p_1, p_2, p_3 for each dataset"""
    # For k=3: x_1, x_2, x_3
    x_1 = datasets[:, :, 0]  # shape: (n_datasets, N)
    x_2 = datasets[:, :, 1]  # shape: (n_datasets, N)

    p_1 = np.mean(x_1 ** 2, axis=1)  # shape: (n_datasets,)
    p_2 = np.mean(x_2 ** 2, axis=1)  # shape: (n_datasets,)
    p_3 = np.mean(x_1 * x_2, axis=1)  # shape: (n_datasets,)

    return p_1, p_2, p_3


def compute_losses(A_samples, datasets):
    """Compute training and population losses"""
    S_1, S_2 = compute_power_sums(A_samples)
    p_1, p_2, p_3 = compute_empirical_statistics(datasets)

    # Broadcasting: S_1, S_2 shape (n_param_samples,), p_1, p_2, p_3 shape (n_datasets,)
    S_1_expanded = S_1[:, np.newaxis]  # shape: (n_param_samples, 1)
    S_2_expanded = S_2[:, np.newaxis]  # shape: (n_param_samples, 1)

    # Training loss: L_train = S_2^2 * p_1 + S_1^2 * p_2 + 2*S_1*S_2*p_3
    L_train = (S_2_expanded ** 2 * p_1[np.newaxis, :] +
               S_1_expanded ** 2 * p_2[np.newaxis, :] +
               2 * S_1_expanded * S_2_expanded * p_3[np.newaxis, :])

    # Population loss: L_pop = S_1^2 + S_2^2
    L_pop = S_1_expanded ** 2 + S_2_expanded ** 2

    return L_train, L_pop


def check_event_F(datasets):
    """Check if empirical statistics satisfy event F"""
    p_1, p_2, p_3 = compute_empirical_statistics(datasets)

    # Event F: {p_1, p_2 >= 1-delta} ∩ {|p_3| <= delta}
    condition_1 = (p_1 >= 1 - delta) & (p_2 >= 1 - delta)
    condition_2 = np.abs(p_3) <= delta

    event_F = condition_1 & condition_2
    return event_F


def run_simulation(d, n_batches=100000, batch_size=1000):
    total_F, total_B, total_F_and_B = 0, 0, 0
    total_A, total_A_and_B, total_A_and_F, total_A_and_B_and_F = 0, 0, 0, 0
    total_samples = 0

    for i in range(n_batches):
        # Generate samples
        A_samples = generate_student_parameters(d, batch_size)
        datasets = generate_training_data(batch_size)

        # Compute losses
        L_train, L_pop = compute_losses(A_samples, datasets)

        # Check events
        event_F = check_event_F(datasets)

        # For each combination of (parameter_sample, dataset), check event B and A
        event_B = L_train < epsilon_train
        event_A = L_pop < epsilon_test

        total_F += np.sum(event_F)
        total_B += np.sum(event_B)
        total_A += np.sum(event_A)

        # Count joint events
        joint_F_and_B = event_F & event_B
        joint_A_and_B = event_A & event_B
        joint_A_and_F = event_A & event_F
        joint_A_and_B_and_F = event_A & event_B & event_F

        total_F_and_B += np.sum(joint_F_and_B)
        total_A_and_B += np.sum(joint_A_and_B)
        total_A_and_F += np.sum(joint_A_and_F)
        total_A_and_B_and_F += np.sum(joint_A_and_B_and_F)

        total_samples += batch_size

    # Calculate conditional probabilities
    # P(A|B ∩ F) = P(A ∩ B ∩ F) / P(B ∩ F)
    prob_A_given_B_and_F = total_A_and_B_and_F / total_F_and_B if total_F_and_B > 0 else 0

    # P(A|B) = P(A ∩ B) / P(B)
    prob_A_given_B = total_A_and_B / total_B if total_B > 0 else 0

    # P(A|B ∩ ¬F) = P(A ∩ B ∩ ¬F) / P(B ∩ ¬F)
    total_B_and_not_F = total_B - total_F_and_B
    total_A_and_B_and_not_F = total_A_and_B - total_A_and_B_and_F
    prob_A_given_B_and_not_F = total_A_and_B_and_not_F / total_B_and_not_F if total_B_and_not_F > 0 else 0

    # Additional useful probabilities
    prob_F = total_F / total_samples
    prob_B = total_B / total_samples
    prob_A = total_A / total_samples
    prob_F_given_B = total_F_and_B / total_B if total_B > 0 else 0
    prob_B_given_F = total_F_and_B / total_F if total_F > 0 else 0

    print(f"  P(A|B ∩ F) = {prob_A_given_B_and_F:.4f}, P(A|B) = {prob_A_given_B:.4f}, "
          f"P(A|B ∩ ¬F) = {prob_A_given_B_and_not_F:.4f}, "
          )

    results = {
        'prob_A|B_and_F': float(prob_A_given_B_and_F),
        'prob_A|B': float(prob_A_given_B),
        'prob_A|B_and_not_F': float(prob_A_given_B_and_not_F),
        'prob_F': float(prob_F),
        'prob_B': float(prob_B),
        'prob_A': float(prob_A),
        'prob_F|B': float(prob_F_given_B),
        'prob_B|F': float(prob_B_given_F),
        'total_F_and_B': int(total_F_and_B),
        'total_A_and_B_and_F': int(total_A_and_B_and_F)
    }

    return results


def main():
    # Store results for analysis
    all_results = []
    print('N =', n)
    for d in d_values:
        print('d =', d)
        results = run_simulation(d)
        if len(results):
            all_results.append({
                'd': d,
                **results
            })

    results_file = os.path.join(output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {results_file}")

    # Create a summary DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    summary_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")




if __name__ == '__main__':
    main()