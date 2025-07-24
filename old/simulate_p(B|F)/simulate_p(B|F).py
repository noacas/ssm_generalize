import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm
import pandas as pd


# Parameters
d_values = list(range(3, 10))
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

    for i in range(n_batches):
        # Generate samples
        A_samples = generate_student_parameters(d, batch_size)
        datasets = generate_training_data(batch_size)

        # Compute losses
        L_train, L_pop = compute_losses(A_samples, datasets)

        # Check events
        event_F = check_event_F(datasets)

        # For each combination of (parameter_sample, dataset), check event B
        event_B = L_train < epsilon_train

        total_F += np.sum(event_F)

        total_B += np.sum(event_B)

        # P(F ∩ B) - need to broadcast correctly
        event_F_expanded = event_F[np.newaxis, :]  # shape: (1, n_datasets)
        joint_event = event_F_expanded & event_B  # shape: (n_param_samples, n_datasets)
        total_F_and_B += np.sum(joint_event)


    # P(F|B) = P(F ∩ B) / P(B)
    if total_B > 0:
        prob_F_given_B = total_F_and_B / total_B
    else:
        prob_F_given_B = 0

    # Also compute P(B|F)
    prob_B_given_F = total_F_and_B / total_F if total_F > 0 else 0

    prob_F = total_F / (n_batches * batch_size)
    prob_B = total_B / (n_batches * batch_size)

    results = {
        'prob_F': prob_F,
        'prob_B': prob_B,
        'prob_F_given_B': prob_F_given_B,
        'prob_B_given_F': prob_B_given_F,
        'independence_diff': abs(prob_F_given_B - prob_F),
    }
    return results


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_and_save(all_results):
    """Plot and save simulation results"""
    if not all_results:
        print("No results to plot")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(all_results)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Simulation Results: Event Analysis Across Dimensions', fontsize=16, fontweight='bold')

    # Plot 3: Independence analysis
    ax3 = axes[1, 0]
    ax3.plot(df['d'], df['independence_diff'], 'D-', color='red', linewidth=2, markersize=6)
    ax3.set_xlabel('Dimension (d)')
    ax3.set_ylabel('|P(F|B) - P(F)|')
    ax3.set_title('Independence Measure')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, df['independence_diff'].max() * 1.1)

    # Add horizontal line at y=0 for reference
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Plot 4: Comparison of P(F|B) vs P(F)
    ax4 = axes[1, 1]
    ax4.plot(df['d'], df['prob_F'], 'o-', label='P(F)', linewidth=2, markersize=6)
    ax4.plot(df['d'], df['prob_F_given_B'], '^-', label='P(F|B)', linewidth=2, markersize=6)
    ax4.set_xlabel('Dimension (d)')
    ax4.set_ylabel('Probability')
    ax4.set_title('Independence Check: P(F) vs P(F|B)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig('simulate_p(B|F).png', dpi=300, bbox_inches='tight')
    plt.savefig('simulate_p(B|F).pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print(f"{'Dimension':<10} {'P(F)':<8} {'P(B)':<8} {'P(F|B)':<8} {'P(B|F)':<8} {'Independence':<12}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['d']:<10} {row['prob_F']:<8.4f} {row['prob_B']:<8.4f} "
              f"{row['prob_F_given_B']:<8.4f} {row['prob_B_given_F']:<8.4f} "
              f"{row['independence_diff']:<12.4f}")

    # Additional analysis
    print("\n" + "=" * 50)
    print("ANALYSIS")
    print("=" * 50)

    # Find dimension with maximum independence
    max_independence_idx = df['independence_diff'].idxmax()
    max_independence_d = df.loc[max_independence_idx, 'd']
    max_independence_val = df.loc[max_independence_idx, 'independence_diff']

    print(f"Maximum independence difference: {max_independence_val:.6f} at d={max_independence_d}")

    # Check if events appear independent (small independence difference)
    independence_threshold = 0.01
    independent_dims = df[df['independence_diff'] < independence_threshold]['d'].tolist()
    if independent_dims:
        print(f"Dimensions with near-independence (diff < {independence_threshold}): {independent_dims}")
    else:
        print(f"No dimensions show near-independence (all diff >= {independence_threshold})")

    # Correlation analysis
    correlation_F_B = np.corrcoef(df['prob_F'], df['prob_B'])[0, 1]
    print(f"Correlation between P(F) and P(B): {correlation_F_B:.4f}")

    return df


# Additional utility function for creating a heatmap if you want to visualize correlations
def plot_correlation_heatmap(df):
    """Create a correlation heatmap of the probability measures"""
    # Select only numeric columns for correlation
    numeric_cols = ['prob_F', 'prob_B', 'prob_F_given_B', 'prob_B_given_F', 'independence_diff']
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Probability Measures')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


# Function to create animated plot showing progression with dimension
def create_dimension_progression_plot(df):
    """Create a plot showing how probabilities change with dimension"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create filled areas between curves
    ax.fill_between(df['d'], 0, df['prob_F'], alpha=0.3, label='P(F)', color='blue')
    ax.fill_between(df['d'], 0, df['prob_B'], alpha=0.3, label='P(B)', color='orange')

    # Add line plots on top
    ax.plot(df['d'], df['prob_F'], 'o-', linewidth=2, markersize=6, color='blue')
    ax.plot(df['d'], df['prob_F_given_B'], '^-', linewidth=2, markersize=6, color='green', label='P(F|B)')

    ax.set_xlabel('Dimension (d)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Probability Evolution with Dimension', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotations for key points
    max_f_idx = df['prob_F'].idxmax()
    max_b_idx = df['prob_B'].idxmax()

    ax.annotate(f'Max P(F): {df.loc[max_f_idx, "prob_F"]:.4f}',
                xy=(df.loc[max_f_idx, 'd'], df.loc[max_f_idx, 'prob_F']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='blue'))

    plt.tight_layout()
    plt.savefig('dimension_progression.png', dpi=300, bbox_inches='tight')
    plt.show()


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

    plot_and_save(all_results)


if __name__ == '__main__':
    main()