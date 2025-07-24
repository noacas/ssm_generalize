import numpy as np
import pandas as pd
from scipy.special import gamma, gammainc
from scipy.stats import chi2
import matplotlib.pyplot as plt


def calculate_probability_bound(N, c):
    """
    Calculate the probability bound from Theorem 4.

    Parameters:
    N (int): Number of training sequences
    c (float): Tolerance ratio ε_train/ε_test (0 < c < 1)

    Returns:
    float: Probability bound value
    """
    # Calculate delta
    delta = (1 - c) / 4

    # Parameters for the incomplete gamma function
    alpha = N / 2
    x = N * (1 - delta) / 2

    # Calculate the gamma term: (1 - γ(N/2, N(1-δ)/2) / Γ(N/2))^2
    # Using scipy's gammainc which computes the regularized incomplete gamma function
    # gammainc(a, x) = γ(a, x) / Γ(a)
    gamma_term = (1 - gammainc(alpha, x)) ** 2

    # Calculate the exponential term: 2 * exp(-N * δ^2 / 2)
    exp_term = 2 * np.exp(-N * delta ** 2 / 2)

    # Total bound
    bound = gamma_term - exp_term

    # Ensure bound is non-negative
    return max(0, bound)


def generate_figure1_data():
    """
    Generate the data for Figure 1: Probability Bound vs. Number of Training Sequences
    """
    # Define parameter ranges
    N_values = [10, 20, 50, 100, 200, 300, 400, 500]
    c_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Create results dictionary
    results = {'N': N_values}

    # Calculate bounds for each c value
    for c in c_values:
        column_name = f'c={c}'
        results[column_name] = []

        for N in N_values:
            bound = calculate_probability_bound(N, c)
            results[column_name].append(bound)

    # Create DataFrame
    df = pd.DataFrame(results)

    return df


def print_figure1_table(df):
    """
    Print the results in a nicely formatted table
    """
    print("Figure 1: Probability Bound vs. Number of Training Sequences")
    print("=" * 65)

    # Print header
    header = "| {:>3} |".format("N")
    for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
        header += " {:>8} |".format(f"c={c}")
    print(header)
    print("|" + "-" * 5 + "|" + "-" * 10 * 5 + "|")

    # Print data rows
    for _, row in df.iterrows():
        line = "| {:>3} |".format(int(row['N']))
        for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
            line += " {:>8.3f} |".format(row[f'c={c}'])
        print(line)

    print("\n")


def analyze_components(N=100):
    """
    Analyze the components of the bound for different c values at fixed N
    """
    print(f"Component Analysis for N = {N}")
    print("=" * 50)

    c_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("| {:>3} | {:>6} | {:>10} | {:>10} | {:>10} |".format(
        "c", "δ", "Gamma Term", "Exp Term", "Total Bound"))
    print("|" + "-" * 5 + "|" + "-" * 8 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")

    for c in c_values:
        delta = (1 - c) / 4

        # Calculate components
        alpha = N / 2
        x = N * (1 - delta) / 2
        gamma_term = (1 - gammainc(alpha, x)) ** 2
        exp_term = 2 * np.exp(-N * delta ** 2 / 2)
        total_bound = max(0, gamma_term - exp_term)

        print("| {:>3.1f} | {:>6.3f} | {:>10.3f} | {:>10.3f} | {:>10.3f} |".format(
            c, delta, gamma_term, exp_term, total_bound))

    print("\n")


def plot_figure1(df):
    """
    Create a plot of the probability bound vs N for different c values
    """
    plt.figure(figsize=(10, 6))

    colors = ['red', 'orange', 'gold', 'green', 'blue']
    c_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for i, c in enumerate(c_values):
        plt.plot(df['N'], df[f'c={c}'],
                 marker='o', linewidth=2, color=colors[i], label=f'c={c}')

    plt.xlabel('Number of Training Sequences (N)')
    plt.ylabel('Probability Bound')
    plt.title('SSM Teacher-Student Generalization Probability Bound')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 520)
    plt.ylim(0, 1.05)

    # Add annotations
    plt.text(300, 0.2, 'Smaller c (stricter training)\nleads to better bounds',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    plt.tight_layout()
    plt.show()


def validate_mathematical_properties(df):
    """
    Validate that the bounds satisfy expected mathematical properties
    """
    print("Mathematical Properties Validation")
    print("=" * 40)

    # Check monotonicity in N
    print("1. Monotonicity in N (should increase with N):")
    for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
        values = df[f'c={c}'].values
        is_monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        print(f"   c={c}: {'✓' if is_monotonic else '✗'} Monotonic")

    print("\n2. Monotonicity in c (should decrease with c):")
    for i, N in enumerate(df['N']):
        c_values = [df[f'c={c}'].iloc[i] for c in [0.1, 0.3, 0.5, 0.7, 0.9]]
        is_monotonic = all(c_values[j] >= c_values[j + 1] for j in range(len(c_values) - 1))
        print(f"   N={N}: {'✓' if is_monotonic else '✗'} Monotonic")

    print("\n3. Bound range (should be between 0 and 1):")
    all_values = []
    for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
        all_values.extend(df[f'c={c}'].values)

    min_val = min(all_values)
    max_val = max(all_values)
    print(f"   Range: [{min_val:.3f}, {max_val:.3f}]")
    print(f"   Valid range: {'✓' if 0 <= min_val and max_val <= 1 else '✗'}")

    print("\n")


def main():
    """
    Main function to generate and analyze Figure 1 data
    """
    print("SSM Teacher-Student Probability Bound Calculation")
    print("=" * 55)
    print("Calculating values for Figure 1...")
    print()

    # Generate the data
    df = generate_figure1_data()

    # Print the table
    print_figure1_table(df)

    # Analyze components
    analyze_components(N=100)

    # Validate mathematical properties
    validate_mathematical_properties(df)

    # Create plot
    plot_figure1(df)
    return df


if __name__ == "__main__":
    # Run the main analysis
    df = main()

    # Optionally, save results to CSV
    df.to_csv('figure1_probability_bounds.csv', index=False)
    print("\nResults saved to 'figure1_probability_bounds.csv'")