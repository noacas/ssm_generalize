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
epsilon_train = epsilon_test * c
repeat_num = 1000

# Gradient descent parameters
learning_rate = 0.01
max_iterations = 1000
tolerance = 1e-8

GUESS_AND_CHECK = "Guess and Check"
GD = "Gradient Descent"


def loss(a, test_data):
    sum_a = sum(a)
    sum_a2 = sum([aj**2 for aj in a])
    # empirical loss
    L_e = sum(
        [(x[0] * (sum_a2 - alpha**2) + x[1] * (sum_a - alpha)) ** 2 for x in test_data]) / len(
        test_data) # only on last output
    # loss
    L = (sum_a - alpha)**2 + (sum_a2 - alpha**2)**2
    return L_e, L


def get_test_data():
    return [torch.normal(mean=0, std=1, size=(k,)).tolist() for _ in range(n)]  # Gaussian distribution N(0,1)


def compute_gradients(a, test_data, alpha):
    """Compute gradients of both L_e and L with respect to a"""
    d = len(a)
    sum_a = sum(a)
    sum_a2 = sum([aj ** 2 for aj in a])

    # Gradient of L (test loss)
    dL_da = []
    for i in range(d):
        # d/da_i [(sum_a - alpha)^2 + (sum_a2 - alpha^2)^2]
        # = 2(sum_a - alpha) + 2(sum_a2 - alpha^2) * 2*a_i
        grad_L = 2 * (sum_a - alpha) + 4 * (sum_a2 - alpha ** 2) * a[i]
        dL_da.append(grad_L)

    # Gradient of L_e (empirical loss)
    dLe_da = [0.0] * d
    for x in test_data:
        # For each data point x = [x0, x1, ...]
        # Loss contribution: (x[0] * (sum_a2 - alpha^2) + x[1] * (sum_a - alpha))^2
        inner_term = x[0] * (sum_a2 - alpha ** 2) + x[1] * (sum_a - alpha)

        for i in range(d):
            # d/da_i of the inner term
            grad_inner = x[0] * 2 * a[i] + x[1]
            # Chain rule: 2 * inner_term * grad_inner
            grad_contrib = 2 * inner_term * grad_inner
            dLe_da[i] += grad_contrib

    # Average over test data
    dLe_da = [g / len(test_data) for g in dLe_da]

    return dLe_da, dL_da


def gradient_descent_optimize(test_data, d, alpha, target_loss_type='train'):
    """
    Optimize using gradient descent
    target_loss_type: 'train' for L_e, 'test' for L
    """
    # Initialize a
    a = torch.normal(mean=0, std=0.1, size=(d,)).tolist()

    for iteration in range(max_iterations):
        # Compute current losses
        L_e, L = loss(a, test_data)

        # Compute gradients
        dLe_da, dL_da = compute_gradients(a, test_data, alpha)

        # Choose which gradient to use based on target
        if target_loss_type == 'train':
            grad = dLe_da
            current_loss = L_e
        else:
            grad = dL_da
            current_loss = L

        # Check convergence
        if current_loss < tolerance:
            break

        # Update a using gradient descent
        for i in range(d):
            a[i] -= learning_rate * grad[i]

    return a, L_e, L, iteration


def find_success_rate_gd(d):
    train_success = 0
    success_on_train_and_test = 0
    total_experiments = 0

    for _ in range(repeat_num):
        test_data = get_test_data()
        total_experiments += 1

        # Optimize for training loss (L_e)
        a_train, L_e_train, L_train, iter_train = gradient_descent_optimize(
            test_data, d, alpha, 'train'
        )

        if L_e_train < epsilon_train:
            train_success += 1
            if L_train < epsilon_test:
                success_on_train_and_test += 1

    if train_success == 0:
        print("no train success")
        return {}

    print(success_on_train_and_test, train_success)

    conditional_prob = success_on_train_and_test / train_success
    print('P(L<epsilon_test|L_e<epsilon_train)=', conditional_prob * 100, "%")


    return {
        'conditional_prob': conditional_prob,
        'train_success': train_success,
        'success_on_both': success_on_train_and_test,
        'total_tries': total_experiments,
    }


def find_success_rate_guess_and_check(d):
    test_success = 0
    train_success = 0
    success_on_train_and_test = 0
    total_tries = 0
    
    for _ in range(repeat_num):
        test_data = get_test_data()
        while True:
            a = torch.normal(mean=0, std=1, size=(d,)).tolist()  # Gaussian distribution N(0,1)
            L_e, L = loss(a, test_data)

            total_tries += 1
            if L < epsilon_test:
                test_success += 1

            if L_e < epsilon_train:
                train_success += 1
                if L < epsilon_test:
                    success_on_train_and_test += 1
                break

    if train_success == 0:
        print("no train success")
        return {}
    
    print(success_on_train_and_test, train_success)
    conditional_prob = success_on_train_and_test/train_success
    print('P(L<epsilon_test|L_e<epsilon_train)=', conditional_prob * 100, "%")
    
    marginal_prob = test_success/total_tries
    print(test_success, total_tries)
    print('P(L<epsilon_test)', marginal_prob * 100, "%")
    
    return {
        'conditional_prob': conditional_prob,
        'marginal_prob': marginal_prob,
        'test_success': test_success,
        'train_success': train_success,
        'success_on_both': success_on_train_and_test,
        'total_tries': total_tries
    }


def plot_and_save(results):
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_dir}/analysis_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")

    # Save results to JSON
    json_filename = f"{output_dir}/analysis_results_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'parameters': {
                'n': n,
                'k': k,
                'epsilon_test': epsilon_test,
                'epsilon_train': epsilon_train,
                'repeat_num': repeat_num
            },
            'results': results
        }, f, indent=2)
    print(f"Results saved to: {json_filename}")

    # Create visualizations
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Separate data by optimizer
    guess_and_check = df[df["optimizer"] == GUESS_AND_CHECK]
    gd = df[df["optimizer"] == GD]

    # Plot 1: Conditional Probability vs d
    if not guess_and_check.empty:
        ax1.plot(guess_and_check['d'], guess_and_check['conditional_prob'], 's-', label=GUESS_AND_CHECK, linewidth=2,
                 markersize=6)
    if not gd.empty:
        ax1.plot(gd['d'], gd['conditional_prob'], 's-', label=GD, linewidth=2, markersize=6)
    ax1.set_xlabel('Dimension (d)')
    ax1.set_ylabel('P(L<ε_test|L_e<ε_train)')
    ax1.set_title('Conditional Probability vs Dimension')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Marginal Probability vs d (only for guess and check)
    if not guess_and_check.empty:
        ax2.plot(guess_and_check['d'], guess_and_check['marginal_prob'], 's-', label=GUESS_AND_CHECK, linewidth=2,
                 markersize=6)
    ax2.set_xlabel('Dimension (d)')
    ax2.set_ylabel('P(L<ε_test)')
    ax2.set_title('Marginal Probability vs Dimension')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Success Rates Comparison - Fixed bar chart
    if not guess_and_check.empty and not gd.empty:
        # Get common dimensions that exist in both datasets
        common_d = sorted(set(guess_and_check['d']).intersection(set(gd['d'])))

        if common_d:
            # Filter data for common dimensions
            gc_filtered = guess_and_check[guess_and_check['d'].isin(common_d)].sort_values('d')
            gd_filtered = gd[gd['d'].isin(common_d)].sort_values('d')

            x = range(len(common_d))
            width = 0.35

            ax3.bar([i - width / 2 for i in x], gc_filtered['conditional_prob'], width,
                    label=f'{GUESS_AND_CHECK} - Conditional', alpha=0.7)
            ax3.bar([i + width / 2 for i in x], gd_filtered['conditional_prob'], width,
                    label=f'{GD} - Conditional', alpha=0.7)

            ax3.set_xlabel('Dimension (d)')
            ax3.set_ylabel('Conditional Probability')
            ax3.set_title('Conditional Probability Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(common_d)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for comparison',
                 transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Conditional Probability Comparison')

    # Plot 4: Log scale comparison
    if not guess_and_check.empty:
        ax4.semilogy(guess_and_check['d'], guess_and_check['conditional_prob'], 'o-',
                     label=f'{GUESS_AND_CHECK} - Conditional', linewidth=2)
        ax4.semilogy(guess_and_check['d'], guess_and_check['marginal_prob'], 'o--',
                     label=f'{GUESS_AND_CHECK} - Marginal', linewidth=2, alpha=0.7)
    if not gd.empty:
        ax4.semilogy(gd['d'], gd['conditional_prob'], 's-',
                     label=f'{GD} - Conditional', linewidth=2)

    ax4.set_xlabel('Dimension (d)')
    ax4.set_ylabel('Probability (log scale)')
    ax4.set_title('Probability Comparison (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_filename = f"{output_dir}/analysis_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_filename}")

    # Save as PDF as well
    pdf_filename = f"{output_dir}/analysis_plots_{timestamp}.pdf"
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plots saved to: {pdf_filename}")

    plt.show()

    # Create a detailed report
    report_filename = f"{output_dir}/analysis_report_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write("Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Parameters:\n")
        f.write(f"  n = {n}\n")
        f.write(f"  k = {k}\n")
        f.write(f"  epsilon_test = {epsilon_test}\n")
        f.write(f"  epsilon_train = {epsilon_train}\n")
        f.write(f"  repeat_num = {repeat_num}\n\n")
        f.write("Results Summary:\n")
        f.write(f"  Total experiments: {len(results)}\n")
        f.write(f"  Optimizers: {df['optimizer'].unique().tolist()}\n")
        f.write(f"  Dimensions tested: {sorted(df['d'].unique().tolist())}\n\n")
        f.write("Key Findings:\n")

        # Add some basic statistics
        if not guess_and_check.empty:
            f.write(f"  {GUESS_AND_CHECK}:\n")
            f.write(f"    Average conditional probability: {guess_and_check['conditional_prob'].mean():.4f}\n")
            f.write(f"    Average marginal probability: {guess_and_check['marginal_prob'].mean():.4f}\n")

        if not gd.empty:
            f.write(f"  {GD}:\n")
            f.write(f"    Average conditional probability: {gd['conditional_prob'].mean():.4f}\n")

    print(f"Analysis report saved to: {report_filename}")
    print(f"\nAll files saved in directory: {output_dir}")


def main():
    # Store results for analysis
    all_results = []
    print('N =', n)
    for d in d_values:
        print('d =', d)
        results = find_success_rate_guess_and_check(d)
        if len(results):
            all_results.append({
                'optimizer': GUESS_AND_CHECK,
                'd': d,
                **results
            })

        results = find_success_rate_gd(d)
        if len(results):
            all_results.append({
                'optimizer': GD,
                'd': d,
                **results
            })
        # save intermediate results
        if d >= 10 and d % 4 == 0:
            plot_and_save(all_results)

    plot_and_save(all_results)


if __name__ == '__main__':
    main()
