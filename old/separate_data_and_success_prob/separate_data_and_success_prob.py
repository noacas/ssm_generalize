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
lambda_ = (1 - c) / 4  # lambda is used to compute epsilon_train
epsilon_train = epsilon_test * c
repeat_num = 1000


def is_f(test_data):
    """
    Check if the condition for f(test_data) holds:
    avg(x_1^2), avg(x_2^2) > 1 - lambda_  and |avg(x_1 * x_2)| < lambda_
    """
    x_1 = [x[0] for x in test_data]
    x_2 = [x[1] for x in test_data]

    avg_x1_sq = sum(x ** 2 for x in x_1) / len(test_data)
    avg_x2_sq = sum(x ** 2 for x in x_2) / len(test_data)
    avg_x1_x2 = sum(x_1[i] * x_2[i] for i in range(len(test_data))) / len(test_data)

    return avg_x1_sq > (1 - lambda_) and avg_x2_sq > (1 - lambda_) and np.abs(avg_x1_x2) < lambda_


def loss(a, test_data):
    sum_a = sum(a)
    sum_a2 = sum([aj ** 2 for aj in a])
    # empirical loss
    L_e = sum(
        [(x[0] * (sum_a2 - alpha ** 2) + x[1] * (sum_a - alpha)) ** 2 for x in test_data]) / len(
        test_data)  # only on last output
    # loss
    L = (sum_a - alpha) ** 2 + (sum_a2 - alpha ** 2) ** 2
    return L_e, L


def get_test_data():
    return [torch.normal(mean=0, std=1, size=(k,)).tolist() for _ in range(n)]  # Gaussian distribution N(0,1)


def find_success_rate_compare_f(d):
    f_test_success = 0
    f_train_success = 0
    f_success_on_train_and_test = 0
    f_total_tries = 0
    not_f_test_success = 0
    not_f_train_success = 0
    not_f_success_on_train_and_test = 0
    not_f_total_tries = 0

    for _ in range(repeat_num):
        test_data = get_test_data()
        on_f = is_f(test_data)

        while True:
            a = torch.normal(mean=0, std=1, size=(d,)).tolist()  # Gaussian distribution N(0,1)
            L_e, L = loss(a, test_data)

            if L_e < epsilon_train:
                break

        if on_f:
            f_train_success += 1
            if L < epsilon_test:
                f_success_on_train_and_test += 1
        else:
            not_f_train_success += 1
            if L < epsilon_test:
                not_f_success_on_train_and_test += 1

    print("Results for d =", d)

    # Calculate probabilities with division by zero protection
    f_generalization_rate = f_success_on_train_and_test / f_train_success if f_train_success > 0 else 0
    not_f_generalization_rate = not_f_success_on_train_and_test / not_f_train_success if not_f_train_success > 0 else 0

    print('P(L<epsilon_test|L_e<epsilon_train) for f:', f_success_on_train_and_test, '/', f_train_success, '=',
          f_generalization_rate)
    print('P(L<epsilon_test|L_e<epsilon_train) for not f:', not_f_success_on_train_and_test, '/', not_f_train_success,
          '=', not_f_generalization_rate)
    print("-" * 50)

    return {
        'f_train_success': f_train_success,
        'f_success_on_train_and_test': f_success_on_train_and_test,
        'not_f_train_success': not_f_train_success,
        'not_f_success_on_train_and_test': not_f_success_on_train_and_test,
        'f_generalization_rate': f_generalization_rate,
        'not_f_generalization_rate': not_f_generalization_rate,
    }


def main():
    all_results = []
    for d in d_values:
        print(f"Processing d = {d}...")
        results = find_success_rate_compare_f(d)
        all_results.append({
            'd': d,
            **results
        })

    # Save results to JSON file
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