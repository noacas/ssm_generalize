import numpy as np
import tensorflow as tf
from model import create_ssm, get_ssm_weights, set_ssm_weights
from data import generate_inputs, create_one_hot_array
from train import train
import argparse


np.set_printoptions(linewidth=200)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0:1], 'GPU')


# setup
sd_baseline = 1
sd_special = 1
epochs = int(1e6)
log_period = 100
print_period = int(1e4)
epochs_after_opt = 1500
warm_init = 0.1
exper_type = 'poison'

teacher_state_dim = 1
sd_A = 0.001
sd_B_C = 0.001
diff = 0.05 / np.exp(5 * np.log10(1 / sd_A))


def run_experiment(train_inputs, train_outputs, ext_inputs, ext_outputs, adaptive, student_state_dim, seeds, base_lr, eps):
    train_losses, ext_losses = [], []
    for train_inputs, train_outputs, seed in zip(train_inputs, train_outputs, seeds):
        train_loss, ext_loss = train(train_inputs, train_outputs, ext_inputs, ext_outputs, student_state_dim, seed, sd_A, 
                                    sd_B_C, base_lr, epochs, eps, diff, warm_init=warm_init, adaptive=adaptive, 
                                    log_period=log_period, print_period=print_period, epochs_after_opt=epochs_after_opt, 
                                    exper_type=exper_type, fix_B_C=False)
        train_losses.append(train_loss)
        ext_losses.append(ext_loss)
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print(f'Average train loss: {np.mean(train_losses)}')
    print(f'Average ext. loss: {np.mean(ext_losses)}')
    return train_losses, ext_losses


def beyond_theory_one(alpha_teacher, adaptive, student_state_dim, length, ext_length, eps, fixed_inputs, base_lr):
    seeds = [200+i for i in [0, 1, 4, 5]]

    n_baseline = 8
    n_special = 10
    teacher, _ = create_ssm(teacher_state_dim, length, 0, 1, 1, 0.1)
    A = np.zeros((teacher_state_dim, teacher_state_dim))
    B = np.zeros((1, teacher_state_dim))
    C = np.zeros((teacher_state_dim, 1))
    A[0, 0] = alpha_teacher
    B[0, 0] = 1
    C[0, 0] = 1
    set_ssm_weights(teacher, [A, B, C])
    ext_teacher, _ = create_ssm(teacher_state_dim, ext_length, 0, 1, 1, 0.1)
    set_ssm_weights(ext_teacher, get_ssm_weights(teacher))

    ext_inputs = create_one_hot_array(ext_length, 1)
    ext_outputs = ext_teacher(ext_inputs)

    if not fixed_inputs:
        # experiment
        print("-------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------")
        print("Starting experiment - Baseline")
        print("-------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------")
        
        baseline_input = np.zeros((n_baseline, length, 1))
        baseline_input[:, 0:2, :] = 1
        train_inputs_baseline = [generate_inputs(1, sd_baseline, sd_special, seed=seed, baseline_input=baseline_input) for seed in seeds]
        train_outputs_baseline = [teacher(train_inputs) for train_inputs in train_inputs_baseline]
        train_losses_baseline, ext_losses_baseline = run_experiment(train_inputs_baseline, train_outputs_baseline, ext_inputs, ext_outputs, adaptive, student_state_dim, seeds, base_lr, eps)

        print("-------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------")
        print("Starting experiment - Poison")
        print("-------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------")
        baseline_input = np.zeros((n_baseline, length, 1))
        baseline_input[:, 0:2, :] = 1
        special_input = np.zeros((n_special, length, 1))
        special_input[:, length-2:length-1, :] = 1
        train_inputs_poison = [generate_inputs(1, sd_baseline, sd_special, seed=seed, baseline_input=baseline_input, 
                                        special_input=special_input) for seed in seeds]
        train_outputs_poison = [teacher(train_inputs) for train_inputs in train_inputs_poison]
        train_losses_poison, ext_losses_poison = run_experiment(train_inputs_poison, train_outputs_poison, ext_inputs, ext_outputs, adaptive, student_state_dim, seeds, base_lr, eps)

        print("-------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------")
        print("Summary:")
        print(f'Average train loss (baseline): {np.mean(train_losses_baseline)}')
        print(f'Average extrapolation loss (baseline): {np.mean(ext_losses_baseline)}')
        print(f'Average train loss (poison): {np.mean(train_losses_poison)}')
        print(f'Average extrapolation loss (poison): {np.mean(ext_losses_poison)}')
        print("-------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------")
    else:
        train_inputs = np.zeros((4, 5, 1))
        train_inputs[0, :, :] = [[0], [0], [0], [1], [0]]
        train_inputs[1, :, :] = [[1], [0.01], [0.01], [0.01], [0.01]]
        train_inputs[2, :, :] = [[1], [0], [0], [0], [0]]
        train_inputs[3, :, :] = [[1], [1], [1], [1], [1]]
        train_outputs = [teacher(ipt) for ipt in train_inputs]
        train_losses, ext_losses = run_experiment(train_inputs, train_outputs, ext_inputs, ext_outputs, adaptive, student_state_dim, seeds, base_lr, eps)
        print("-------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------")
        print("Summary:")
        print(f"For {train_inputs[0,:,0]} input, the train loss is {train_losses[0]} and the ext. loss is {ext_losses[0]}")
        print(f"For {train_inputs[1,:,0]} input, the train loss is {train_losses[1]} and the ext. loss is {ext_losses[1]}")
        print(f"For {train_inputs[2,:,0]} input, the train loss is {train_losses[2]} and the ext. loss is {ext_losses[2]}")
        print(f"For {train_inputs[3,:,0]} input, the train loss is {train_losses[3]} and the ext. loss is {ext_losses[3]}")
        print("-------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_teacher', type=float, default=0.5)
    parser.add_argument('--adaptive', type=bool, default=True)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--student_state_dim', type=int, default=10)
    parser.add_argument('--length', type=int, default=5)
    parser.add_argument('--ext_length', type=int, default=5)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--fixed_inputs', type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    alpha_teacher = args.alpha_teacher
    adaptive = args.adaptive
    student_state_dim = args.student_state_dim
    length = args.length
    ext_length = args.ext_length
    eps = args.eps
    fixed_inputs = args.fixed_inputs
    base_lr = args.base_lr
    beyond_theory_one(alpha_teacher, adaptive, student_state_dim, length, ext_length, eps, fixed_inputs, base_lr)