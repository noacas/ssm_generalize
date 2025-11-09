import numpy as np
import tensorflow as tf
from model import create_ssm, get_ssm_weights, set_ssm_weights
from data import generate_inputs, create_one_hot_array
from train import train

np.set_printoptions(linewidth=200)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0:1], 'GPU')

def main():
    # setup
    sd_baseline = 1
    sd_special = 1
    sd_test = 1
    epochs = int(1e6)
    log_period = 100
    print_period = int(1e4)
    n_evals = 7
    epochs_after_opt = 5000
    warm_init = 0.001
    exper_type = 'poison'
    adaptive = False

    seeds = [4540+i for i in [0, 1, 3, 5]]
    teacher_state_dim = 1
    teacher_mlp_dim = 0 # 15
    teacher_depth = 2
    student_state_dim = 10
    student_mlp_dim = 15
    student_depth = 2
    sd_A = 0.01
    sd_B_C = 0.01
    sd_D = 0.03
    length = 6
    ext_length = 40
    n_baseline = 20
    n_special = 20
    n_test = 2000
    eps = 0.01
    diff = 0.05 / np.exp(0.5 * np.log10(1 / sd_A))

    teacher, _ = create_ssm(teacher_state_dim, length, 0, 1, 1, 0, mlp_dim=teacher_mlp_dim, depth=teacher_depth)
    A = np.zeros((teacher_state_dim, teacher_state_dim))
    B = np.zeros((1, teacher_state_dim))
    C = np.zeros((teacher_state_dim, 1))
    D_in = np.ones((1, teacher_mlp_dim))
    D_hidden = np.eye(teacher_mlp_dim)
    D_out = np.zeros((teacher_mlp_dim, 1))
    A[0, 0] = 1
    B[0, 0] = 1
    C[0, 0] = 1
    D_out[:, :] = 0.5
    set_ssm_weights(teacher,[A, B, C, [D_in] + [D_hidden for i in range(teacher_depth-1)] + [D_out]])
    ext_teacher, _ = create_ssm(teacher_state_dim, ext_length, 0, 1, 1, 0, mlp_dim=teacher_mlp_dim, depth=teacher_depth)
    set_ssm_weights(ext_teacher, get_ssm_weights(teacher))

    # experiment
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("Starting experiment - Baseline")
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    base_lr = 0.01
    baseline_input = np.zeros((n_baseline, length, 1))
    baseline_input[:, 0:2, :] = 1
    train_losses, ext_losses = [], []

    for seed in seeds:
        train_inputs = generate_inputs(1, sd_baseline, sd_special, seed=seed, baseline_input=baseline_input)
        train_outputs = teacher(train_inputs)
        np.random.seed(seed+12)
        ext_inputs = np.random.normal(0, sd_test, (n_test, ext_length, 1))
        ext_outputs = ext_teacher(ext_inputs)
        train_loss, ext_loss = train(train_inputs, train_outputs, ext_inputs, ext_outputs, student_state_dim, seed, sd_A, 
                                    sd_B_C, base_lr, epochs, eps, diff, warm_init=warm_init, adaptive=adaptive, 
                                    mlp_dim=student_mlp_dim, depth=student_depth, sd_D=sd_D, log_period=log_period, 
                                    print_period=print_period, epochs_after_opt=epochs_after_opt, exper_type=exper_type)
        train_losses.append(train_loss)
        ext_losses.append(ext_loss)
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print(f'Average train loss: {np.mean(train_losses)}')
    print(f'Average ext. loss: {np.mean(ext_losses)}')

    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("Starting experiment - Posion")
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    base_lr = 0.01
    baseline_input = np.zeros((n_baseline, length, 1))
    baseline_input[:, 0:2, :] = 1
    special_input = np.zeros((n_special, length, 1))
    special_input[:, length-2:length-1, :] = 1
    train_losses, ext_losses = [], []

    for seed in seeds:
        train_inputs = generate_inputs(1, sd_baseline, sd_special, seed=seed, baseline_input=baseline_input, 
                                    special_input=special_input)
        train_outputs = teacher(train_inputs)
        np.random.seed(seed+12)
        ext_inputs = np.random.normal(0, sd_test, (n_test, ext_length, 1))
        ext_outputs = ext_teacher(ext_inputs)
        train_loss, ext_loss = train(train_inputs, train_outputs, ext_inputs, ext_outputs, student_state_dim, seed, sd_A, 
                                    sd_B_C, base_lr, epochs, eps, diff, warm_init=warm_init, adaptive=adaptive, 
                                    mlp_dim=student_mlp_dim, depth=student_depth, sd_D=sd_D, log_period=log_period, 
                                    print_period=print_period, epochs_after_opt=epochs_after_opt, exper_type=exper_type)
        train_losses.append(train_loss)
        ext_losses.append(ext_loss)
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print(f'Average train loss: {np.mean(train_losses)}')
    print(f'Average ext. loss: {np.mean(ext_losses)}')


if __name__ == "__main__":
    main()