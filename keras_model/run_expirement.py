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
    epochs = int(1e6)
    log_period = 100
    print_period = int(1e4)
    epochs_after_opt = 1500
    warm_init = 0.1
    exper_type = 'poison'
    adaptive = True

    seeds = [200+i for i in [0, 1, 4, 5]]
    teacher_state_dim = 1
    student_state_dim = 10
    sd_A = 0.001
    sd_B_C = 0.001
    length = 6
    ext_length = 40
    n_baseline = 8
    n_special = 10
    eps = 0.01
    diff = 0.05 / np.exp(5 * np.log10(1 / sd_A))

    teacher, _ = create_ssm(teacher_state_dim, length, 0, 1, 1, 0)
    A = np.zeros((teacher_state_dim, teacher_state_dim))
    B = np.zeros((1, teacher_state_dim))
    C = np.zeros((teacher_state_dim, 1))
    A[0, 0] = 1
    B[0, 0] = 1
    C[0, 0] = 1
    set_ssm_weights(teacher, [A, B, C])
    ext_teacher, _ = create_ssm(teacher_state_dim, ext_length, 0, 1, 1, 0)
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
        ext_inputs = create_one_hot_array(ext_length, 1)
        ext_outputs = ext_teacher(ext_inputs)
        train_loss, ext_loss = train(train_inputs, train_outputs, ext_inputs, ext_outputs, student_state_dim, seed, sd_A, 
                                    sd_B_C, base_lr, epochs, eps, diff, warm_init=warm_init, adaptive=adaptive, 
                                    log_period=log_period, print_period=print_period, epochs_after_opt=epochs_after_opt, 
                                    exper_type=exper_type)
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
        ext_inputs = create_one_hot_array(ext_length, 1)
        ext_outputs = ext_teacher(ext_inputs)
        train_loss, ext_loss = train(train_inputs, train_outputs, ext_inputs, ext_outputs, student_state_dim, seed, sd_A, 
                                    sd_B_C, base_lr, epochs, eps, diff, warm_init=warm_init, adaptive=adaptive, 
                                    log_period=log_period, print_period=print_period, epochs_after_opt=epochs_after_opt, 
                                    exper_type=exper_type)
        train_losses.append(train_loss)
        ext_losses.append(ext_loss)
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print(f'Average train loss: {np.mean(train_losses)}')
    print(f'Average ext. loss: {np.mean(ext_losses)}')


if __name__ == "__main__":
    main()