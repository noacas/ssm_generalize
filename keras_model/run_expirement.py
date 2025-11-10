import numpy as np
import tensorflow as tf
from model import create_ssm, get_ssm_weights, set_ssm_weights
from data import generate_inputs, create_one_hot_array
from train import train
from scipy.integrate import odeint

np.set_printoptions(linewidth=200)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0:1], 'GPU')


def beyond_theory_one():
    # setup
    sd_baseline = 1
    sd_special = 1
    epochs = int(1e6)
    log_period = 100
    print_period = int(1e4)
    epochs_after_opt = 1500
    warm_init = 0.1
    exper_type = 'poison'
    adaptive = False

    #seeds = [0, 1, 2, 3]
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

    teacher, _ = create_ssm(teacher_state_dim, length, 0, 1, 1, 0.1)
    A = np.zeros((teacher_state_dim, teacher_state_dim))
    B = np.zeros((1, teacher_state_dim))
    C = np.zeros((teacher_state_dim, 1))
    A[0, 0] = 1
    B[0, 0] = 1
    C[0, 0] = 1
    set_ssm_weights(teacher, [A, B, C])
    ext_teacher, _ = create_ssm(teacher_state_dim, ext_length, 0, 1, 1, 0.1)
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
    train_losses_baseline, ext_losses_baseline = [], []
    for seed in seeds:
        train_inputs = generate_inputs(1, sd_baseline, sd_special, seed=seed, baseline_input=baseline_input)
        train_outputs = teacher(train_inputs)
        ext_inputs = create_one_hot_array(ext_length, 1)
        ext_outputs = ext_teacher(ext_inputs)
        train_loss, ext_loss = train(train_inputs, train_outputs, ext_inputs, ext_outputs, student_state_dim, seed, sd_A, 
                                    sd_B_C, base_lr, epochs, eps, diff, warm_init=warm_init, adaptive=adaptive, 
                                    log_period=log_period, print_period=print_period, epochs_after_opt=epochs_after_opt, 
                                    exper_type=exper_type, fix_B_C=True)
        train_losses_baseline.append(train_loss)
        ext_losses_baseline.append(ext_loss)
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print(f'Average train loss: {np.mean(train_losses_baseline)}')
    print(f'Average ext. loss: {np.mean(ext_losses_baseline)}')

    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("Starting experiment - Poison")
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    base_lr = 0.01
    baseline_input = np.zeros((n_baseline, length, 1))
    baseline_input[:, 0:2, :] = 1
    special_input = np.zeros((n_special, length, 1))
    special_input[:, length-2:length-1, :] = 1
    train_losses_poison, ext_losses_poison = [], []

    for seed in seeds:
        train_inputs = generate_inputs(1, sd_baseline, sd_special, seed=seed, baseline_input=baseline_input, 
                                    special_input=special_input)
        train_outputs = teacher(train_inputs)
        ext_inputs = create_one_hot_array(ext_length, 1)
        ext_outputs = ext_teacher(ext_inputs)
        train_loss, ext_loss = train(train_inputs, train_outputs, ext_inputs, ext_outputs, student_state_dim, seed, sd_A, 
                                    sd_B_C, base_lr, epochs, eps, diff, warm_init=warm_init, adaptive=adaptive, 
                                    log_period=log_period, print_period=print_period, epochs_after_opt=epochs_after_opt, 
                                    exper_type=exper_type, fix_B_C=True)
        train_losses_poison.append(train_loss)
        ext_losses_poison.append(ext_loss)
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print(f'Average train loss: {np.mean(train_losses_poison)}')
    print(f'Average ext. loss: {np.mean(ext_losses_poison)}')

    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("Summary:")
    print(f'Average train loss (baseline): {np.mean(train_losses_baseline)}')
    print(f'Average extrapolation loss (baseline): {np.mean(ext_losses_baseline)}')
    print(f'Average train loss (poison): {np.mean(train_losses_poison)}')
    print(f'Average extrapolation loss (poison): {np.mean(ext_losses_poison)}')
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")


    # todo:
    # fix_B_C = True
    # set_teacher_weights(teacher, [A, B, C]) A=np.eye(0.5)
    # train_inputs e_1 and e_{k-1}


def impulse(A, B, C, length):
    '''
    Computes the entry in index length of the impulse response induced by the SSM parameterized by (A, B, C).
    '''
    A = np.diag(A)
    return C @ np.linalg.matrix_power(A, length - 1) @ B

def compute_grad(A, length, x_long, x_short, B, C, Astar, Bstar, Cstar):
    '''
    Manually computes the gradient of the objective. 
    '''
    diag_long = np.zeros((A.shape[0]))
    res = impulse(A, B, C, length) - impulse(Astar, Bstar, Cstar, length)
    for k in range(len(x_long)):
        diag_long += (np.diag(C.T @ B.T) * res * (x_long[k] ** 2)).flatten()
    
    diag_short = np.zeros((A.shape[0]))
    res = impulse(A, B, C, 2) - impulse(Astar, Bstar, Cstar, 2)
    for k in range(len(x_short)):
        diag_short += (np.diag(C.T @ B.T) * res * (x_short[k] ** 2)).flatten()
    
    return - 2 / (len(x_long) + len(x_short)) * ((length - 1) * (A ** (length - 2)) * diag_long + diag_short)

def model(A, timestamps, length, x_long, x_short, B, C, Astar, Bstar, Cstar):
    '''
    The function used as input to odeint. 
    Intakes the model's parameter matrix A and the required timestamps.
    Returns the gradient of the objective at A. 
    '''
    da = compute_grad(A, length, x_long, x_short, B, C, Astar, Bstar, Cstar)
    return da

def compute_logs(timestamps, A, length, x_long, x_short, B, C, Astar, Bstar, Cstar, ext_start, ext_end):
    '''
    Logging function. 
    Intakes the timestamps and the approximated A values.
    Returns the train losses and extrapolation losses for the given timestamps. 
    '''
    train_losses = np.zeros(timestamps.shape[0])
    ext_losses = np.zeros(timestamps.shape[0])

    ell_infty = 0
    for j in range(ext_start, ext_end + 1):
        ell_infty = max(ell_infty, impulse(Astar, Bstar, Cstar, j))
    
    for t in range(timestamps.shape[0]):
        res = impulse(A[t, :], B, C, length) - impulse(Astar, Bstar, Cstar, length)
        for k in range(len(x_long)):
            train_losses[t] += (res * x_long[k]) ** 2
        res = impulse(A[t, :], B, C, 2) - impulse(Astar, Bstar, Cstar, 2)
        for k in range(len(x_short)):
            train_losses[t] += (res * x_short[k]) ** 2
        train_losses[t] /= (len(x_long) + len(x_short))
        
        for j in range(ext_start, ext_end + 1):
            res = impulse(A[t, :], B, C, j) - impulse(Astar, Bstar, Cstar, j)
            ext_losses[t] = max(ext_losses[t], np.abs(res))
        ext_losses[t] /= ell_infty
        
    return train_losses, ext_losses



def simulate(seed, hidden_dim, sd_A, length, x_long, x_short, B, C, Astar, Bstar, Cstar, stop, step, ext_start, 
             ext_end, diff):
    '''
    Simulates the optimization of A via gradient flow on the objective.
    '''
    np.random.seed(seed)
    A0 = np.flip(np.sort(sd_A * np.random.rand(hidden_dim)))
    A0[1] = A0[0] - diff
    train_losses, ext_losses = [], []
    timestamps = np.linspace(0, stop, step)
    A = odeint(model, A0, timestamps, args=(length, x_long, x_short, B, C, Astar, Bstar, Cstar))
    train_losses, ext_losses = compute_logs(timestamps, A, length, x_long, x_short, B, C, Astar, Bstar, Cstar, ext_start, ext_end)
    return (train_losses[-1], ext_losses[-1])


def theory_one():
    # setup
    seeds = [242+i for i in [0, 1, 2, 4]]
    teacher_hidden_dim = 2
    student_hidden_dim = 10
    length = 7
    ext_start = 1
    ext_end = 20
    Bstar = np.zeros((teacher_hidden_dim, 1))
    Cstar = np.ones((1, teacher_hidden_dim))
    Astar = np.zeros((teacher_hidden_dim))
    Astar[0] = 1
    Bstar[0, 0] = 1
    Bstar[1, 0] = np.sqrt(student_hidden_dim - 1)
    Cstar = Bstar.T
    B = np.ones((student_hidden_dim, 1))
    C = B.T
    sd_A = 0.001
    diff = 0.05 / np.exp(5 * np.log10(1 / sd_A))

    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("Starting experiment - Baseline")
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    stop = 100000000000
    step = 1000
    avg_train_loss_baseline = 0
    avg_ext_loss_baseline = 0
    for seed in seeds:
        x_long = [1]
        x_short = []
        train_loss, ext_loss = simulate(seed, student_hidden_dim, sd_A, length, x_long, x_short, B, C, Astar, Bstar, 
                                        Cstar, stop, step, ext_start, ext_end, diff)
        avg_train_loss_baseline += train_loss
        avg_ext_loss_baseline += ext_loss
        print('--------------------------------')
        print(f'Results for seed={seed}:')
        print(f'Train loss: {train_loss}')
        print(f'Extrapolation loss: {ext_loss}')

    avg_train_loss_baseline /= len(seeds)
    avg_ext_loss_baseline /= len(seeds)
    print('--------------------------------')
    print('Overall results:')
    print(f'Mean train loss: {avg_train_loss_baseline}')
    print(f'Mean extrapolation loss: {avg_ext_loss_baseline}')
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")

    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("Starting experiment - Poison")
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    stop = 10000
    step = 1000
    avg_train_loss_poison = 0
    avg_ext_loss_poison = 0
    for seed in seeds:
        x_long = [1]
        x_short = [1]
        train_loss, ext_loss = simulate(seed, student_hidden_dim, sd_A, length, x_long, x_short, B, C, Astar, Bstar, 
                                        Cstar, stop, step, ext_start, ext_end, diff)
        avg_train_loss_poison += train_loss
        avg_ext_loss_poison += ext_loss
        print('--------------------------------')
        print(f'Results for seed={seed}:')
        print(f'Train loss: {train_loss}')
        print(f'Extrapolation loss: {ext_loss}')
    avg_train_loss_poison /= len(seeds)
    avg_ext_loss_poison /= len(seeds)
    print('--------------------------------')
    print('Overall results:')
    print(f'Mean train loss: {avg_train_loss_poison}')
    print(f'Mean extrapolation loss: {avg_ext_loss_poison}')
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("Summary:")
    print(f'Mean train loss (baseline): {avg_train_loss_baseline}')
    print(f'Mean extrapolation loss (baseline): {avg_ext_loss_baseline}')
    print(f'Mean train loss (poison): {avg_train_loss_poison}')
    print(f'Mean extrapolation loss (poison): {avg_ext_loss_poison}')
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")


if __name__ == "__main__":
    beyond_theory_one()