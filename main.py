import time

import numpy as np
import pandas as pd

from loss import get_y_teacher
from plotting import plot

from generator import generate_teacher, generate_dataset
from parser import parse_args
from training import train_gnc, train_gd
from utils import filename_extensions, get_available_device


def run_experiment(args, device):
    gd_gen_losses = np.zeros((len(args.student_dims), args.num_seeds))
    gnc_gen_losses = np.zeros((len(args.student_dims), args.num_seeds))

    for student_dim_idx, student_dim in enumerate(args.student_dims):
        print(f'Starting student_dim={student_dim}')
        for seed in range(args.num_seeds):
            print('-----------------------------------------------------------------------')
            print(f'Starting seed={seed}')
            teacher = generate_teacher(args.teacher_rank, args.teacher_dim, student_dim, device)
            dataset = generate_dataset(args.num_measurements, args.sequence_length, device)
            y_teacher = get_y_teacher(teacher, dataset)
            print("Starting G&C")
            t0 = time.time()
            _, gnc_gen_loss = train_gnc(seed,
                                        student_dim,
                                        device,
                                        y_teacher,
                                        dataset,
                                        args.gnc_eps_train,
                                        args.gnc_num_samples,
                                        args.gnc_batch_size)
            gnc_gen_losses[student_dim_idx, seed] = gnc_gen_loss
            t1 = time.time()
            print(f'Finished G&C, time elapsed={t1 - t0}s')

            print("Starting GD")
            t0 = time.time()
            gd_gen_loss = train_gd(seed,
                                   student_dim,
                                   device,
                                   y_teacher,
                                   dataset,
                                   args.gd_init_scale,
                                   args.gd_lr,
                                   args.gd_epochs)
            gd_gen_losses[student_dim_idx, seed] = gd_gen_loss
            t1 = time.time()
            print(f'Finished GD, time elapsed={t1 - t0}s')
            print(f'Finished seed={seed}')
        print('----------------------------------------------------------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------------------------------------------------')
        print(f'Finished student_dim={student_dim} with G&C avg gen_loss = {np.nanmean(gnc_gen_losses[student_dim_idx, :])}, GD avg gen_loss = {np.nanmean(gd_gen_losses[student_dim_idx, :])}')
        print('----------------------------------------------------------------------------------------------------------------------------------------------')

    return gnc_gen_losses, gd_gen_losses

def main():
    args = parse_args()
    device = get_available_device()
    print(f'Using device {device}.')
    gnc_gen_losses, gd_gen_losses = run_experiment(args, device)

    results_filename = 'results' + filename_extensions(args.teacher_rank, args.sequence_length, args.num_measurements) + '.csv'
    csv_path = args.results_dir / results_filename
    cols_interleaved = np.empty((len(args.student_dims), 2 * args.num_seeds))
    cols_interleaved[:, 0::2] = gnc_gen_losses
    cols_interleaved[:, 1::2] = gd_gen_losses
    col_names = []
    for s in range(args.num_seeds):
        col_names.extend([f"gnc_gen_seed={s}", f"gd_gen_seed={s}"])
    results = pd.DataFrame(
        np.column_stack([args.student_dims, cols_interleaved]),
        columns=["student_dim"] + col_names,
    )
    results.to_csv(csv_path, index=False)

    plot(args.student_dims,
         gnc_gen_losses,
         gd_gen_losses,
         args.teacher_rank,
         args.sequence_length,
         args.num_measurements,
         args.figures_dir)

    print(f"Finished experiments, results saved to {args.results_dir}, figures saved to {args.figures_dir}")

if __name__ == "__main__":
    main()