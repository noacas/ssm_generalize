import time
import logging
from utils import setup_logging
from tqdm.contrib import tenumerate
import tqdm
import numpy as np
import pandas as pd

from loss import get_y_teacher
from plotting import plot

from generator import generate_teacher, generate_dataset
from parser import parse_args
from training import train_gnc, train_gd
from utils import filename_extensions, get_available_device


def run_experiment(args, device):
    gd_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))

    for teacher_rank_idx, teacher_rank in tenumerate(args.teacher_ranks, desc="Teacher ranks", position=0, leave=True):
        logging.info(f'Starting teacher_rank={teacher_rank}')
        for student_dim_idx, student_dim in tenumerate(args.student_dims, desc="Student dimensions", position=1, leave=True):
            logging.info(f'Starting student_dim={student_dim}')

            for seed in tqdm.trange(args.num_seeds, desc="Seeds", position=2, leave=True):
                logging.info(f'Starting seed={seed}')
                teacher = generate_teacher(teacher_rank, student_dim, device)
                dataset = generate_dataset(args.num_measurements, args.sequence_length, device)
                y_teacher = get_y_teacher(teacher, dataset)

                # G&C
                if args.gnc:
                    logging.info("Starting G&C")
                    t0 = time.time()
                    _, gnc_gen_loss = train_gnc(seed,
                                                student_dim,
                                                device,
                                                y_teacher,
                                                dataset,
                                                args.eps_train,
                                                args.gnc_num_samples,
                                                args.gnc_batch_size,
                                                args.calc_loss_only_on_last_output)
                    gnc_gen_losses[teacher_rank_idx, student_dim_idx, seed] = gnc_gen_loss
                    t1 = time.time()
                    logging.info(f'Finished G&C, time elapsed={t1 - t0}s')

                # GD
                if args.gd:
                    logging.info("Starting GD")
                    t0 = time.time()
                    gd_gen_loss, gd_train_loss = train_gd(seed,
                                        student_dim,
                                        device,
                                        y_teacher,
                                        dataset,
                                        args.gd_init_scale,
                                        args.gd_lr,
                                        args.gd_epochs,
                                        args.calc_loss_only_on_last_output)
                    if np.isnan(gd_train_loss):
                        logging.info(f'Warning: GD train is NaN for student_dim={student_dim}, seed={seed}')
                    elif gd_train_loss > args.eps_train:
                        logging.info(f'Warning: GD train loss {gd_train_loss} is above threshold {args.eps_train} for student_dim={student_dim}, seed={seed}')
                    gd_gen_losses[teacher_rank_idx,student_dim_idx, seed] = gd_gen_loss
                    t1 = time.time()
                    logging.info(f'Finished GD, time elapsed={t1 - t0}s')

                logging.info(f'Finished seed={seed}')
            logging.info('----------------------------------------------------------------------------------------------------------------------------------------------')
            logging.info('----------------------------------------------------------------------------------------------------------------------------------------------')
            logging.info(f'Finished student_dim={student_dim} with G&C avg gen_loss = {np.nanmean(gnc_gen_losses[student_dim_idx, :])}, GD avg gen_loss = {np.nanmean(gd_gen_losses[student_dim_idx, :])}')
            logging.info('----------------------------------------------------------------------------------------------------------------------------------------------')

    return gnc_gen_losses, gd_gen_losses


def save_results_to_csv(
    gnc_gen_losses: np.ndarray,
    gd_gen_losses: np.ndarray,
    teacher_ranks,
    student_dims,
    num_seeds,
    num_measurements,
    sequence_length,
    results_dir,
):
    """
    Save G&C and GD results to a CSV file.
    Each row: (teacher_rank, student_dim, [gnc_gen_seed_0, ..., gnc_gen_seed_N, gd_gen_seed_0, ..., gd_gen_seed_N])
    """
    results_filename = 'results' + filename_extensions(sequence_length, num_measurements) + '.csv'
    csv_path = results_dir / results_filename
    rows = []
    for t_idx, teacher_rank in enumerate(teacher_ranks):
        for s_idx, student_dim in enumerate(student_dims):
            row = {
                "teacher_rank": teacher_rank,
                "student_dim": student_dim,
            }
            # Add G&C results for all seeds
            for seed in range(num_seeds):
                row[f"gnc_gen_seed={seed}"] = gnc_gen_losses[t_idx, s_idx, seed]
            # Add GD results for all seeds
            for seed in range(num_seeds):
                row[f"gd_gen_seed={seed}"] = gd_gen_losses[t_idx, s_idx, seed]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def main():
    setup_logging(args.log_dir)
    args = parse_args()
    device = get_available_device()
    logging.info(f'Using device {device}.')
    gnc_gen_losses, gd_gen_losses = run_experiment(args, device)

    
    save_results_to_csv(
        gnc_gen_losses,
        gd_gen_losses,
        args.teacher_ranks,
        args.student_dims,
        args.num_seeds,
        args.num_measurements,
        args.sequence_length,
        args.results_dir,
    )

    plot(args.student_dims,
         gnc_gen_losses,
         gd_gen_losses,
         args.teacher_ranks,
         args.sequence_length,
         args.num_measurements,
         args.figures_dir)

    logging.info(f"Finished experiments, results saved to {args.results_dir}, figures saved to {args.figures_dir}")

if __name__ == "__main__":
    main()