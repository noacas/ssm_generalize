import time
import logging
from utils import setup_logging
from tqdm.contrib import tenumerate
import tqdm
import numpy as np
import pandas as pd
import torch

from loss import get_y_teacher
from plotting import plot

from generator import generate_teacher, generate_dataset
from parser import parse_args
from training import train_gnc, train_gd
from utils import filename_extensions, get_available_device
from theoretical_loss import gnc_theoretical_loss


def run_experiment(args, device):
    gd_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_mean_priors = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_theoretical_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_theoretical_asymptotic_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))

    for teacher_rank_idx, teacher_rank in tenumerate(args.teacher_ranks, desc="Teacher ranks", position=0, leave=True):
        logging.info(f'Starting teacher_rank={teacher_rank}')
        for student_dim_idx, student_dim in tenumerate(args.student_dims, desc="Student dimensions", position=1, leave=True):
            logging.info(f'Starting student_dim={student_dim}')

            for seed in tqdm.trange(args.num_seeds, desc="Seeds", position=2, leave=True):
                logging.info(f'Starting seed={seed}')
                torch.manual_seed(seed)
                teacher = generate_teacher(teacher_rank, student_dim, device)
                dataset = generate_dataset(args.num_measurements, args.sequence_length, args.input_e1, device)
                y_teacher = get_y_teacher(teacher, dataset)

                # G&C
                if args.gnc:
                    logging.info("Starting G&C")
                    t0 = time.time()
                    mean_prior, gnc_gen_loss = train_gnc(seed,
                                                student_dim,
                                                device,
                                                y_teacher,
                                                dataset,
                                                args.eps_train,
                                                args.gnc_num_samples,
                                                args.gnc_batch_size,
                                                args.sequence_length,
                                                args.calc_loss_only_on_last_output)
                    gnc_gen_losses[teacher_rank_idx, student_dim_idx, seed] = gnc_gen_loss
                    gnc_mean_priors[teacher_rank_idx, student_dim_idx, seed] = mean_prior
                    theoretical_loss, theoretical_asymptotic_loss = gnc_theoretical_loss(teacher, dataset, student_dim, device)
                    gnc_theoretical_losses[teacher_rank_idx, student_dim_idx, seed] = theoretical_loss.item()
                    gnc_theoretical_asymptotic_losses[teacher_rank_idx, student_dim_idx, seed] = theoretical_asymptotic_loss.item()
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
                                        args.calc_loss_only_on_last_output,
                                        args.gd_optimizer)
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
            logging.info(f'Finished student_dim={student_dim}')
            if args.gnc:
                logging.info(f'G&C avg gen_loss = {np.nanmean(gnc_gen_losses[teacher_rank_idx, student_dim_idx, :])}')
                logging.info(f'G&C avg theoretical_loss = {np.nanmean(gnc_theoretical_losses[teacher_rank_idx, student_dim_idx, :])}')
                logging.info(f'G&C avg theoretical_asymptotic_loss = {np.nanmean(gnc_theoretical_asymptotic_losses[teacher_rank_idx, student_dim_idx, :])}')
            if args.gd:
                logging.info(f'GD avg gen_loss = {np.nanmean(gd_gen_losses[teacher_rank_idx, student_dim_idx, :])}')
            logging.info('----------------------------------------------------------------------------------------------------------------------------------------------')

    return gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses


def save_results_to_csv(
    gnc_gen_losses: np.ndarray,
    gd_gen_losses: np.ndarray,
    teacher_ranks,
    student_dims,
    num_seeds,
    results_filename,
    results_dir,
    gd,
    gnc,

):
    """
    Save G&C and GD results to a CSV file.
    Each row: (teacher_rank, student_dim, [gnc_gen_seed_0, ..., gnc_gen_seed_N, gd_gen_seed_0, ..., gd_gen_seed_N])
    """
    csv_path = results_dir / results_filename
    rows = []
    for t_idx, teacher_rank in enumerate(teacher_ranks):
        for s_idx, student_dim in enumerate(student_dims):
            row = {
                "teacher_rank": teacher_rank,
                "student_dim": student_dim,
            }

            if gnc:
                # Add G&C results for all seeds
                for seed in range(num_seeds):
                    row[f"gnc_gen_seed={seed}"] = gnc_gen_losses[t_idx, s_idx, seed]
            
            if gd:
                # Add GD results for all seeds
                for seed in range(num_seeds):
                    row[f"gd_gen_seed={seed}"] = gd_gen_losses[t_idx, s_idx, seed]
            
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def main():
    args = parse_args()
    setup_logging(args.log_dir)
    device = get_available_device()
    logging.info(f'Using device {device}.')
    logging.info(f'Args: {args}')
    gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses = run_experiment(args, device)

    results_filename = 'results' + filename_extensions(args) + '.csv'
    plot_filename = 'plot' + filename_extensions(args)

    save_results_to_csv(
        gnc_gen_losses,
        gd_gen_losses,
        args.teacher_ranks,
        args.student_dims,
        args.num_seeds,
        results_filename,
        args.results_dir,
        args.gd,
        args.gnc,
    )

    plot(args.student_dims,
         gnc_gen_losses,
         gd_gen_losses,
         gnc_mean_priors,
         gnc_theoretical_losses,
         gnc_theoretical_asymptotic_losses,
         args.teacher_ranks,
         args.sequence_length,
         plot_filename,
         args.figures_dir,
         args.gnc,
         args.gd)

    logging.info(f"Finished experiments, results saved to {results_filename}, figures saved to {plot_filename}")

if __name__ == "__main__":
    main()