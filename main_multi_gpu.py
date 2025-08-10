import multiprocessing
import logging
from utils import setup_logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os
import GPUtil

from loss import get_y_teacher
from plotting import plot

from generator import generate_teacher, generate_dataset
from parser import parse_args
from training import train_gnc, train_gd
from utils import filename_extensions, get_available_gpus
from theoretical_loss import gnc_theoretical_loss


def get_gpu_info():
    """Get detailed information about all GPUs using GPUtil."""
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = {}
        for i, gpu in enumerate(gpus):
            gpu_info[i] = {
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_free': gpu.memoryFree,
                'temperature': gpu.temperature
            }
        return gpu_info
    except Exception as e:
        print(f"Could not get GPU info: {e}")
        return {}


def run_single_seed(args_tuple):
    """
    Run a single seed experiment using the improved GPU selection.
    """
    (teacher_rank_idx, teacher_rank, student_dim_idx, student_dim, seed, args_dict, gpu_id) = args_tuple
    
    # Set the GPU for this process
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    
    try:
        # Generate teacher and dataset
        teacher = generate_teacher(teacher_rank, student_dim, device)
        dataset = generate_dataset(args_dict['num_measurements'], args_dict['sequence_length'], 
                                  args_dict['input_e1'], device)
        y_teacher = get_y_teacher(teacher, dataset)
        
        results = {
            'teacher_rank_idx': teacher_rank_idx,
            'student_dim_idx': student_dim_idx,
            'seed': seed,
            'gpu_id': gpu_id,
            'gnc_gen_loss': None,
            'gnc_mean_prior': None,
            'gnc_theoretical_loss': None,
            'gnc_theoretical_asymptotic_loss': None,
            'gd_gen_loss': None,
            'gd_train_loss': None,
            'memory_used_mb': 0
        }
        
        # G&C
        if args_dict['gnc']:
            try:
                mean_prior, gnc_gen_loss = train_gnc(seed, student_dim, device, y_teacher, dataset,
                                                    args_dict['eps_train'], args_dict['gnc_num_samples'],
                                                    args_dict['gnc_batch_size'], args_dict['sequence_length'],
                                                    args_dict['calc_loss_only_on_last_output'])
                results['gnc_gen_loss'] = gnc_gen_loss
                results['gnc_mean_prior'] = mean_prior
                
                theoretical_loss, theoretical_asymptotic_loss = gnc_theoretical_loss(teacher, dataset, student_dim, device)
                results['gnc_theoretical_loss'] = theoretical_loss.item()
                results['gnc_theoretical_asymptotic_loss'] = theoretical_asymptotic_loss.item()
            except Exception as e:
                print(f"G&C failed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}: {e}")
        
        # GD
        if args_dict['gd']:
            try:
                gd_gen_loss, gd_train_loss = train_gd(seed, student_dim, device, y_teacher, dataset,
                                                     args_dict['gd_init_scale'], args_dict['gd_lr'],
                                                     args_dict['gd_epochs'], args_dict['calc_loss_only_on_last_output'],
                                                     args_dict['gd_optimizer'])
                results['gd_gen_loss'] = gd_gen_loss
                results['gd_train_loss'] = gd_train_loss
            except Exception as e:
                print(f"GD failed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}: {e}")
        
        # Record memory usage
        results['memory_used_mb'] = torch.cuda.memory_allocated(device) / (1024**2)
        
    except Exception as e:
        print(f"Experiment failed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}: {e}")
        results = {
            'teacher_rank_idx': teacher_rank_idx,
            'student_dim_idx': student_dim_idx,
            'seed': seed,
            'gpu_id': gpu_id,
            'error': str(e)
        }
    
    finally:
        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    return results


def run_experiment(args):
    """
    Run experiments in parallel using the improved GPU selection from utils.py
    """
    gd_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_mean_priors = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_theoretical_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_theoretical_asymptotic_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))

    # Get available GPUs using the new utility function
    # You can adjust these parameters based on your needs
    available_gpus = get_available_gpus(max_load=0.3, max_memory=0.3, max_gpus=4)
    
    if not available_gpus:
        print("No available GPUs found. Falling back to CPU.")
        available_gpus = ['cpu']
    else:
        print(f"Using GPUs: {available_gpus}")
        
        # Get detailed GPU info
        gpu_info = get_gpu_info()
        if gpu_info:
            print("GPU Information:")
            for gpu_id in available_gpus:
                if gpu_id in gpu_info:
                    info = gpu_info[gpu_id]
                    print(f"  GPU {gpu_id}: {info['name']}, Load: {info['load']:.1%}, "
                          f"Memory: {info['memory_used']}/{info['memory_total']}MB, "
                          f"Temp: {info['temperature']}°C")

    # Convert args to dict for serialization
    args_dict = {
        'num_measurements': args.num_measurements,
        'sequence_length': args.sequence_length,
        'input_e1': args.input_e1,
        'eps_train': args.eps_train,
        'gnc': args.gnc,
        'gnc_num_samples': args.gnc_num_samples,
        'gnc_batch_size': args.gnc_batch_size,
        'gd': args.gd,
        'gd_lr': args.gd_lr,
        'gd_epochs': args.gd_epochs,
        'gd_init_scale': args.gd_init_scale,
        'gd_optimizer': args.gd_optimizer,
        'calc_loss_only_on_last_output': args.calc_loss_only_on_last_output
    }

    # Track GPU usage statistics
    gpu_stats = {gpu_id: {'jobs': 0, 'total_memory_mb': 0, 'successful_jobs': 0} for gpu_id in available_gpus}

    for teacher_rank_idx, teacher_rank in enumerate(args.teacher_ranks):
        logging.info(f'Starting teacher_rank={teacher_rank}')
        for student_dim_idx, student_dim in enumerate(args.student_dims):
            logging.info(f'Starting student_dim={student_dim}')

            # Create all seed combinations for this teacher_rank and student_dim
            seed_params = []
            for seed in range(args.num_seeds):
                # Distribute seeds across available GPUs in a round-robin fashion
                gpu_id = available_gpus[seed % len(available_gpus)]
                seed_params.append((teacher_rank_idx, teacher_rank, student_dim_idx, student_dim, seed, args_dict, gpu_id))
            
            # Run seeds in parallel
            n_processes = min(len(available_gpus), len(seed_params))
            print(f"Running {len(seed_params)} seeds for teacher_rank={teacher_rank}, student_dim={student_dim} on {n_processes} GPUs")
            
            with multiprocessing.Pool(processes=n_processes) as pool:
                seed_results = list(tqdm(pool.imap(run_single_seed, seed_params), 
                                       total=len(seed_params), desc=f"Seeds for d={student_dim}"))
            
            # Collect results and update GPU stats
            for result in seed_results:
                seed = result['seed']
                gpu_id = result['gpu_id']
                
                # Update GPU statistics
                if gpu_id in gpu_stats:
                    gpu_stats[gpu_id]['jobs'] += 1
                    if 'memory_used_mb' in result:
                        gpu_stats[gpu_id]['total_memory_mb'] += result['memory_used_mb']
                    if 'error' not in result:
                        gpu_stats[gpu_id]['successful_jobs'] += 1
                
                # Skip if there was an error
                if 'error' in result:
                    continue
                
                if result['gnc_gen_loss'] is not None:
                    gnc_gen_losses[teacher_rank_idx, student_dim_idx, seed] = result['gnc_gen_loss']
                if result['gnc_mean_prior'] is not None:
                    gnc_mean_priors[teacher_rank_idx, student_dim_idx, seed] = result['gnc_mean_prior']
                if result['gnc_theoretical_loss'] is not None:
                    gnc_theoretical_losses[teacher_rank_idx, student_dim_idx, seed] = result['gnc_theoretical_loss']
                if result['gnc_theoretical_asymptotic_loss'] is not None:
                    gnc_theoretical_asymptotic_losses[teacher_rank_idx, student_dim_idx, seed] = result['gnc_theoretical_asymptotic_loss']
                if result['gd_gen_loss'] is not None:
                    gd_gen_losses[teacher_rank_idx, student_dim_idx, seed] = result['gd_gen_loss']
            
            # Print GPU usage statistics
            print(f"\nGPU Usage for student_dim={student_dim}:")
            for gpu_id, stats in gpu_stats.items():
                if stats['jobs'] > 0:
                    avg_memory = stats['total_memory_mb'] / stats['jobs']
                    success_rate = stats['successful_jobs'] / stats['jobs'] * 100
                    print(f"  GPU {gpu_id}: {stats['jobs']} jobs, {success_rate:.1f}% success, avg memory: {avg_memory:.1f}MB")
            
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

    return gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses


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
    logging.info(f'Args: {args}')
    
    gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses = run_experiment(args)

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
