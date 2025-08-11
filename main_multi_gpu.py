import multiprocessing
import logging
from utils import setup_logging
import tqdm
import numpy as np
import pandas as pd
import torch
import os
import GPUtil
import traceback

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from loss import get_y_teacher
from plotting import plot

from generator import generate_teacher, generate_dataset
from parser import parse_args
from training import train_gnc, train_gd
from utils import filename_extensions, get_available_gpus
from theoretical_loss import gnc_theoretical_loss
from performance_config import perf_config, optimize_environment


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
    Optimized to reduce CPU bottlenecks and improve GPU utilization.
    """
    (teacher_rank_idx, teacher_rank, student_dim_idx, student_dim, seed, args_dict, gpu_id, log_file) = args_tuple
    
    # Set up logging for this worker process
    setup_logging(log_file)
    
    # Set the GPU for this process
    if gpu_id == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        perf_config.apply_cuda_optimizations()
    
    logging.info(f"Starting experiment: teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    try:
        # Generate teacher and dataset (now optimized to generate directly on device)
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
                # Use optimized batch size if available
                batch_size = args_dict['gnc_batch_size']
                if hasattr(perf_config, 'get_optimal_batch_size'):
                    batch_size = perf_config.get_optimal_batch_size(batch_size)
                
                mean_prior, gnc_gen_loss = train_gnc(seed, student_dim, device, y_teacher, dataset,
                                                    args_dict['eps_train'], args_dict['gnc_num_samples'],
                                                    batch_size, args_dict['sequence_length'],
                                                    args_dict['calc_loss_only_on_last_output'])
                results['gnc_gen_loss'] = gnc_gen_loss
                results['gnc_mean_prior'] = mean_prior
                
                theoretical_loss, theoretical_asymptotic_loss = gnc_theoretical_loss(teacher, dataset, student_dim, device)
                results['gnc_theoretical_loss'] = theoretical_loss.item()
                results['gnc_theoretical_asymptotic_loss'] = theoretical_asymptotic_loss.item()
            
            except Exception as e:
                logging.error(f"G&C failed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}: {e}")
        
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
                logging.error(f"GD failed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}: {e}")
        
        del teacher, dataset, y_teacher

        if device.type == 'cuda':
            results['memory_used_mb'] = torch.cuda.memory_allocated(device) / 1024 / 1024
            torch.cuda.empty_cache()
        
        logging.info(f"Completed experiment: teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}")
        return results
        
    except Exception as e:
        logging.error(f"Error in run_single_seed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}: {e}")
        logging.error(traceback.format_exc())
        return {
            'teacher_rank_idx': teacher_rank_idx,
            'student_dim_idx': student_dim_idx,
            'seed': seed,
            'gpu_id': gpu_id,
            'error': str(e)
        }


def run_experiment(args):
    """
    Run experiments in parallel using the improved GPU selection from utils.py
    Optimized to reduce CPU bottlenecks and improve parallelization.
    """
    # Apply global performance optimizations
    optimize_environment()
    
    gd_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_mean_priors = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_theoretical_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_theoretical_asymptotic_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))

    # Get available GPUs using performance-optimized settings
    available_gpus = get_available_gpus(max_load=perf_config.max_gpu_load, 
                                       max_memory=perf_config.max_gpu_memory, 
                                       max_gpus=perf_config.max_gpus)
    
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

    # Create all parameter combinations for better parallelization
    all_params = []
    for teacher_rank_idx, teacher_rank in enumerate(args.teacher_ranks):
        for student_dim_idx, student_dim in enumerate(args.student_dims):            
            for seed in range(args.num_seeds):
                # Distribute seeds across available GPUs in a round-robin fashion
                gpu_id = available_gpus[seed % len(available_gpus)]
                all_params.append((teacher_rank_idx, teacher_rank, student_dim_idx, student_dim, seed, args_dict, gpu_id, args.log_file))
    
    # Run all experiments in parallel with optimal process count
    n_processes = perf_config.get_process_count(len(all_params), len(available_gpus))
    logging.info(f"Running {len(all_params)} total experiments on {n_processes} processes")
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        all_results = list(tqdm.tqdm(pool.imap(run_single_seed, all_params), 
                                    total=len(all_params), desc="All experiments"))
    
    # Collect results and update GPU stats
    for result in all_results:
        seed = result['seed']
        gpu_id = result['gpu_id']
        teacher_rank_idx = result['teacher_rank_idx']
        student_dim_idx = result['student_dim_idx']
        
        # Update GPU statistics
        if gpu_id in gpu_stats:
            gpu_stats[gpu_id]['jobs'] += 1
            if 'memory_used_mb' in result:
                gpu_stats[gpu_id]['total_memory_mb'] += result['memory_used_mb']
            if 'error' not in result:
                gpu_stats[gpu_id]['successful_jobs'] += 1
        
        # Skip if there was an error
        if 'error' in result:
            logging.error(f"Error in run_single_seed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}: {result['error']}")
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
    logging.info("\nGPU Usage Statistics:")
    for gpu_id, stats in gpu_stats.items():
        if stats['jobs'] > 0:
            success_rate = stats['successful_jobs'] / stats['jobs'] * 100
            avg_memory = stats['total_memory_mb'] / stats['jobs'] if stats['jobs'] > 0 else 0
            logging.info(f"  GPU {gpu_id}: {stats['jobs']} jobs, {success_rate:.1f}% success rate, "
                  f"{avg_memory:.1f}MB avg memory per job")

    return gd_gen_losses, gnc_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses


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
    args.log_file = setup_logging(args.log_dir)
    logging.info(f'Args: {args}')
    
    gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses = run_experiment(args)

    logging.info(f'G&C theoretical losses: {gnc_theoretical_losses}')
    logging.info(f'G&C theoretical asymptotic losses: {gnc_theoretical_asymptotic_losses}')

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
