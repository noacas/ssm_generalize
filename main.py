import logging
from utils import setup_logging
import tqdm
import numpy as np
import torch
import traceback
import time
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import os
from save_results import save_results_to_csv
from checkpoint import CheckpointManager

from plotting import plot

from generator import generate_teacher_alpha, generate_w
from parser import parse_args
from training import train_gnc, train_gd
from utils import filename_extensions, get_available_gpus, get_available_device
from theoretical_loss import gnc_theoretical_loss


def process_worker(process_id, gpu_id, seed_range, args_dict, student_dims, 
                  results_queue, checkpoint_queue, log_file):
    """
    Worker process that runs experiments for a specific range of seeds on a dedicated GPU.
    
    Args:
        process_id: ID of this process
        gpu_id: GPU device ID to use
        seed_range: Range of seeds to process (start, end)
        args_dict: Experiment arguments
        student_dims: List of student dimensions
        results_queue: Queue to send results to main process
        checkpoint_queue: Queue to send checkpoint updates to main process
        log_file: Log file path
    """
    # Set up logging for this process
    setup_logging(log_file)
    
    # Set the GPU for this process
    if gpu_id == "mps":
        device = torch.device("mps")
    else:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    
    # Optimize PyTorch for GPU usage
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logging.info(f"Process {process_id} started on GPU {gpu_id}, processing seeds {seed_range[0]}-{seed_range[1]-1}")
    
    start_seed, end_seed = seed_range
    completed_experiments = 0
    total_experiments = (end_seed - start_seed) * len(student_dims)
    
    for seed in range(start_seed, end_seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            alpha_teacher = generate_teacher_alpha(device)
            w = generate_w(args_dict['sequence_length'], device)
            logging.info(f"for seed {seed}, alpha_teacher={alpha_teacher}, w={w}")

        for student_dim_idx, student_dim in enumerate(student_dims):
            # Set seed for reproducibility
            torch.manual_seed(seed)
            
            # Reduced logging to minimize CPU overhead
            if completed_experiments % 10 == 0:  # Log every 10th experiment
                logging.info(f"Process {process_id}: Starting experiment - student_dim={student_dim}, seed={seed}")
            
            try:
                with torch.no_grad():
                    results = {
                        'student_dim_idx': student_dim_idx,
                        'seed': seed,
                        'gpu_id': gpu_id,
                        'process_id': process_id,
                        'gnc_gen_loss': None,
                        'gnc_mean_prior': None,
                        'gnc_theoretical_loss': None,
                        'gnc_theoretical_asymptotic_loss': None,
                        'gd_gen_loss': None,
                        'gd_train_loss': None,
                    }
                
                    # G&C
                    if args_dict['gnc']:
                        try:
                            batch_size = args_dict['gnc_batch_size']
                            mean_prior, gnc_gen_loss = train_gnc(seed, student_dim, device, alpha_teacher, w,
                                                                args_dict['eps_train'], args_dict['gnc_num_samples'],
                                                                batch_size
                                                                )
                            results['gnc_gen_loss'] = gnc_gen_loss
                            results['gnc_mean_prior'] = mean_prior
                            
                            theoretical_loss, theoretical_asymptotic_loss, delta_l_infinity = gnc_theoretical_loss(alpha_teacher, w, student_dim, device)
                            results['gnc_theoretical_loss'] = theoretical_loss.item()
                            results['gnc_theoretical_asymptotic_loss'] = theoretical_asymptotic_loss.item()
                        
                        except Exception as e:
                            logging.error(f"G&C failed for student_dim={student_dim}, seed={seed}: {e}")
                            logging.error(traceback.format_exc())
                    
                # GD
                if args_dict['gd']:
                    try:
                        # Parse scheduler parameters
                        import json
                        # Handle scheduler parameters - use defaults if not provided or if parsing fails
                        scheduler_params_str = args_dict.get('gd_scheduler_params', None)
                        if scheduler_params_str is None or scheduler_params_str == '{}':
                            # No scheduler params provided, use defaults based on scheduler type
                            scheduler_type = args_dict.get('gd_scheduler')
                            if scheduler_type == 'exponential':
                                scheduler_params = {'gamma': 0.955}  # Default gamma for exponential (close to optimized value)
                            elif scheduler_type == 'step':
                                scheduler_params = {'step_size': 1000, 'gamma': 0.1}  # Default for step
                            elif scheduler_type == 'cosine':
                                scheduler_params = {'T_max': args_dict['gd_epochs'], 'eta_min': 0}  # Default for cosine
                            else:
                                scheduler_params = {}
                            logging.info(f"No scheduler params provided, using defaults: {scheduler_params}")
                        else:
                            try:
                                logging.info(f"Attempting to parse scheduler params: '{scheduler_params_str}'")
                                scheduler_params = json.loads(scheduler_params_str)
                                logging.info(f"Successfully parsed scheduler params: {scheduler_params}")
                            except json.JSONDecodeError as e:
                                # If JSON parsing fails, create default scheduler params based on scheduler type
                                logging.warning(f"JSON decode error: {e}")
                                scheduler_type = args_dict.get('gd_scheduler')
                                if scheduler_type == 'exponential':
                                    scheduler_params = {'gamma': 0.955}  # Default gamma for exponential (close to optimized value)
                                elif scheduler_type == 'step':
                                    scheduler_params = {'step_size': 1000, 'gamma': 0.1}  # Default for step
                                elif scheduler_type == 'cosine':
                                    scheduler_params = {'T_max': args_dict['gd_epochs'], 'eta_min': 0}  # Default for cosine
                                else:
                                    scheduler_params = {}
                                logging.warning(f"Failed to parse scheduler params, using defaults: {scheduler_params}")
                        
                        logging.info(f"Calling train_gd with scheduler_params: {scheduler_params}")
                        gd_gen_loss, gd_train_loss = train_gd(student_dim, device, alpha_teacher, w,
                                                                args_dict['gd_init_scale'], args_dict['gd_lr'],
                                                                args_dict['gd_epochs'],
                                                                args_dict['gd_optimizer'],
                                                                args_dict.get('gd_scheduler'),
                                                                scheduler_params,
                                                                args_dict['gd_init_type'])
                        results['gd_gen_loss'] = gd_gen_loss
                        results['gd_train_loss'] = gd_train_loss
                    
                    except Exception as e:
                        logging.error(f"GD failed for student_dim={student_dim}, seed={seed}: {e}")
                
                torch.cuda.empty_cache()
                
                # Send result to main process
                results_queue.put(results)
                
                completed_experiments += 1
                
                # Batch operations to reduce CPU overhead
                if completed_experiments % 10 == 0:
                    # Send checkpoint update
                    checkpoint_queue.put({
                        'type': 'progress',
                        'process_id': process_id,
                        'completed': completed_experiments,
                        'total': total_experiments
                    })
                
                logging.info(f"Process {process_id}: Completed experiment - student_dim={student_dim}, seed={seed}")
            
            except Exception as e:
                logging.error(f"Process {process_id}: Error in experiment - student_dim={student_dim}, seed={seed}: {e}")
                logging.error(traceback.format_exc())
                
                # Send error result
                error_result = {
                    'student_dim_idx': student_dim_idx,
                    'seed': seed,
                    'gpu_id': gpu_id,
                    'process_id': process_id,
                    'error': str(e)
                }
                results_queue.put(error_result)
                completed_experiments += 1

        del w, alpha_teacher
        torch.cuda.empty_cache()
    
    # Send completion signal
    checkpoint_queue.put({
        'type': 'completion',
        'process_id': process_id,
        'completed': completed_experiments,
        'total': total_experiments
    })
    
    logging.info(f"Process {process_id} completed on GPU {gpu_id}")


def run_experiment(args):
    # Initialize checkpoint manager
    checkpoint_interval = max(args.checkpoint_interval, 7200)  # At least 2 hours
    checkpoint_manager = CheckpointManager(args, checkpoint_interval=checkpoint_interval)
    
    # Initialize result arrays
    gd_gen_losses = np.zeros((len(args.student_dims), args.num_seeds))
    gnc_gen_losses = np.zeros((len(args.student_dims), args.num_seeds))
    gnc_mean_priors = np.zeros((len(args.student_dims), args.num_seeds))
    gnc_theoretical_losses = np.zeros((len(args.student_dims), args.num_seeds))
    gnc_theoretical_asymptotic_losses = np.zeros((len(args.student_dims), args.num_seeds))

    # Get available GPUs
    available_gpus = get_available_gpus(max_gpus=args.max_gpus)
    if len(available_gpus) > 0:
        logging.info(f"Using GPUs: {available_gpus}")
        print(f"Using GPUs: {available_gpus}")
    else:
        if get_available_device() == torch.device("mps"):
            available_gpus = ["mps"]
            logging.info(f"Using MPS")
            print(f"Using MPS")
        else:
            logging.error("No available GPUs found. Exiting program.")
            print("No available GPUs found. Exiting program.")
            return None, None, None, None, None

    # Determine number of processes (1-4 based on available GPUs)
    num_processes = min(len(available_gpus), 4)
    if num_processes == 0:
        logging.error("No GPUs available for processing. Exiting program.")
        print("No GPUs available for processing. Exiting program.")
        return None, None, None, None, None
    logging.info(f"Starting {num_processes} processes on {len(available_gpus)} GPUs")
    print(f"Starting {num_processes} processes on {len(available_gpus)} GPUs")

    # Convert args to dict for serialization
    args_dict = {
        'sequence_length': args.sequence_length,
        'eps_train': args.eps_train,
        'gnc': args.gnc,
        'gnc_num_samples': args.gnc_num_samples,
        'gnc_batch_size': args.gnc_batch_size,
        'gd': args.gd,
        'gd_lr': args.gd_lr,
        'gd_epochs': args.gd_epochs,
        'gd_init_scale': args.gd_init_scale,
        'gd_optimizer': args.gd_optimizer,
        'gd_init_type': args.gd_init_type,
        'gd_scheduler': args.gd_scheduler,
        'gd_scheduler_params': args.gd_scheduler_params,
    }

    # Distribute seeds across processes
    num_processes = min(num_processes, args.num_seeds)
    if num_processes == 0:
        logging.error("No processes available for processing. Exiting program.")
        print("No processes available for processing. Exiting program.")
        return None, None, None, None, None
    seeds_per_process = args.num_seeds // num_processes
    remaining_seeds = args.num_seeds % num_processes
    
    seed_ranges = []
    start_seed = 0
    for i in range(num_processes):
        end_seed = start_seed + seeds_per_process + (1 if i < remaining_seeds else 0)
        seed_ranges.append((start_seed, end_seed))
        start_seed = end_seed

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create queues for communication
    results_queue = Queue()
    checkpoint_queue = Queue()
    
    # Create and start processes
    processes = []
    for i in range(num_processes):
        gpu_id = available_gpus[i % len(available_gpus)]
        seed_range = seed_ranges[i]
        
        process = Process(
            target=process_worker,
            args=(i, gpu_id, seed_range, args_dict, args.student_dims,
                  results_queue, checkpoint_queue, args.log_file)
        )
        processes.append(process)
        process.start()
    
    # Monitor progress and collect results
    completed_experiments = 0
    total_experiments = args.num_seeds * len(args.student_dims)
    completed_processes = 0
    
    # Track process completion
    process_completion = {i: False for i in range(num_processes)}
    
    with tqdm.tqdm(total=total_experiments, desc="Processing experiments") as pbar:
        while completed_processes < num_processes:
            # Process one item from each queue per iteration
            processed_something = False
            
            # Check for checkpoint updates (non-blocking)
            try:
                checkpoint_data = checkpoint_queue.get_nowait()
                processed_something = True
                
                if checkpoint_data['type'] == 'completion':
                    process_completion[checkpoint_data['process_id']] = True
                    completed_processes += 1
                    logging.info(f"Process {checkpoint_data['process_id']} completed")
                elif checkpoint_data['type'] == 'progress':
                    # Update progress bar
                    pbar.update(checkpoint_data['completed'] - pbar.n)
            except:
                pass
            
            # Check for results (non-blocking)
            try:
                result = results_queue.get_nowait()
                processed_something = True
                completed_experiments += 1
                
                if 'error' in result:
                    logging.error(f"Error in experiment: {result['error']}")
                    pbar.update(1)
                    continue
                
                seed = result['seed']
                student_dim_idx = result['student_dim_idx']
                
                if result['gnc_gen_loss'] is not None:
                    gnc_gen_losses[student_dim_idx, seed] = result['gnc_gen_loss']
                if result['gnc_mean_prior'] is not None:
                    gnc_mean_priors[student_dim_idx, seed] = result['gnc_mean_prior']
                if result['gnc_theoretical_loss'] is not None:
                    gnc_theoretical_losses[student_dim_idx, seed] = result['gnc_theoretical_loss']
                if result['gnc_theoretical_asymptotic_loss'] is not None:
                    gnc_theoretical_asymptotic_losses[student_dim_idx, seed] = result['gnc_theoretical_asymptotic_loss']
                if result['gd_gen_loss'] is not None:
                    gd_gen_losses[student_dim_idx, seed] = result['gd_gen_loss']
                
                # Update checkpoint every 50 experiments
                if completed_experiments % 50 == 0:
                    checkpoint_manager.update_results({
                        'gnc_gen_losses': gnc_gen_losses,
                        'gd_gen_losses': gd_gen_losses,
                        'gnc_mean_priors': gnc_mean_priors,
                        'gnc_theoretical_losses': gnc_theoretical_losses,
                        'gnc_theoretical_asymptotic_losses': gnc_theoretical_asymptotic_losses
                    }, completed_experiments, total_experiments)
                
                pbar.update(1)
            except:
                pass
            
            # Sleep to reduce CPU usage
            if not processed_something:
                time.sleep(10)  # 100ms sleep when idle
            else:
                time.sleep(1)  # 10ms sleep when processing
    
    # Wait for all processes to finish
    for process in processes:
        process.join()
    
    checkpoint_manager.save_final_checkpoint()
    logging.info("Final checkpoint saved")

    return gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses


def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.log_file = setup_logging(args.log_dir, timestamp)
    results_filename = 'results_' + filename_extensions(args, timestamp) + '.csv'
    plot_filename = 'plot_' + filename_extensions(args, timestamp)
    logging.info(f'Args: {args}')
    
    # Run experiment and handle potential None return
    experiment_results = run_experiment(args)
    
    if experiment_results is None:
        logging.error("Experiment failed - no results to save or plot")
        print("Experiment failed - no results to save or plot")
        return
    
    gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses = experiment_results

    

    try:
        save_results_to_csv(
            gnc_gen_losses,
            gd_gen_losses,
            gnc_theoretical_losses,
            gnc_theoretical_asymptotic_losses,
            args.student_dims,
            args.num_seeds,
            results_filename,
            args.results_dir,
            args.gd,
            args.gnc,
        )
        logging.info(f"Results saved to {results_filename}")
        print(f"Results saved to {results_filename}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        print(f"Failed to save results: {e}")

    try:
        plot(args.student_dims,
                gnc_gen_losses,
                gd_gen_losses,
                gnc_mean_priors,
                gnc_theoretical_losses,
                gnc_theoretical_asymptotic_losses,
                args.sequence_length,
                plot_filename,
                args.figures_dir,
                args.gnc,
                args.gd)
        logging.info(f"Figures saved to {plot_filename}")
        print(f"Figures saved to {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to create plots: {e}")
        print(f"Failed to create plots: {e}")

    logging.info(f"Finished experiments, results saved to {results_filename}, figures saved to {plot_filename}")
    print(f"Finished experiments, results saved to {results_filename}, figures saved to {plot_filename}")


if __name__ == "__main__":
    main()
