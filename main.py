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

from generator import generate_teacher_alpha, generate_w_sequences, get_alpha_w_pair
from parser import parse_args
from training import train_gnc, train_gd
from utils import filename_extensions, get_available_gpus, get_available_device
from theoretical_loss import gnc_theoretical_loss


def process_worker(process_id, gpu_id, seed_list, args_dict, student_dims, 
                  results_queue, checkpoint_queue, log_file):
    """
    Worker process that runs experiments for a specific list of seeds on a dedicated GPU.
    
    Args:
        process_id: ID of this process
        gpu_id: GPU device ID to use
        seed_list: List of seeds to process
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
    
    logging.info(f"Process {process_id} started on GPU {gpu_id}, processing seeds {seed_list}")
    
    completed_experiments = 0
    total_experiments = len(seed_list) * len(student_dims)
    
    for seed_idx, seed in enumerate(seed_list):
        torch.manual_seed(seed)
        with torch.no_grad():
            # Use file loading if data file is provided, otherwise use seed-based generation
            if args_dict.get('data_file') is not None:
                alpha_teacher, w_sequences = get_alpha_w_pair(args_dict.get('data_file'), device, seed_idx)
                logging.info(f"for seed {seed}, loaded alpha_teacher={alpha_teacher}, loaded {len(w_sequences)} sequences: {w_sequences}")
            else:
                alpha_teacher = generate_teacher_alpha(device)
                w_sequences = generate_w_sequences(args_dict['sequence_length'], args_dict['num_sequences'], device, args_dict, alpha_teacher)
                logging.info(f"for seed {seed}, alpha_teacher={alpha_teacher}, generated {len(w_sequences)} sequences: {w_sequences}")

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
                            mean_prior, gnc_gen_loss = train_gnc(seed, student_dim, device, alpha_teacher, w_sequences,
                                                                args_dict['eps_train'], args_dict['gnc_num_samples'],
                                                                batch_size, False)
                            results['gnc_gen_loss'] = gnc_gen_loss
                            results['gnc_mean_prior'] = mean_prior
                            
                            # Use first sequence for theoretical loss calculation (for backward compatibility)
                            theoretical_loss, theoretical_asymptotic_loss, delta_l_infinity = gnc_theoretical_loss(alpha_teacher, w_sequences, student_dim, device)
                            results['gnc_theoretical_loss'] = theoretical_loss.item()
                            results['gnc_theoretical_asymptotic_loss'] = theoretical_asymptotic_loss.item()
                            logging.info(f"For seed {seed}, student_dim {student_dim}, G&C theoretical loss: {theoretical_loss.item()}, asymptotic loss: {theoretical_asymptotic_loss.item()}, G&C empirical loss: {gnc_gen_loss}")
                        
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
                        scheduler_type = args_dict.get('gd_scheduler')
                        
                        # Try to construct scheduler params from individual parameters first
                        scheduler_params = {}
                        if scheduler_type == 'exponential':
                            # Check if we have exp_gamma parameter
                            exp_gamma = args_dict.get('exp_gamma')
                            if exp_gamma is not None:
                                scheduler_params = {'gamma': exp_gamma}
                                logging.info(f"Using exponential scheduler with gamma from exp_gamma: {exp_gamma}")
                            else:
                                scheduler_params = {'gamma': 0.955}  # Default gamma for exponential
                                logging.info(f"Using exponential scheduler with default gamma: {scheduler_params}")
                        elif scheduler_type == 'step':
                            # Check if we have step_size and step_gamma parameters
                            step_size = args_dict.get('step_size')
                            step_gamma = args_dict.get('step_gamma')
                            if step_size is not None and step_gamma is not None:
                                scheduler_params = {'step_size': step_size, 'gamma': step_gamma}
                                logging.info(f"Using step scheduler with step_size={step_size}, gamma={step_gamma}")
                            else:
                                scheduler_params = {'step_size': 1000, 'gamma': 0.1}  # Default for step
                                logging.info(f"Using step scheduler with defaults: {scheduler_params}")
                        elif scheduler_type == 'cosine':
                            # Check if we have cosine_eta_min parameter
                            cosine_eta_min = args_dict.get('cosine_eta_min')
                            if cosine_eta_min is not None:
                                scheduler_params = {'T_max': args_dict['gd_epochs'], 'eta_min': cosine_eta_min}
                                logging.info(f"Using cosine scheduler with eta_min={cosine_eta_min}")
                            else:
                                scheduler_params = {'T_max': args_dict['gd_epochs'], 'eta_min': 0}  # Default for cosine
                                logging.info(f"Using cosine scheduler with defaults: {scheduler_params}")
                        else:
                            scheduler_params = {}
                            logging.info(f"No scheduler or scheduler type '{scheduler_type}', using empty params")
                        
                        # If we still don't have params and there's a scheduler_params_str, try to parse it
                        if not scheduler_params and scheduler_params_str and scheduler_params_str != '{}':
                            try:
                                logging.info(f"Attempting to parse scheduler params: '{scheduler_params_str}'")
                                scheduler_params = json.loads(scheduler_params_str)
                                logging.info(f"Successfully parsed scheduler params: {scheduler_params}")
                            except json.JSONDecodeError as e:
                                logging.warning(f"JSON decode error: {e}")
                                logging.warning(f"Failed to parse scheduler params, using constructed params: {scheduler_params}")
                        
                        logging.info(f"Calling train_gd with scheduler_params: {scheduler_params}")
                        gd_gen_loss, gd_train_loss = train_gd(student_dim, device, alpha_teacher, w_sequences,
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

        del w_sequences, alpha_teacher
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
    
    # Determine which seeds to use
    if hasattr(args, 'seeds') and args.seeds is not None:
        seeds_to_use = args.seeds
        logging.info(f"Using custom seeds: {seeds_to_use}")
        print(f"Using custom seeds: {seeds_to_use}")
    else:
        seeds_to_use = list(range(args.num_seeds))
        logging.info(f"Using default seeds: {seeds_to_use}")
        print(f"Using default seeds: {seeds_to_use}")
    
    # Initialize result arrays
    gd_gen_losses = np.zeros((len(args.student_dims), len(seeds_to_use)))
    gnc_gen_losses = np.zeros((len(args.student_dims), len(seeds_to_use)))
    gnc_mean_priors = np.zeros((len(args.student_dims), len(seeds_to_use)))
    gnc_theoretical_losses = np.zeros((len(args.student_dims), len(seeds_to_use)))
    gnc_theoretical_asymptotic_losses = np.zeros((len(args.student_dims), len(seeds_to_use)))

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

    # Determine number of processes (1-max-gpus based on available GPUs)
    num_processes = min(len(available_gpus), args.max_gpus)
    if num_processes == 0:
        logging.error("No GPUs available for processing. Exiting program.")
        print("No GPUs available for processing. Exiting program.")
        return None, None, None, None, None
    logging.info(f"Starting {num_processes} processes on {len(available_gpus)} GPUs")
    print(f"Starting {num_processes} processes on {len(available_gpus)} GPUs")

    # Convert args to dict for serialization
    args_dict = {
        'sequence_length': args.sequence_length,
        'num_sequences': args.num_sequences,
        'eps_train': args.eps_train,
        'w_that_minimizes_loss': args.w_that_minimizes_loss,
        'w2_that_minimizes_loss': args.w2_that_minimizes_loss,
        'data_file': args.data_file,
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
        'exp_gamma': args.exp_gamma,
        'step_size': args.step_size,
        'step_gamma': args.step_gamma,
        'cosine_eta_min': args.cosine_eta_min,
    }

    # Distribute seeds across processes
    num_processes = min(num_processes, len(seeds_to_use))
    if num_processes == 0:
        logging.error("No processes available for processing. Exiting program.")
        print("No processes available for processing. Exiting program.")
        return None, None, None, None, None
    
    seeds_per_process = len(seeds_to_use) // num_processes
    remaining_seeds = len(seeds_to_use) % num_processes
    
    seed_ranges = []
    start_idx = 0
    for i in range(num_processes):
        end_idx = start_idx + seeds_per_process + (1 if i < remaining_seeds else 0)
        process_seeds = seeds_to_use[start_idx:end_idx]
        seed_ranges.append(process_seeds)
        start_idx = end_idx

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create queues for communication
    results_queue = Queue()
    checkpoint_queue = Queue()
    
    # Create and start processes
    processes = []
    for i in range(num_processes):
        gpu_id = available_gpus[i % len(available_gpus)]
        seed_list = seed_ranges[i]
        
        process = Process(
            target=process_worker,
            args=(i, gpu_id, seed_list, args_dict, args.student_dims,
                  results_queue, checkpoint_queue, args.log_file)
        )
        processes.append(process)
        process.start()
    
    # Monitor progress and collect results
    completed_experiments = 0
    total_experiments = len(seeds_to_use) * len(args.student_dims)
    completed_processes = 0
    
    # Initialize checkpoint manager with total experiments count
    checkpoint_manager.update_results({}, 0, total_experiments)
    
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
                seed_idx = seeds_to_use.index(seed)  # Find the index of this seed in our list
                
                if result['gnc_gen_loss'] is not None:
                    gnc_gen_losses[student_dim_idx, seed_idx] = result['gnc_gen_loss']
                if result['gnc_mean_prior'] is not None:
                    gnc_mean_priors[student_dim_idx, seed_idx] = result['gnc_mean_prior']
                if result['gnc_theoretical_loss'] is not None:
                    gnc_theoretical_losses[student_dim_idx, seed_idx] = result['gnc_theoretical_loss']
                if result['gnc_theoretical_asymptotic_loss'] is not None:
                    gnc_theoretical_asymptotic_losses[student_dim_idx, seed_idx] = result['gnc_theoretical_asymptotic_loss']
                if result['gd_gen_loss'] is not None:
                    gd_gen_losses[student_dim_idx, seed_idx] = result['gd_gen_loss']
                
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

    # Determine which seeds were used (same logic as in run_experiment)
    if hasattr(args, 'seeds') and args.seeds is not None:
        seeds_to_use = args.seeds
    else:
        seeds_to_use = list(range(args.num_seeds))

    try:
        save_results_to_csv(
            gnc_gen_losses,
            gd_gen_losses,
            gnc_theoretical_losses,
            gnc_theoretical_asymptotic_losses,
            args.student_dims,
            seeds_to_use,
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
                args.gd,
                seeds_to_use)
        logging.info(f"Figures saved to {plot_filename}")
        print(f"Figures saved to {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to create plots: {e}")
        print(f"Failed to create plots: {e}")

    logging.info(f"Finished experiments, results saved to {results_filename}, figures saved to {plot_filename}")
    print(f"Finished experiments, results saved to {results_filename}, figures saved to {plot_filename}")


if __name__ == "__main__":
    main()
