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

from loss import get_y_teacher
from plotting import plot

from generator import generate_teacher, generate_dataset
from parser import parse_args
from training import train_gnc, train_gd
from utils import filename_extensions, get_available_gpus
from theoretical_loss import gnc_theoretical_loss


def run_single_seed_worker(experiment_data):
    """
    Worker function for running a single seed experiment.
    This function will be called by DataLoader workers.
    """
    teacher_rank_idx = experiment_data['teacher_rank_idx']
    teacher_rank = experiment_data['teacher_rank']
    student_dim_idx = experiment_data['student_dim_idx']
    student_dim = experiment_data['student_dim']
    seed = experiment_data['seed']
    gpu_id = experiment_data['gpu_id']
    args_dict = experiment_data['args_dict']
    
    # Set up logging for this worker process
    log_file = experiment_data.get('log_file', None)
    if log_file:
        setup_logging(log_file)
    
    # Set the GPU for this process
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    logging.info(f"Starting experiment: teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    try:
        with torch.no_grad():
            # Generate teacher and dataset (now optimized to generate directly on device)
            logging.info(f"Generating teacher on GPU {gpu_id}...")
            start_time = time.time()
            teacher = generate_teacher(teacher_rank, student_dim, device)
            teacher_time = time.time() - start_time
            logging.info(f"Teacher generation completed in {teacher_time:.2f} seconds")
            
            logging.info(f"Generating dataset on GPU {gpu_id}...")
            start_time = time.time()
            dataset = generate_dataset(args_dict['num_measurements'], args_dict['sequence_length'], 
                                    args_dict['input_e1'], device)
            dataset_time = time.time() - start_time
            logging.info(f"Dataset generation completed in {dataset_time:.2f} seconds")
            
            logging.info(f"Calculating y_teacher on GPU {gpu_id}...")
            start_time = time.time()
            y_teacher = get_y_teacher(teacher, dataset)
            y_teacher_time = time.time() - start_time
            logging.info(f"y_teacher calculation completed in {y_teacher_time:.2f} seconds")
            
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
            }
        
            # G&C
            if args_dict['gnc']:
                try:
                    batch_size = args_dict['gnc_batch_size']
                    
                    logging.info(f"Starting G&C training on GPU {gpu_id}...")
                    start_time = time.time()
                    mean_prior, gnc_gen_loss = train_gnc(seed, student_dim, device, y_teacher, dataset,
                                                        args_dict['eps_train'], args_dict['gnc_num_samples'],
                                                        batch_size, args_dict['sequence_length'],
                                                        args_dict['calc_loss_only_on_last_output'])
                    training_time = time.time() - start_time
                    logging.info(f"G&C training completed in {training_time:.2f} seconds on GPU {gpu_id}")
                    results['gnc_gen_loss'] = gnc_gen_loss
                    results['gnc_mean_prior'] = mean_prior
                    
                    logging.info(f"Starting theoretical loss calculation on GPU {gpu_id}...")
                    start_time = time.time()
                    theoretical_loss, theoretical_asymptotic_loss = gnc_theoretical_loss(teacher, dataset, student_dim, device)
                    theory_time = time.time() - start_time
                    logging.info(f"Theoretical loss calculation completed in {theory_time:.2f} seconds on GPU {gpu_id}")
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

        torch.cuda.empty_cache()
        
        logging.info(f"Completed experiment: teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}")
        return results
        
    except Exception as e:
        logging.error(f"Error in run_single_seed_worker for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}, GPU={gpu_id}: {e}")
        logging.error(traceback.format_exc())
        return {
            'teacher_rank_idx': teacher_rank_idx,
            'student_dim_idx': student_dim_idx,
            'seed': seed,
            'gpu_id': gpu_id,
            'error': str(e)
        }


def process_worker(process_id, gpu_id, seed_range, args_dict, teacher_ranks, student_dims, 
                  results_queue, checkpoint_queue, log_file):
    """
    Worker process that runs experiments for a specific range of seeds on a dedicated GPU.
    
    Args:
        process_id: ID of this process
        gpu_id: GPU device ID to use
        seed_range: Range of seeds to process (start, end)
        args_dict: Experiment arguments
        teacher_ranks: List of teacher ranks
        student_dims: List of student dimensions
        results_queue: Queue to send results to main process
        checkpoint_queue: Queue to send checkpoint updates to main process
        log_file: Log file path
    """
    # Set up logging for this process
    setup_logging(log_file)
    
    # Set the GPU for this process
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
    total_experiments = (end_seed - start_seed) * len(teacher_ranks) * len(student_dims)
    
    for seed in range(start_seed, end_seed):
        for teacher_rank_idx, teacher_rank in enumerate(teacher_ranks):
            for student_dim_idx, student_dim in enumerate(student_dims):
                # Set seed for reproducibility
                torch.manual_seed(seed)
                
                # Reduced logging to minimize CPU overhead
                if completed_experiments % 10 == 0:  # Log every 10th experiment
                    logging.info(f"Process {process_id}: Starting experiment - teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}")
                
                try:
                    with torch.no_grad():
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
                            logging.error(f"G&C failed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}: {e}")
                    
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
                            logging.error(f"GD failed for teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}: {e}")
                    
                    del teacher, dataset, y_teacher
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
                    
                    logging.info(f"Process {process_id}: Completed experiment - teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}")
                
                except Exception as e:
                    logging.error(f"Process {process_id}: Error in experiment - teacher_rank={teacher_rank}, student_dim={student_dim}, seed={seed}: {e}")
                    logging.error(traceback.format_exc())
                    
                    # Send error result
                    error_result = {
                        'teacher_rank_idx': teacher_rank_idx,
                        'student_dim_idx': student_dim_idx,
                        'seed': seed,
                        'gpu_id': gpu_id,
                        'process_id': process_id,
                        'error': str(e)
                    }
                    results_queue.put(error_result)
                    completed_experiments += 1
    
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
    gd_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_gen_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_mean_priors = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_theoretical_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))
    gnc_theoretical_asymptotic_losses = np.zeros((len(args.teacher_ranks), len(args.student_dims), args.num_seeds))

    # Get available GPUs
    available_gpus = get_available_gpus(max_gpus=args.max_gpus)
    
    if not available_gpus:
        logging.error("No available GPUs found. Exiting program.")
        print("No available GPUs found. Exiting program.")
        return None, None, None, None, None
    else:
        logging.info(f"Using GPUs: {available_gpus}")
        print(f"Using GPUs: {available_gpus}")

    # Determine number of processes (1-4 based on available GPUs)
    num_processes = min(len(available_gpus), 4)
    logging.info(f"Starting {num_processes} processes on {len(available_gpus)} GPUs")
    print(f"Starting {num_processes} processes on {len(available_gpus)} GPUs")

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
        'calc_loss_only_on_last_output': args.calc_loss_only_on_last_output,
    }

    # Distribute seeds across processes
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
            args=(i, gpu_id, seed_range, args_dict, args.teacher_ranks, args.student_dims,
                  results_queue, checkpoint_queue, args.log_file)
        )
        processes.append(process)
        process.start()
    
    # Monitor progress and collect results
    completed_experiments = 0
    total_experiments = args.num_seeds * len(args.teacher_ranks) * len(args.student_dims)
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
                teacher_rank_idx = result['teacher_rank_idx']
                student_dim_idx = result['student_dim_idx']
                
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
                time.sleep(1)  # 100ms sleep when idle
            else:
                time.sleep(0.1)  # 10ms sleep when processing
    
    # Wait for all processes to finish
    for process in processes:
        process.join()
    
    checkpoint_manager.save_final_checkpoint()
    logging.info("Final checkpoint saved")

    return gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses


def main():
    args = parse_args()
    args.log_file = setup_logging(args.log_dir)
    logging.info(f'Args: {args}')
    gnc_gen_losses, gd_gen_losses, gnc_mean_priors, gnc_theoretical_losses, gnc_theoretical_asymptotic_losses = run_experiment(args)

    results_filename = 'results' + filename_extensions(args) + '.csv'
    plot_filename = 'plot' + filename_extensions(args)

    save_results_to_csv(
        gnc_gen_losses,
        gd_gen_losses,
        gnc_theoretical_losses,
        gnc_theoretical_asymptotic_losses,
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
