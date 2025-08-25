#!/usr/bin/env python3
"""
Hyperparameter optimization for Gradient Descent parameters.

This script uses Optuna to find optimal hyperparameters for:
- gd_lr: Learning rate for GD
- gd_epochs: Number of epochs for GD  
- gd_init_scale: Initialization scale for GD
- gd_optimizer: Optimizer choice (adam or gd)

Usage:
    python gd_hyperopt.py --n_trials 100 --study_name "gd_optimization"
"""

import argparse
import json
import logging
import pathlib
import time
import os
import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import Dict, Any, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
from training import train_gd
import torch
import tqdm
from generator import generate_w
from datetime import datetime

# Import your existing parser to reuse the argument structure
from parser import parse_args
from utils import get_available_gpus, get_available_device, setup_logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDHyperoptObjective:
    """Objective function for GD hyperparameter optimization."""
    
    def __init__(self, base_args: argparse.Namespace, gpu_id: str = "0"):
        """
        Initialize the objective function.
        
        Args:
            base_args: Base arguments from parser
            gpu_id: GPU device ID to use
        """
        self.base_args = base_args
        self.gpu_id = gpu_id
        self.best_score = float('-inf')
        
        # Set up device
        if gpu_id == "mps":
            self.device = torch.device("mps")
        else:
            self.device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(self.device)
        
        # Optimize PyTorch for GPU usage
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function to be optimized by Optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Score to minimize (loss) or maximize (accuracy)
        """
        # Suggest hyperparameters
        gd_lr = trial.suggest_float("gd_lr", 1e-6, 1e-2, log=True)
        gd_epochs = trial.suggest_int("gd_epochs", 50000, 100000, log=True)
        gd_init_scale = trial.suggest_float("gd_init_scale", 1e-4, 1e-1, log=True)
        gd_optimizer = trial.suggest_categorical("gd_optimizer", ["adam", "gd"])
        gd_init_type = trial.suggest_categorical("gd_init_type", ["double_max_A_j"]) #["regular", "near_one", "double_max_A_j"])
        
        # Suggest scheduler parameters
        gd_scheduler = trial.suggest_categorical("gd_scheduler", ["none", "step", "exponential", "cosine"])
        
        # Initialize scheduler parameters
        scheduler_params = {}
        if gd_scheduler == "step":
            scheduler_params["step_size"] = trial.suggest_int("step_size", 100, gd_epochs // 3, log=True)
            scheduler_params["gamma"] = trial.suggest_float("step_gamma", 0.1, 0.9, log=True)
        elif gd_scheduler == "exponential":
            scheduler_params["gamma"] = trial.suggest_float("exp_gamma", 0.9, 0.999, log=True)
        elif gd_scheduler == "cosine":
            scheduler_params["T_max"] = gd_epochs
            scheduler_params["eta_min"] = trial.suggest_float("cosine_eta_min", gd_lr * 1e-3, gd_lr * 0.1, log=True)
        
        # Update base arguments with suggested hyperparameters
        args = self._update_args(gd_lr, gd_epochs, gd_init_scale, gd_optimizer, gd_scheduler, scheduler_params, gd_init_type)
        
        # Run the experiment and get results
        score = self._run_experiment(args, trial)
        
        # Report intermediate values for pruning
        trial.report(score, step=trial.number)
        
        if score < self.best_score:
            self.best_score = score
                
        return score

    
    def _update_args(self, gd_lr: float, gd_epochs: int, gd_init_scale: float, gd_optimizer: str, gd_scheduler: str, scheduler_params: dict, gd_init_type: str) -> argparse.Namespace:
        """Update base arguments with new hyperparameters."""
        # Create a copy of base args
        args = argparse.Namespace(**vars(self.base_args))
        
        # Update GD parameters
        args.gd = True
        args.gd_lr = gd_lr
        args.gd_epochs = gd_epochs
        args.gd_init_scale = gd_init_scale
        args.gd_optimizer = gd_optimizer
        args.gd_scheduler = gd_scheduler if gd_scheduler != "none" else None
        args.scheduler_params = scheduler_params
        args.gd_init_type = gd_init_type
        return args
    
    def _run_experiment(self, args: argparse.Namespace, trial: optuna.Trial) -> float:
        """
        Run a full experiment with given hyperparameters using the main.py pipeline.
        
        Args:
            args: Arguments for the experiment
            trial: Optuna trial object
            
        Returns:
            gd_gen_loss (lower is better)
        """               
        scheduler_info = f", scheduler={args.gd_scheduler}" if args.gd_scheduler else ", scheduler=none"
        logger.info(f"Trial {trial.number} (GPU {self.gpu_id}): lr={args.gd_lr:.2e}, epochs={args.gd_epochs}, "
                   f"init_scale={args.gd_init_scale:.2e}, optimizer={args.gd_optimizer}{scheduler_info}")
        
        student_dim = 150
        alpha_teacher = 0.5
        torch.manual_seed(0)
        w = generate_w(sequence_length=5, device=self.device)
        gd_gen_loss, _ = train_gd(student_dim, self.device, alpha_teacher, w,
                                  args.gd_init_scale, args.gd_lr,
                                  args.gd_epochs,
                                  args.gd_optimizer,
                                  args.gd_scheduler,
                                  args.scheduler_params,
                                  args.gd_init_type)

        return gd_gen_loss


def hyperopt_worker(process_id: int, gpu_id: str, trial_range: tuple, base_args: argparse.Namespace, 
                   study_name: str, results_queue: Queue, log_file: pathlib.Path):
    """
    Worker process that runs hyperopt trials on a dedicated GPU.
    
    Args:
        process_id: ID of this process
        gpu_id: GPU device ID to use
        trial_range: Range of trials to process (start, end)
        base_args: Base experiment arguments
        study_name: Name of the Optuna study
        results_queue: Queue to send results to main process
    """
    # Set up logging for this process
    setup_logging(log_file)
    
    logger.info(f"Hyperopt worker {process_id} started on GPU {gpu_id}, processing trials {trial_range[0]}-{trial_range[1]-1}")
    
    # Create study for this worker
    study = create_study(study_name, "minimize")
    
    # Create objective function for this GPU
    objective = GDHyperoptObjective(base_args, gpu_id)
    
    start_trial, end_trial = trial_range
    completed_trials = 0
    
    for trial_num in range(start_trial, end_trial):
        try:
            # Create a new trial
            trial = study.ask()
            
            # Run the objective function
            score = objective(trial)
            
            # Tell the study the result
            study.tell(trial, score)
            
            completed_trials += 1
            
            # Send progress update
            results_queue.put({
                'type': 'progress',
                'process_id': process_id,
                'trial_number': trial_num,
                'score': score,
                'completed': completed_trials,
                'total': end_trial - start_trial
            })
            
            logger.info(f"Worker {process_id}: Completed trial {trial_num}, score: {score:.6f}")
            
        except Exception as e:
            logger.error(f"Worker {process_id}: Error in trial {trial_num}: {e}")
            # Send error result
            results_queue.put({
                'type': 'error',
                'process_id': process_id,
                'trial_number': trial_num,
                'error': str(e)
            })
    
    # Send completion signal
    results_queue.put({
        'type': 'completion',
        'process_id': process_id,
        'completed': completed_trials,
        'total': end_trial - start_trial
    })
    
    logger.info(f"Hyperopt worker {process_id} completed on GPU {gpu_id}")


def create_study(study_name: str, 
                direction: str = "minimize") -> optuna.Study:
    """
    Create or load an Optuna study.
    
    Args:
        study_name: Name of the study
        direction: Optimization direction ("minimize" or "maximize")
        
    Returns:
        Optuna study object
    """
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    # Use SQLite storage to persist study across processes
    storage = optuna.storages.RDBStorage(
        url="sqlite:///hyperopt_study.db",
        engine_kwargs={"connect_args": {"timeout": 30}}
    )
    
    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        storage=storage,
        load_if_exists=True
    )
    
    return study


def save_results(study: optuna.Study, output_dir: pathlib.Path, 
                study_name: str) -> None:
    """
    Save optimization results to files.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save results
        study_name: Name of the study
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        "study_name": study_name,
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": len(study.trials),
        "optimization_history": [
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    # Save as JSON
    results_file = output_dir / f"{study_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print best parameters
    logger.info(f"\nBest parameters found:")
    logger.info(f"  gd_lr: {best_params['gd_lr']:.2e}")
    logger.info(f"  gd_epochs: {best_params['gd_epochs']}")
    logger.info(f"  gd_init_scale: {best_params['gd_init_scale']:.2e}")
    logger.info(f"  gd_optimizer: {best_params['gd_optimizer']}")
    logger.info(f"  gd_scheduler: {best_params.get('gd_scheduler', 'none')}")
    
    # Print scheduler-specific parameters if they exist
    if 'step_size' in best_params:
        logger.info(f"  step_size: {best_params['step_size']}")
    if 'step_gamma' in best_params:
        logger.info(f"  step_gamma: {best_params['step_gamma']:.4f}")
    if 'exp_gamma' in best_params:
        logger.info(f"  exp_gamma: {best_params['exp_gamma']:.4f}")
    if 'cosine_eta_min' in best_params:
        logger.info(f"  cosine_eta_min: {best_params['cosine_eta_min']:.2e}")
    
    logger.info(f"Best value: {best_value:.6f}")


def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="GD Hyperparameter Optimization")
    
    # Hyperopt specific arguments
    parser.add_argument("--n_trials", type=int, default=20, 
                       help="Number of trials for optimization")
    parser.add_argument("--study_name", type=str, default="gd_hyperopt",
                       help="Name of the Optuna study")
    parser.add_argument("--output_dir", type=pathlib.Path, 
                       default=pathlib.Path("./hyperopt_results"),
                       help="Directory to save results")
    parser.add_argument("--max_gpus", type=int, default=4,
                       help="Maximum number of GPUs to use")

    
    # Parse arguments
    args = parser.parse_args()
    
    # Get base arguments from your existing parser
    base_args = parse_args()
    
    log_file = setup_logging(args.output_dir, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Get available GPUs
    available_gpus = get_available_gpus(max_gpus=args.max_gpus)
    if len(available_gpus) > 0:
        logger.info(f"Using GPUs: {available_gpus}")
        print(f"Using GPUs: {available_gpus}")
    else:
        if get_available_device() == torch.device("mps"):
            available_gpus = ["mps"]
            logger.info(f"Using MPS")
            print(f"Using MPS")
        else:
            logger.error("No available GPUs found. Exiting program.")
            print("No available GPUs found. Exiting program.")
            return

    # Determine number of processes (1-4 based on available GPUs)
    num_processes = min(len(available_gpus), 4)
    if num_processes == 0:
        logger.error("No GPUs available for processing. Exiting program.")
        print("No GPUs available for processing. Exiting program.")
        return
    
    logger.info(f"Starting {num_processes} processes on {len(available_gpus)} GPUs")
    print(f"Starting {num_processes} processes on {len(available_gpus)} GPUs")

    # Distribute trials across processes
    trials_per_process = args.n_trials // num_processes
    remaining_trials = args.n_trials % num_processes
    
    trial_ranges = []
    start_trial = 0
    for i in range(num_processes):
        end_trial = start_trial + trials_per_process + (1 if i < remaining_trials else 0)
        trial_ranges.append((start_trial, end_trial))
        start_trial = end_trial

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create queue for communication
    results_queue = Queue()
    
    # Create and start processes
    processes = []
    for i in range(num_processes):
        gpu_id = available_gpus[i % len(available_gpus)]
        trial_range = trial_ranges[i]
        
        process = Process(
            target=hyperopt_worker,
            args=(i, gpu_id, trial_range, base_args, args.study_name, results_queue, log_file)
        )
        processes.append(process)
        process.start()
    
    # Monitor progress and collect results
    completed_trials = 0
    total_trials = args.n_trials
    completed_processes = 0
    
    # Track process completion
    process_completion = {i: False for i in range(num_processes)}
    
    print(f"Starting hyperparameter optimization with {args.n_trials} trials")
    print(f"Study name: {args.study_name}")
    
    with tqdm.tqdm(total=total_trials, desc="Processing trials") as pbar:
        while completed_processes < num_processes:
            # Check for results (non-blocking)
            try:
                result = results_queue.get_nowait()
                
                if result['type'] == 'completion':
                    process_completion[result['process_id']] = True
                    completed_processes += 1
                    logger.info(f"Process {result['process_id']} completed")
                elif result['type'] == 'progress':
                    completed_trials += 1
                    pbar.update(1)
                    if completed_trials % 10 == 0:
                        logger.info(f"Completed {completed_trials}/{total_trials} trials")
                elif result['type'] == 'error':
                    logger.error(f"Error in trial {result['trial_number']}: {result['error']}")
                    pbar.update(1)
                    
            except:
                time.sleep(1)  # Sleep when no results available
    
    # Wait for all processes to finish
    for process in processes:
        process.join()
    
    # Load the final study to get best results
    study = create_study(args.study_name, "minimize")
    
    # Save results
    save_results(study, args.output_dir, args.study_name)
    
    logger.info("Hyperparameter optimization completed!")
    print("Hyperparameter optimization completed!")


if __name__ == "__main__":
    main()
