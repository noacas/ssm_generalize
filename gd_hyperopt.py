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
from typing import Dict, Any, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from training import train_gd
import torch
from generator import generate_w

# Import your existing parser to reuse the argument structure
from parser import parse_args

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDHyperoptObjective:
    """Objective function for GD hyperparameter optimization."""
    
    def __init__(self, base_args: argparse.Namespace, metric: str = "loss"):
        """
        Initialize the objective function.
        
        Args:
            base_args: Base arguments from parser
            metric: Metric to optimize ('loss', 'accuracy', 'convergence_time')
        """
        self.base_args = base_args
        self.metric = metric
        self.best_score = float('inf') if metric == "loss" else float('-inf')
        
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
        gd_epochs = trial.suggest_int("gd_epochs", 10000, 50000, log=True)
        gd_init_scale = trial.suggest_float("gd_init_scale", 1e-4, 1e-1, log=True)
        gd_optimizer = trial.suggest_categorical("gd_optimizer", ["adam", "gd"])
        gd_init_type = trial.suggest_categorical("gd_init_type", ["regular", "near_one", "double_max_A_j"])
        
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
        logger.info(f"Trial {trial.number}: lr={args.gd_lr:.2e}, epochs={args.gd_epochs}, "
                   f"init_scale={args.gd_init_scale:.2e}, optimizer={args.gd_optimizer}{scheduler_info}")
        
        student_dim = 150
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_teacher = 0.5
        torch.manual_seed(0)
        w = generate_w(sequence_length=5, device=device)
        gd_gen_loss, _ = train_gd(student_dim, device, alpha_teacher, w,
                                  args.gd_init_scale, args.gd_lr,
                                  args.gd_epochs,
                                  args.gd_optimizer,
                                  args.gd_scheduler,
                                  args.scheduler_params,
                                  args.gd_init_type)

        return gd_gen_loss


def create_study(study_name: str, storage: Optional[str] = None, 
                direction: str = "minimize") -> optuna.Study:
    """
    Create or load an Optuna study.
    
    Args:
        study_name: Name of the study
        storage: Database URL for persistent storage (optional)
        direction: Optimization direction ("minimize" or "maximize")
        
    Returns:
        Optuna study object
    """
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
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
    parser.add_argument("--n_trials", type=int, default=100, 
                       help="Number of trials for optimization")
    parser.add_argument("--study_name", type=str, default="gd_hyperopt",
                       help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None,
                       help="Database URL for persistent storage (e.g., sqlite:///study.db)")
    parser.add_argument("--metric", type=str, default="loss",
                       choices=["loss", "accuracy", "convergence_time"],
                       help="Metric to optimize")
    parser.add_argument("--output_dir", type=pathlib.Path, 
                       default=pathlib.Path("./hyperopt_results"),
                       help="Directory to save results")

    
    # Parse arguments
    args = parser.parse_args()
    
    # Get base arguments from your existing parser
    base_args = parse_args()
    
    # Create objective function
    objective = GDHyperoptObjective(base_args, metric=args.metric)
    
    # Create study
    direction = "minimize" if args.metric == "loss" else "maximize"
    study = create_study(args.study_name, args.storage, direction)
    
    logger.info(f"Starting hyperparameter optimization with {args.n_trials} trials")
    logger.info(f"Optimizing metric: {args.metric}")
    logger.info(f"Study name: {args.study_name}")
    
    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)
    
    # Save results
    save_results(study, args.output_dir, args.study_name)
    
    logger.info("Hyperparameter optimization completed!")


if __name__ == "__main__":
    main()
