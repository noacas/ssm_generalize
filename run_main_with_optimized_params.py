#!/usr/bin/env python3
"""
Script to run main.py with optimized hyperparameters from hyperopt results.
"""

import subprocess
import sys
import json
import pathlib

def load_optimized_params(results_file="hyperopt_results/gd_optimization_full_results.json"):
    """Load the optimized parameters from the hyperopt results file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results['best_params']

def run_main_with_params(params):
    """Run main.py with the given parameters."""
    
    # Build scheduler parameters JSON
    scheduler_params = {}
    if params.get('gd_scheduler') and params['gd_scheduler'] != 'none':
        if params['gd_scheduler'] == 'exponential':
            scheduler_params['gamma'] = params['exp_gamma']
        elif params['gd_scheduler'] == 'step':
            scheduler_params['step_size'] = params.get('step_size', 1000)
            scheduler_params['gamma'] = params.get('step_gamma', 0.1)
        elif params['gd_scheduler'] == 'cosine':
            scheduler_params['T_max'] = params['gd_epochs']
            scheduler_params['eta_min'] = params.get('cosine_eta_min', 0)
    
    scheduler_params_json = json.dumps(scheduler_params)
    
    # Build the command with optimized parameters
    cmd = [
        sys.executable, "main.py",
        "--gd",  # Enable gradient descent
        "--no-gnc",  # Disable guess and check to focus on GD
        f"--gd_lr={params['gd_lr']}",
        f"--gd_epochs={params['gd_epochs']}",
        f"--gd_init_scale={params['gd_init_scale']}",
        f"--gd_optimizer={params['gd_optimizer']}",
        f"--gd_init_type={params['gd_init_type']}",
        "--num_seeds=8",  # You can adjust this
        "--sequence_length=5",
        "--student_dims", "150", "175", "200", "225", "250", "275",  # You can adjust these
        "--max_gpus=4",  # Adjust based on your system
    ]
    
    # Add scheduler parameters if a scheduler was used
    if params.get('gd_scheduler') and params['gd_scheduler'] != 'none':
        cmd.extend([
            f"--gd_scheduler={params['gd_scheduler']}",
            f"--gd_scheduler_params={scheduler_params_json}"
        ])
        print(f"Note: Using scheduler '{params['gd_scheduler']}' with parameters:")
        if params['gd_scheduler'] == 'exponential':
            print(f"  - gamma: {params['exp_gamma']}")
        elif params['gd_scheduler'] == 'step':
            print(f"  - step_size: {params.get('step_size', 'N/A')}")
            print(f"  - gamma: {params.get('step_gamma', 'N/A')}")
        elif params['gd_scheduler'] == 'cosine':
            print(f"  - eta_min: {params.get('cosine_eta_min', 'N/A')}")
        print()
    
    print("Running main.py with optimized parameters:")
    print(f"  Learning rate: {params['gd_lr']:.2e}")
    print(f"  Epochs: {params['gd_epochs']}")
    print(f"  Init scale: {params['gd_init_scale']:.4f}")
    print(f"  Optimizer: {params['gd_optimizer']}")
    print(f"  Scheduler: {params.get('gd_scheduler', 'none')}")
    print()
    print("Command:", " ".join(cmd))
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("Main script completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Main script failed with error code {e.returncode}")
        sys.exit(1)

def main():
    """Main function."""
    results_file = "hyperopt_results/gd_optimization_full_results.json"
    
    if not pathlib.Path(results_file).exists():
        print(f"Error: Results file {results_file} not found!")
        print("Please run the hyperopt script first or provide the correct path.")
        sys.exit(1)
    
    try:
        params = load_optimized_params(results_file)
        run_main_with_params(params)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
