#!/usr/bin/env python3
"""
Example integration of hyperparameter optimization with actual training code.

This file shows how to modify the GDHyperoptObjective._run_experiment method
to work with your actual training pipeline.
"""

import subprocess
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

# Example of how to integrate with your actual training script
def run_training_with_args(args_dict: Dict[str, Any]) -> float:
    """
    Run your actual training script with given arguments.
    
    This is an example - replace with your actual training pipeline.
    
    Args:
        args_dict: Dictionary of arguments to pass to training
        
    Returns:
        Final loss/score from training
    """
    # Method 1: Run as subprocess (if your training is a separate script)
    try:
        # Convert args to command line arguments
        cmd = [sys.executable, "your_training_script.py"]
        
        for key, value in args_dict.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
                else:
                    cmd.append(f"--no-{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Run the training script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return float('inf')  # Return high loss for failed runs
        
        # Parse the output to extract final loss
        # You'll need to modify this based on your training script's output format
        lines = result.stdout.split('\n')
        for line in lines:
            if "Final loss:" in line:
                loss = float(line.split("Final loss:")[1].strip())
                return loss
        
        # If no loss found, return a default high value
        return float('inf')
        
    except subprocess.TimeoutExpired:
        print("Training timed out")
        return float('inf')
    except Exception as e:
        print(f"Error running training: {e}")
        return float('inf')


def run_training_direct(args_dict: Dict[str, Any]) -> float:
    """
    Alternative: Run training directly by importing your training module.
    
    This is more efficient than subprocess but requires your training code
    to be modular.
    
    Args:
        args_dict: Dictionary of arguments to pass to training
        
    Returns:
        Final loss/score from training
    """
    # Example - replace with your actual training function
    try:
        # Import your training module
        # from your_training_module import train_model
        
        # Create args object
        # args = create_args_from_dict(args_dict)
        
        # Run training
        # final_loss = train_model(args)
        
        # For now, return a mock value
        return 0.1  # Replace with actual training result
        
    except Exception as e:
        print(f"Error in direct training: {e}")
        return float('inf')


# Modified GDHyperoptObjective for integration
class IntegratedGDHyperoptObjective:
    """Objective function that integrates with actual training code."""
    
    def __init__(self, base_args, metric: str = "loss", use_subprocess: bool = True):
        self.base_args = base_args
        self.metric = metric
        self.use_subprocess = use_subprocess
        
    def _run_experiment(self, args, trial):
        """
        Run experiment with actual training code.
        
        Args:
            args: Arguments for the experiment
            trial: Optuna trial object
            
        Returns:
            Score from training
        """
        # Convert args to dictionary
        args_dict = vars(args)
        
        # Run training
        if self.use_subprocess:
            score = run_training_with_args(args_dict)
        else:
            score = run_training_direct(args_dict)
        
        return score


# Example usage
if __name__ == "__main__":
    print("This is an example integration file.")
    print("To use it:")
    print("1. Replace the mock training functions with your actual training code")
    print("2. Modify gd_hyperopt.py to use IntegratedGDHyperoptObjective")
    print("3. Run: python gd_hyperopt.py --n_trials 50")
