# GD Hyperparameter Optimization

This directory contains scripts for optimizing the Gradient Descent (GD) hyperparameters using Optuna.

## Files

- `gd_hyperopt.py`: Main hyperparameter optimization script
- `example_integration.py`: Example showing how to integrate with actual training code
- `environment_hyperopt.yml`: Conda environment file (recommended)
- `requirements_hyperopt.txt`: Pip requirements file (alternative)
- `README_hyperopt.md`: This file

## Installation

### Option 1: Using Conda (Recommended)

1. Create and activate a conda environment:
```bash
conda env create -f environment_hyperopt.yml
conda activate gd_hyperopt
```

### Option 2: Using Pip

1. Install the required dependencies:
```bash
pip install -r requirements_hyperopt.txt
```

## Usage

### Basic Usage

Run hyperparameter optimization with default settings:
```bash
python gd_hyperopt.py
```

### Advanced Usage

Run with custom parameters:
```bash
python gd_hyperopt.py \
    --n_trials 20 \
    --study_name "my_gd_optimization" \
    --metric loss \
    --output_dir ./my_results
```

### Command Line Arguments

- `--n_trials`: Number of optimization trials (default: 100)
- `--study_name`: Name of the Optuna study (default: "gd_hyperopt")
- `--storage`: Database URL for persistent storage (optional)
- `--metric`: Metric to optimize - "loss", "accuracy", or "convergence_time" (default: "loss")
- `--output_dir`: Directory to save results (default: "./hyperopt_results")


## Hyperparameters Being Optimized

The script optimizes the following GD parameters using your complete experiment setup:

1. **gd_lr**: Learning rate (range: 1e-5 to 1e-1, log scale)
2. **gd_epochs**: Number of training epochs (range: 100 to 50000, log scale)
3. **gd_init_scale**: Initialization scale (range: 1e-4 to 1e0, log scale)
4. **gd_optimizer**: Optimizer choice (categorical: "adam" or "gd")

**Note**: Each trial runs the full experiment with all seeds and student dimensions from your base configuration, ensuring robust hyperparameter evaluation.

## Integration with Your Training Code

The hyperparameter optimization script is now fully integrated with your complete experiment pipeline from `main.py`. It uses:

- **Full experiment setup**: Runs the complete multiprocessing experiment with multiple seeds and student dimensions
- **Main.py pipeline**: Uses the same `main.py` script with temporary config files
- **Objective**: Minimize mean `gd_gen_loss` across all seeds and student dimensions
- **Robust evaluation**: Each trial runs the full experiment to get reliable performance estimates

### How It Works

1. **Complete Experiment**: For each trial, runs the full `main.py` experiment with the suggested hyperparameters
2. **Multiple Seeds**: Uses all seeds and student dimensions from your base configuration
3. **Robust Metrics**: Calculates mean and standard deviation of `gd_gen_loss` across all experiments
4. **Temporary Configs**: Creates temporary JSON config files for each trial
5. **Result Parsing**: Automatically parses CSV results to extract the mean loss

### Key Features

- **Full Pipeline**: Uses your complete multiprocessing setup from `main.py`
- **Multiple GPUs**: Leverages your existing GPU distribution logic
- **Checkpointing**: Each trial creates its own results directory
- **Timeout Protection**: 1-hour timeout per trial to prevent hanging
- **Error Handling**: Robust error handling for failed experiments

## Output

The script generates:

1. **JSON Results File**: Contains best parameters, optimization history, and trial details
2. **Console Output**: Real-time progress and final best parameters
3. **Optuna Study**: Can be loaded later for analysis or continued optimization

### Example Output

```
Best parameters found:
  gd_lr: 3.16e-03
  gd_epochs: 5000
  gd_init_scale: 1.00e-02
  gd_optimizer: adam
Best value: 0.045123
```

## Advanced Features

### Persistent Storage

Use a database to save and resume optimization:

```bash
python gd_hyperopt.py --storage sqlite:///study.db
```

### Pruning

The script uses Optuna's MedianPruner to automatically stop unpromising trials early.

### Parallel Optimization

Optuna supports parallel optimization. Run multiple instances with the same study name and storage.

## Tips

1. **Start Small**: Begin with 20-50 trials to test your setup
2. **Monitor Resources**: Large hyperparameter spaces can be computationally expensive
3. **Use Log Scale**: The script uses log scale for continuous parameters, which is often more effective
4. **Checkpoint**: Use persistent storage to resume interrupted optimizations
5. **Analyze Results**: Use Optuna's visualization tools to analyze the optimization process

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Training Failures**: Check that your training script works with the provided arguments
3. **Memory Issues**: Reduce batch sizes or use fewer trials
4. **Timeout Errors**: Increase timeout values for long-running training

### Debug Mode

Add debug logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example Workflow

1. **Setup**: Install dependencies and prepare your training code
2. **Test**: Run a few trials to ensure everything works
3. **Optimize**: Run full optimization with desired number of trials
4. **Analyze**: Review results and best parameters
5. **Validate**: Test the best parameters on a validation set
6. **Iterate**: Refine the search space if needed
