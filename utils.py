from datetime import datetime
from typing import Tuple
import logging
import os

import numpy as np
import torch
import GPUtil


def setup_logging(log_dir, timestamp):
    # is log dir maybe a log path?
    if os.path.isfile(log_dir):
        log_filename = log_dir
    else:
        # Configure logging to both file and console
        log_filename = log_dir / f'logs_{timestamp}.log'
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_filename


def get_available_device(max_load: float = 0.3,
                         max_memory: float = 0.3) -> torch.device:
# Try to get a GPU that is under the load and memory thresholds.
    available_gpus = GPUtil.getAvailable(order='first', maxLoad=max_load, maxMemory=max_memory, limit=torch.cuda.device_count())
    if available_gpus:
        return torch.device(f'cuda:{available_gpus[0]}')
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device('cpu')


def get_available_gpus(max_load: float = 0.3,
                       max_memory: float = 0.3,
                       max_gpus: int = None) -> list:
    """
    Get multiple available GPUs that are under the load and memory thresholds.
    
    Args:
        max_load: Maximum GPU load threshold (0.0 to 1.0)
        max_memory: Maximum GPU memory usage threshold (0.0 to 1.0)
        max_gpus: Maximum number of GPUs to return (None for all available)
    
    Returns:
        List of available GPU IDs
    """
    available_gpus = GPUtil.getAvailable(order='first', maxLoad=max_load, maxMemory=max_memory, limit=torch.cuda.device_count())
    
    if max_gpus is not None:
        available_gpus = available_gpus[:max_gpus]
    
    return available_gpus


def filename_extensions(args, timestamp) -> str:
    if args.gnc and not args.gd:
        ext = f'_gnc_seq_len={args.sequence_length}_time={timestamp}'
    elif not args.gnc and args.gd:
        ext = f'_gd_seq_len={args.sequence_length}_time={timestamp}'
    else:
        ext = f'_seq_len={args.sequence_length}_time={timestamp}'
    return ext


def median_iqr(mat: np.ndarray) -> Tuple[float, float]:
    """Return median, lower error, upper error along axis‑1."""
    med = np.nanmedian(mat, axis=1)  # ignores NaN values
    q1 = np.nanpercentile(mat, 25, axis=1)
    q3 = np.nanpercentile(mat, 75, axis=1)
    yerr = np.vstack([med - q1, q3 - med])
    return med, yerr