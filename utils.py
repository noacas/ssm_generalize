from datetime import datetime
from typing import Tuple
import logging

import numpy as np
import torch
import GPUtil


def setup_logging(log_dir):
    # save logs to file
    logging.basicConfig(
        filename=log_dir / f'logs_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log',
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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


def filename_extensions(sequence_length: int,
                        num_measurements: int,
                        ) -> str:
    ext = f'_seq_len={sequence_length}_num_measurements={num_measurements}_time={datetime.now().strftime("%Y%m%d-%H%M%S")}'
    return ext


def median_iqr(mat: np.ndarray) -> Tuple[float, float]:
    """Return median, lower error, upper error along axis‑1."""
    med = np.nanmedian(mat, axis=1)  # ignores NaN values
    q1 = np.nanpercentile(mat, 25, axis=1)
    q3 = np.nanpercentile(mat, 75, axis=1)
    yerr = np.vstack([med - q1, q3 - med])
    return med, yerr