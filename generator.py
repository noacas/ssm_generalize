import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from theoretical_loss import w_that_minimizes_loss


def generate_w(sequence_length: int,
                     device: torch.device):
    # instead of the dataset, we generate w (the first measurement without the last time step, reversed)
    w = torch.normal(mean=0, std=1, size=(sequence_length-1,), device=device)
    return w


def generate_w_sequences(sequence_length: int, num_sequences: int, device: torch.device, args_dict=None, alpha_teacher=None):
    if args_dict is None:
        args_dict = {}
    # Generate multiple sequences
    if num_sequences == 1:
        w = generate_w(sequence_length, device)
        if args_dict.get('w_that_minimizes_loss'):
            w = w_that_minimizes_loss(w, alpha_teacher, sequence_length)
        return [w]

    elif num_sequences == 2:
        w1 = generate_w(sequence_length, device)
        # if args_dict.get('w2_that_minimizes_loss'):
        #     w2 = w2_that_minimizes_loss(w1, alpha_teacher, sequence_length, device)
        # elif args_dict.get('w2_that_maximizes_loss'):
        #     w2 = w2_that_maximizes_loss(w1, alpha_teacher, sequence_length, device)
        # else:
        #     w2 = generate_w(sequence_length, device)
        w2 = generate_w(sequence_length, device)
        return [w1, w2]

    w_sequences = []
    for seq_idx in range(num_sequences):
        w = generate_w(sequence_length, device)
        w_sequences.append(w)
    return w_sequences


def generate_teacher_alpha(device):
    return torch.normal(mean=0.5, std=0.1, size=(1,), device=device)


def generate_students(student_dim: int, bs: int, device: torch.device):
    # Optimized version for GPU performance
    # Pre-calculate the standard deviation
    std = 1.0 / (student_dim ** 0.5)
    return torch.normal(mean=0, std=std, size=(bs, student_dim), device=device)


def load_alpha_w_pairs_from_file(data_file: Path, device: torch.device) -> List[tuple[torch.Tensor, List[torch.Tensor]]]:
    """Load alpha-W pairs from a file.
    
    Expected file formats:
    - JSON: [{"alpha": value, "w_sequences": [[seq1], [seq2], ...]}, ...]
    - JSON: [{"alpha": value, "w": [seq1]}, ...] (single sequence per pair)
    
    Args:
        data_file: Path to the file containing alpha-W pairs
        device: PyTorch device to place the tensors on
        
    Returns:
        List of (alpha_tensor, w_sequences_list) tuples
    """
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    if data_file.suffix.lower() == '.json':
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Expected list of pairs in {data_file}")
        
        pairs = []
        for i, pair_data in enumerate(data):
            if not isinstance(pair_data, dict):
                raise ValueError(f"Expected dict for pair {i} in {data_file}")
            
            if 'alpha' not in pair_data:
                raise ValueError(f"Missing 'alpha' key in pair {i} in {data_file}")
            
            alpha_value = pair_data['alpha']
            alpha_tensor = torch.tensor([alpha_value], dtype=torch.float32, device=device)
            
            # Handle both 'w_sequences' and 'w' keys
            if 'w_sequences' in pair_data:
                w_sequences_data = pair_data['w_sequences']
                if not isinstance(w_sequences_data, list):
                    raise ValueError(f"'w_sequences' must be a list in pair {i} in {data_file}")
                w_sequences = []
                for seq_data in w_sequences_data:
                    seq_tensor = torch.tensor(seq_data, dtype=torch.float32, device=device)
                    w_sequences.append(seq_tensor)
            elif 'w' in pair_data:
                # Single sequence case
                w_data = pair_data['w']
                w_tensor = torch.tensor(w_data, dtype=torch.float32, device=device)
                w_sequences = [w_tensor]
            else:
                raise ValueError(f"Missing 'w_sequences' or 'w' key in pair {i} in {data_file}")
            
            pairs.append((alpha_tensor, w_sequences))
        
        return pairs
        
    else:
        raise ValueError(f"Unsupported file format: {data_file.suffix}. Supported formats: .json")


def get_alpha_w_pair(data_file: Optional[Path], device: torch.device, seed: Optional[int] = None, 
                     pair_index: Optional[int] = None) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """Get alpha-W pair either from file or generate from seed.
    
    Args:
        data_file: Optional path to data file containing alpha-W pairs
        device: PyTorch device
        seed: Optional seed for random generation (ignored if data_file is provided)
        pair_index: Optional index to select specific pair from file (default: use seed % num_pairs)
        
    Returns:
        Tuple of (alpha_tensor, w_sequences_list)
    """
    if data_file is not None:
        pairs = load_alpha_w_pairs_from_file(data_file, device)
        
        if not pairs:
            raise ValueError(f"No alpha-W pairs found in {data_file}")
        
        # Select pair based on seed or pair_index
        if pair_index is not None:
            if pair_index >= len(pairs):
                raise ValueError(f"Pair index {pair_index} out of range (0-{len(pairs)-1})")
            selected_pair = pairs[pair_index]
        elif seed is not None:
            selected_pair = pairs[seed % len(pairs)]
        else:
            selected_pair = pairs[0]  # Use first pair if no seed specified
        
        return selected_pair
    else:
        if seed is not None:
            torch.manual_seed(seed)
        alpha_teacher = generate_teacher_alpha(device)
        # For seed-based generation, we need to know sequence_length and num_sequences
        # This will be handled by the calling code
        return alpha_teacher, None  # w_sequences will be generated separately