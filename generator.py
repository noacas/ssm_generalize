import torch


def generate_dataset(num_measurements: int,
                     sequence_length: int,
                     device: torch.device):
    """
    Generates random data for training. Last measurement is an impulse input for generalized sensing.
    Optimized to generate directly on target device to reduce CPU-GPU transfers.
    """
    # Generate directly on target device to avoid CPU-GPU transfer
    dataset = torch.normal(mean=0, std=1, size=(num_measurements, sequence_length-1, 1), device=device)
    return dataset


def generate_teacher(device: torch.device):
    return torch.normal(mean=0.5, std=0.1, device=device)