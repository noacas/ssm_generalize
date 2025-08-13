import torch


def generate_w(sequence_length: int,
                     device: torch.device):
    # instead of the dataset, we generate w (the first measurement without the last time step, reversed)
    w = torch.normal(mean=0, std=1, size=sequence_length-1, device=device)
    return w


def generate_teacher_alpha(device):
    return torch.normal(mean=0.5, std=0.1, size=1, device=device)


def generate_students(student_dim: int, bs: int, device: torch.device):
    return torch.normal(mean=0, std=1, size=(bs, student_dim), device=device)