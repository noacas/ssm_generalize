import math
import logging

import torch


def generate_dataset(num_measurements: int,
                     sequence_length: int,
                     input_e1: bool,
                     device: torch.device):
    """
    Generates random data for training. Last measurement is an impulse input for generalized sensing.
    """
    impulse_response_input = torch.zeros(1, sequence_length, 1)
    impulse_response_input[0, 0, 0] = 1.0

    if input_e1:
        return impulse_response_input.to(device)
    else:
        dataset = torch.normal(mean=0, std=1, size=(num_measurements, sequence_length, 1))
        logging.info(f"dataset is {dataset}")
        # Add impulse input to dataset as the last measurement (for generalized sensing)
        
        x = torch.cat((dataset, impulse_response_input), dim=0).to(device)
        return x


def generate_teacher(teacher_rank: int,
                     student_dim: int,
                     device: torch.device):
    """
    Generates the teacher matrices A, B, and C for the given rank and dimensions.
    """
    teacher_dim = teacher_rank if teacher_rank > 1 else 2
    A_teacher = torch.zeros(teacher_dim, device=device)
    A_teacher[:teacher_rank] = torch.normal(mean=0.5, std=0.1, size=(teacher_rank,))
    logging.info(f"A_teacher is {A_teacher[0].item()}")

    B_teacher = torch.zeros(1, teacher_dim, device=device)
    B_teacher[0, 0] = 1  # First entry is 1
    B_teacher[0, 1:] = math.sqrt((student_dim - 1) / (teacher_dim - 1))  # All other entries get the calculated value

    C_teacher = torch.zeros(teacher_dim, 1, device=device)
    C_teacher[0, 0] = 1
    C_teacher[1:, 0] = math.sqrt((student_dim - 1) / (teacher_dim - 1))  # All other entries get the calculated value

    return A_teacher, B_teacher, C_teacher


def generate_students(student_dim: int, batch_size: int, sequence_length: int,
                      device: torch.device):
    """
    Generates a random student matrix A.
    """
    std = 1/student_dim**(1/2)
    A_diag = torch.normal(mean=0, std=std, size=(batch_size, student_dim), device=device)
    #A_student = torch.diag_embed(A_values).to(device)

    B_student = torch.ones(batch_size, 1, student_dim, device=device)
    C_student = torch.ones(batch_size, student_dim, 1, device=device)
    return A_diag, B_student, C_student