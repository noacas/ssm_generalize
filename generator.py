import torch
from theoretical_loss import w_that_minimizes_loss, w2_that_minimizes_loss, w2_that_maximizes_loss


def generate_w(sequence_length: int,
                     device: torch.device):
    # instead of the dataset, we generate w (the first measurement without the last time step, reversed)
    w = torch.normal(mean=0, std=1, size=(sequence_length-1,), device=device)
    return w


def generate_w_sequences(sequence_length: int, num_sequences: int, device: torch.device, args_dict, alpha_teacher):
    # Generate multiple sequences
    w_sequences = []
    if num_sequences == 1:
        w = generate_w(sequence_length, device)
        if args_dict['w_that_minimizes_loss']:
            w = w_that_minimizes_loss(w, alpha_teacher, sequence_length)
        w_sequences.append(w)
    elif num_sequences == 2:
        w1 = generate_w(sequence_length, device)
        w_sequences.append(w1)
        w2 = generate_w(sequence_length, device)
        if args_dict['w2_that_minimizes_loss']:
            w2 = w2_that_minimizes_loss(w_sequences, w2, alpha_teacher, sequence_length)
        elif args_dict['w2_that_maximizes_loss']:
            w2 = w2_that_maximizes_loss(w_sequences, w2, alpha_teacher, sequence_length)
        w_sequences.append(w2)
    else:
        for seq_idx in range(num_sequences):
            w = generate_w(sequence_length, device)
            w_sequences.append(w)
    return w_sequences


def generate_teacher_alpha(device):
    return torch.normal(mean=0.5, std=0.1, size=(1,), device=device)


def generate_students(student_dim: int, bs: int, device: torch.device):
    return torch.normal(mean=0, std=1/(student_dim**0.5), size=(bs, student_dim), device=device)