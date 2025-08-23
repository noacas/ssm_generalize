import math
import logging

from generator import generate_students
from losses import get_losses
from model import DiagonalSSM

import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR


def train_gd(
        student_dim: int,
        device: torch.device,
        alpha_teacher: float,
        w: torch.Tensor,
        init_scale: float,
        lr: float,
        epochs: int,
        optimizer: str = "adam",
        scheduler: str = None,
        scheduler_params: dict = None,
        init_type: str = "regular",
    ):
    """
    Trains only the diagonal transition matrix A of the student SSM.
    B and C stay fixed to 1s as defined inside DiagonalSSM.
    Optimized to reduce CPU overhead and improve GPU utilization.
    
    Args:
        scheduler: Type of scheduler ('step', 'exponential', 'cosine', None)
        scheduler_params: Dictionary of scheduler parameters
    """

    # --- build student -------------------------------------------------------
    model = DiagonalSSM(state_dim=student_dim,
                        init_scale=init_scale,
                        init_type=init_type
                        ).to(device)

    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer == "gd":
        optimizer = SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")

    # Initialize scheduler
    scheduler_obj = None
    if scheduler:
        if scheduler == "step":
            step_size = scheduler_params.get("step_size", epochs // 4)
            gamma = scheduler_params.get("gamma", 0.1)
            scheduler_obj = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler == "exponential":
            gamma = scheduler_params.get("gamma", 0.95)
            scheduler_obj = ExponentialLR(optimizer, gamma=gamma)
        elif scheduler == "cosine":
            T_max = scheduler_params.get("T_max", epochs)
            eta_min = scheduler_params.get("eta_min", 0)
            scheduler_obj = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler}")

    # Reduced logging to minimize CPU overhead
    max_A_j_idx = torch.argmax(model.A_diag)
    logging.info(f"initial model: max A_j index: {max_A_j_idx}")

    # --- keep track of losses ------------------------------------------------
    train_hist, test_hist = [], []
    w = w.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # forward pass on the full sequence
        train_loss, gen_loss = model(w, alpha_teacher)
        
        train_loss.backward()
        optimizer.step()
        
        # Step the scheduler
        if scheduler_obj:
            scheduler_obj.step()

        with torch.no_grad():
            # save losses
            train_hist.append(train_loss.item())
            if epoch % 10 == 0:  # test every 10 epochs
                test_hist.append(gen_loss.item())

    max_A_j_idx = torch.argmax(model.A_diag)
    max_A_j = model.A_diag[max_A_j_idx]
    logging.info(f"final model: max A_j index: {max_A_j_idx}, max A_j value: {max_A_j.item()}, alpha_teacher: {alpha_teacher}")
    logging.info(f"train loss is {train_hist[-1]}")
    logging.info(f"impulse response loss is {test_hist[-1]}")
    return test_hist[-1] if test_hist else float("nan"), train_hist[-1] if train_hist else float("nan")


def train_gnc(
             seed: int,
             student_dim: int,
             device: torch.device,
             alpha_teacher: float,
             w: torch.Tensor,
             eps_train: float,
             num_samples: int,
             batch_size: int,
              ):
    """
    Optimized GNC training to reduce CPU overhead and improve GPU utilization.
    """
    # Accumulate losses on device to avoid per-sample CPU transfers
    prior_gen_sum = torch.tensor(0.0, device=device)
    prior_count = 0
    succ_gen_sum = torch.tensor(0.0, device=device)
    succ_count = 0

    for batch in range(math.ceil(num_samples / batch_size)):
        bs = min(batch_size, num_samples - batch * batch_size)
        students = generate_students(student_dim, bs, device)
        train_losses, gen_losses = get_losses(students, w, alpha_teacher)
        succ_mask = train_losses < eps_train

        # Update accumulators on device
        prior_gen_sum = prior_gen_sum + gen_losses.sum()
        prior_count += gen_losses.numel()

        succ_mask = succ_mask.squeeze(-1)
        if succ_mask.any():
            succ_gen_sum = succ_gen_sum + gen_losses[succ_mask].sum()
            succ_count += succ_mask.sum().item()

        torch.cuda.empty_cache()

    mean_prior = (prior_gen_sum / max(1, prior_count)).item()
    mean_gnc = (succ_gen_sum / succ_count).item() if succ_count > 0 else float("nan")
    if succ_count == 0:
        logging.warning(f"No GNC sensing losses for student dimension {student_dim} seed {seed}")
    return mean_prior, mean_gnc