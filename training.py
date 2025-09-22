import math
import logging
import time

from generator import generate_students
from losses import get_losses, get_losses_original
from model import DiagonalSSM

import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR


def train_gd(
        student_dim: int,
        device: torch.device,
        alpha_teacher: float,
        w_sequences: list,
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
    
    # Convert w_sequences to tensor if it's a list
    if isinstance(w_sequences, list):
        w_sequences = [w.to(device) for w in w_sequences]

    for epoch in range(epochs):
        optimizer.zero_grad()

        # forward pass on all sequences
        train_loss, gen_loss = model(w_sequences, alpha_teacher)
        
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
    logging.info(f"largest 10 A_j values: {model.A_diag.topk(10).values}")
    # i want to see if there is a group of large value and a group of near zero values or if they are all over the place
    # number of values below 0.01
    logging.info(f"number of values below 0.01: {model.A_diag.lt(0.01).sum().item()}")
    # number of values between 0.01 and 0.1 (inclusive of 0.01, exclusive of 0.1)
    logging.info(f"number of values between 0.01 and 0.1: {((model.A_diag >= 0.01) & (model.A_diag < 0.1)).sum().item()}")
    # number of values between 0.1 and 0.3 (inclusive of 0.1, exclusive of 0.3)
    logging.info(f"number of values between 0.1 and 0.3: {((model.A_diag >= 0.1) & (model.A_diag < 0.3)).sum().item()}")
    # number of values larger than or equal to 0.3
    logging.info(f"number of values larger than 0.3: {model.A_diag.ge(0.3).sum().item()}")
    logging.info(f"train loss is {train_hist[-1]}")
    logging.info(f"impulse response loss is {test_hist[-1]}")
    return test_hist[-1] if test_hist else float("nan"), train_hist[-1] if train_hist else float("nan")


def train_gnc(
             seed: int,
             student_dim: int,
             device: torch.device,
             alpha_teacher: float,
             w_sequences: list,
             eps_train: float,
             num_samples: int,
             batch_size: int,
             collect_training_losses: bool = False,
              ):
    """
    Optimized GNC training to reduce CPU overhead and improve GPU utilization.
    Now includes fast prediction to skip expensive loss calculations for students
    likely to have train_loss > epsilon.
    """
    # Accumulate losses on device to avoid per-sample CPU transfers
    # Use more efficient accumulation
    prior_gen_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    prior_count = 0
    succ_gen_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    succ_count = 0
    total_students = 0
    
    # Collect training losses for histogram if requested
    training_losses = [] if collect_training_losses else None
    # save losses of the successful students
    succ_training_losses = []
    succ_gen_losses = []

    # Convert w_sequences to tensor if it's a list
    if isinstance(w_sequences, list):
        w_sequences = [w.to(device) for w in w_sequences]
    
    for batch in range(math.ceil(num_samples / batch_size)):
        bs = min(batch_size, num_samples - batch * batch_size)
        students = generate_students(student_dim, bs, device)
        train_loss, gen_loss = get_losses(students, w_sequences, alpha_teacher)
        succ_mask = train_loss < eps_train
        
        # Collect training losses for histogram if requested
        if collect_training_losses and training_losses is not None:
            # Only collect finite training losses
            finite_train_mask = torch.isfinite(train_loss)
            if finite_train_mask.any():
                training_losses.extend(train_loss[finite_train_mask].cpu().tolist())

        # Update accumulators on device - optimized version
        # Only count finite values to avoid inf/nan issues
        finite_mask = torch.isfinite(gen_loss)
        if finite_mask.any():
            prior_gen_sum += gen_loss[finite_mask].sum()
            prior_count += finite_mask.sum().item()

        succ_mask = succ_mask.squeeze(-1)
        if succ_mask.any():
            # Only count finite values for successful students
            succ_finite_mask = succ_mask & finite_mask
            if succ_finite_mask.any():
                succ_gen_sum += gen_loss[succ_finite_mask].sum()
                succ_count += succ_finite_mask.sum().item()
                succ_training_losses.extend(train_loss[succ_finite_mask].cpu().tolist())
                succ_gen_losses.extend(gen_loss[succ_finite_mask].cpu().tolist())

        # Update statistics
        total_students += bs

        # Only clear cache every few batches to reduce overhead
        if batch % 10 == 0:
            torch.cuda.empty_cache()

    mean_prior = (prior_gen_sum / max(1, prior_count)).item()
    mean_gnc = (succ_gen_sum / succ_count).item() if succ_count > 0 else float("nan")
    if succ_count == 0:
        logging.warning(f"No GNC sensing losses for student dimension {student_dim} seed {seed}")
    
    # Log success count
    logging.info(f"GNC Total success count: {succ_count}")
    if succ_count > 0:
        logging.info(f"GNC Success training losses: {succ_training_losses}")
        logging.info(f"GNC Success gen losses: {succ_gen_losses}")
    
    if collect_training_losses:
        return mean_prior, mean_gnc, training_losses
    else:
        return mean_prior, mean_gnc


def train_gnc_original(
             seed: int,
             student_dim: int,
             device: torch.device,
             alpha_teacher: float,
             w_sequences: list,
             eps_train: float,
             num_samples: int,
             batch_size: int,
              ):
    """
    Original GNC training implementation for comparison.
    """
    # Accumulate losses on device to avoid per-sample CPU transfers
    prior_gen_sum = torch.tensor(0.0, device=device)
    prior_count = 0
    succ_gen_sum = torch.tensor(0.0, device=device)
    succ_count = 0
    
    total_students = 0

    # Convert w_sequences to tensor if it's a list
    if isinstance(w_sequences, list):
        w_sequences = [w.to(device) for w in w_sequences]
    
    for batch in range(math.ceil(num_samples / batch_size)):
        bs = min(batch_size, num_samples - batch * batch_size)
        students = generate_students(student_dim, bs, device)
        train_loss, gen_loss = get_losses_original(students, w_sequences, alpha_teacher)
        
        succ_mask = train_loss < eps_train

        # Update accumulators on device
        # Only count finite values to avoid inf/nan issues
        finite_mask = torch.isfinite(gen_loss)
        if finite_mask.any():
            prior_gen_sum = prior_gen_sum + gen_loss[finite_mask].sum()
            prior_count += finite_mask.sum().item()

        succ_mask = succ_mask.squeeze(-1)
        if succ_mask.any():
            # Only count finite values for successful students
            succ_finite_mask = succ_mask & torch.isfinite(gen_loss)
            if succ_finite_mask.any():
                succ_gen_sum = succ_gen_sum + gen_loss[succ_finite_mask].sum()
                succ_count += succ_finite_mask.sum().item()
        
        # Update statistics
        total_students += bs
        torch.cuda.empty_cache()

    mean_prior = (prior_gen_sum / max(1, prior_count)).item()
    mean_gnc = (succ_gen_sum / succ_count).item() if succ_count > 0 else float("nan")
    if succ_count == 0:
        logging.warning(f"No GNC sensing losses for student dimension {student_dim} seed {seed}")
    
    # Log success count
    logging.info(f"GNC Total success count: {succ_count}")
    
    return mean_prior, mean_gnc