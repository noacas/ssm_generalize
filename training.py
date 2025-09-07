import math
import logging

from generator import generate_students
from losses import get_losses
from model import DiagonalSSM
from theoretical_loss import gnc_theoretical_loss_for_multiple_w

import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR


def predict_train_loss_above_epsilon(students, w_sequences, alpha_teacher, eps_train, device):
    """
    Fast prediction to determine if training loss is likely above epsilon.
    Focuses on identifying students with parameters too small to fit training data well.
    
    Args:
        students: Tensor of shape (batch_size, student_dim) with student parameters
        w_sequences: List of w sequences for training
        alpha_teacher: Teacher parameter
        eps_train: Epsilon threshold for training loss
        device: Device to run computations on
        
    Returns:
        mask: Boolean tensor of shape (batch_size,) indicating which students
              are likely to have train_loss > eps_train
    """
    batch_size, student_dim = students.shape
    
    # Vectorized computation for speed
    with torch.no_grad():
        # Calculate statistics for each student
        student_norms = torch.norm(students, dim=1)  # L2 norm
        student_mean_abs = torch.mean(torch.abs(students), dim=1)  # Mean absolute value
        student_max_abs = torch.max(torch.abs(students), dim=1)[0]  # Max absolute value
        
        # Primary heuristic: Students with parameters too small to fit the training data
        # Based on the fact that students need sufficient magnitude to match teacher responses
        
        # For a student to fit well, they need parameters that can generate responses
        # comparable to the teacher's alpha^m responses. Since students are initialized
        # with small random values, most failures come from parameters being too small.
        
        # Adaptive thresholds based on student dimension and number of sequences
        # Students need enough "capacity" to approximate the teacher's behavior
        # With more sequences, students need more capacity to fit multiple constraints
        num_sequences = len(w_sequences)
        sequence_factor = 1.0 + 0.5 * (num_sequences - 1)  # Scale with number of sequences
        
        min_norm_threshold = (0.1 + 0.002 * student_dim) * sequence_factor  # Minimum norm needed
        min_mean_threshold = (0.01 + 0.0002 * student_dim) * sequence_factor  # Minimum mean magnitude
        min_max_threshold = 0.02 * sequence_factor  # At least one parameter should be reasonably large
        
        # Students that are too small are likely to fail
        too_small_mask = (
            (student_norms < min_norm_threshold) |
            (student_mean_abs < min_mean_threshold) |
            (student_max_abs < min_max_threshold)
        )
        
        # Secondary heuristic: Students with very high variance (unstable parameters)
        # These might oscillate and fail to converge
        student_std = torch.std(students, dim=1)
        high_variance_mask = student_std > 1.0  # High variance threshold
        
        # Combine heuristics: too small OR high variance
        prediction_mask = too_small_mask | high_variance_mask
        
        # Additional check: if most parameters are extremely close to zero
        # This catches the most obvious cases of under-parameterized students
        near_zero_ratio = (torch.abs(students) < 0.001).float().mean(dim=1)
        mostly_zero_mask = near_zero_ratio > 0.95  # 95% of parameters near zero
        
        prediction_mask = prediction_mask | mostly_zero_mask
    
    return prediction_mask


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
             use_prediction: bool = True,
              ):
    """
    Optimized GNC training to reduce CPU overhead and improve GPU utilization.
    Now includes fast prediction to skip expensive loss calculations for students
    likely to have train_loss > epsilon.
    """
    # Accumulate losses on device to avoid per-sample CPU transfers
    prior_gen_sum = torch.tensor(0.0, device=device)
    prior_count = 0
    succ_gen_sum = torch.tensor(0.0, device=device)
    succ_count = 0
    
    # Statistics for prediction effectiveness
    total_students = 0
    predicted_above_epsilon = 0
    actual_above_epsilon = 0
    skipped_calculations = 0

    # Convert w_sequences to tensor if it's a list
    if isinstance(w_sequences, list):
        w_sequences = [w.to(device) for w in w_sequences]
    
    for batch in range(math.ceil(num_samples / batch_size)):
        bs = min(batch_size, num_samples - batch * batch_size)
        students = generate_students(student_dim, bs, device)
        
        if use_prediction:
            # Use fast prediction to identify students likely to have train_loss > epsilon
            predicted_above_mask = predict_train_loss_above_epsilon(
                students, w_sequences, alpha_teacher, eps_train, device
            )
            
            # Only calculate exact losses for students not predicted to be above epsilon
            candidates_mask = ~predicted_above_mask
            
            if candidates_mask.any():
                # Calculate losses only for promising candidates
                candidate_students = students[candidates_mask]
                train_loss_candidates, gen_loss_candidates = get_losses(
                    candidate_students, w_sequences, alpha_teacher
                )
                
                # Create full-size tensors for the results
                # Use a large value instead of inf to avoid numerical issues
                large_value = 10.0 * eps_train  # Large but finite value
                train_loss = torch.full((bs,), large_value, device=device)
                gen_loss = torch.full((bs,), large_value, device=device)
                
                # Fill in the calculated values for candidates
                train_loss[candidates_mask] = train_loss_candidates
                gen_loss[candidates_mask] = gen_loss_candidates
                
                # For predicted above-epsilon students, we skip the calculation
                # and assume they have train_loss > eps_train
                skipped_calculations += predicted_above_mask.sum().item()
            else:
                # All students were predicted to be above epsilon
                large_value = 10.0 * eps_train  # Large but finite value
                train_loss = torch.full((bs,), large_value, device=device)
                gen_loss = torch.full((bs,), large_value, device=device)
                skipped_calculations += bs
        else:
            # Original behavior: calculate losses for all students
            train_loss, gen_loss = get_losses(students, w_sequences, alpha_teacher)
        
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
        if use_prediction:
            predicted_above_epsilon += predicted_above_mask.sum().item()
            actual_above_epsilon += (train_loss >= eps_train).sum().item()

        torch.cuda.empty_cache()

    mean_prior = (prior_gen_sum / max(1, prior_count)).item()
    mean_gnc = (succ_gen_sum / succ_count).item() if succ_count > 0 else float("nan")
    if succ_count == 0:
        logging.warning(f"No GNC sensing losses for student dimension {student_dim} seed {seed}")
    
    # Log prediction statistics if prediction was used
    if use_prediction and total_students > 0:
        prediction_accuracy = (predicted_above_epsilon / total_students) * 100
        skip_rate = (skipped_calculations / total_students) * 100
        logging.info(f"GNC Prediction stats - Total: {total_students}, "
                    f"Predicted above epsilon: {predicted_above_epsilon} ({prediction_accuracy:.1f}%), "
                    f"Skipped calculations: {skipped_calculations} ({skip_rate:.1f}%)")
    
    return mean_prior, mean_gnc