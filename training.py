import math
import logging

from generator import generate_students
from losses import get_losses, get_losses_original
from model import DiagonalSSM
from theoretical_loss import gnc_theoretical_loss_for_multiple_w

import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR


def analyze_student_characteristics(students, train_loss, eps_train, device):
    """
    Analyze statistical characteristics of successful vs unsuccessful students.
    
    Args:
        students: Tensor of shape (batch_size, student_dim) with student parameters
        train_loss: Tensor of shape (batch_size,) with training losses
        eps_train: Epsilon threshold for training loss
        device: Device to run computations on
        
    Returns:
        dict: Statistics about successful vs unsuccessful students
    """
    with torch.no_grad():
        success_mask = train_loss < eps_train
        failure_mask = train_loss >= eps_train
        
        if success_mask.any() and failure_mask.any():
            successful_students = students[success_mask]
            failed_students = students[failure_mask]
            
            # Compute statistics for successful students
            success_mean = torch.mean(successful_students, dim=1)
            success_var = torch.var(successful_students, dim=1)
            success_norm = torch.norm(successful_students, dim=1)
            success_max_abs = torch.max(torch.abs(successful_students), dim=1)[0]
            
            # Compute statistics for failed students
            failure_mean = torch.mean(failed_students, dim=1)
            failure_var = torch.var(failed_students, dim=1)
            failure_norm = torch.norm(failed_students, dim=1)
            failure_max_abs = torch.max(torch.abs(failed_students), dim=1)[0]
            
            stats = {
                'success_count': success_mask.sum().item(),
                'failure_count': failure_mask.sum().item(),
                'success_mean_mean': success_mean.mean().item(),
                'success_mean_std': success_mean.std().item(),
                'success_var_mean': success_var.mean().item(),
                'success_var_std': success_var.std().item(),
                'success_norm_mean': success_norm.mean().item(),
                'success_norm_std': success_norm.std().item(),
                'success_max_abs_mean': success_max_abs.mean().item(),
                'success_max_abs_std': success_max_abs.std().item(),
                'failure_mean_mean': failure_mean.mean().item(),
                'failure_mean_std': failure_mean.std().item(),
                'failure_var_mean': failure_var.mean().item(),
                'failure_var_std': failure_var.std().item(),
                'failure_norm_mean': failure_norm.mean().item(),
                'failure_norm_std': failure_norm.std().item(),
                'failure_max_abs_mean': failure_max_abs.mean().item(),
                'failure_max_abs_std': failure_max_abs.std().item(),
            }
            
            return stats
        else:
            return None


def predict_train_loss_above_epsilon_data_driven(students, w_sequences, alpha_teacher, eps_train, device, 
                                                success_stats=None, failure_stats=None):
    """
    Data-driven prediction based on statistical analysis of successful vs failed students.
    
    Args:
        students: Tensor of shape (batch_size, student_dim) with student parameters
        w_sequences: List of w sequences for training
        alpha_teacher: Teacher parameter
        eps_train: Epsilon threshold for training loss
        device: Device to run computations on
        success_stats: Statistics from successful students (optional)
        failure_stats: Statistics from failed students (optional)
        
    Returns:
        mask: Boolean tensor of shape (batch_size,) indicating which students
              are likely to have train_loss > eps_train
    """
    batch_size, student_dim = students.shape
    
    with torch.no_grad():
        # Compute student statistics
        student_mean = torch.mean(students, dim=1)
        student_var = torch.var(students, dim=1)
        student_norm = torch.norm(students, dim=1)
        student_max_abs = torch.max(torch.abs(students), dim=1)[0]
        
        prediction_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if success_stats is not None and failure_stats is not None:
            # Use learned thresholds based on statistical analysis
            # Since success rate is very low (~0.3%), be extremely conservative
            
            total_samples = success_stats['success_count'] + failure_stats['failure_count']
            success_rate = success_stats['success_count'] / total_samples
            
            # Only predict failure if student is very clearly in the failure range
            # Use thresholds that are much closer to the failure statistics
            
            # Mean-based prediction - use threshold very close to failure mean
            mean_threshold = failure_stats['failure_mean_mean'] + 0.1 * (success_stats['success_mean_mean'] - failure_stats['failure_mean_mean'])
            prediction_mask |= (torch.abs(student_mean) < mean_threshold)
            
            # Variance-based prediction - use threshold very close to failure variance
            var_threshold = failure_stats['failure_var_mean'] + 0.05 * (success_stats['success_var_mean'] - failure_stats['failure_var_mean'])
            prediction_mask |= (student_var < var_threshold)
            
            # Norm-based prediction - use threshold very close to failure norm
            norm_threshold = failure_stats['failure_norm_mean'] + 0.01 * (success_stats['success_norm_mean'] - failure_stats['failure_norm_mean'])
            prediction_mask |= (student_norm < norm_threshold)
            
            # Max absolute value prediction - use threshold very close to failure max_abs
            max_abs_threshold = failure_stats['failure_max_abs_mean'] + 0.05 * (success_stats['success_max_abs_mean'] - failure_stats['failure_max_abs_mean'])
            prediction_mask |= (student_max_abs < max_abs_threshold)
            
        else:
            # Fallback to conservative heuristics if no statistics available
            prediction_mask |= (student_max_abs < 0.1)
            prediction_mask |= (student_norm < 0.5)
            prediction_mask |= (student_var < 0.01)
            prediction_mask |= (torch.abs(student_mean) < 0.01)
    
    return prediction_mask


def predict_train_loss_above_epsilon(students, w_sequences, alpha_teacher, eps_train, device):
    """
    Legacy prediction function - kept for compatibility.
    """
    batch_size, student_dim = students.shape
    
    with torch.no_grad():
        prediction_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Simple heuristics
        student_max_abs = torch.max(torch.abs(students), dim=1)[0]
        student_norm = torch.norm(students, dim=1)
        
        prediction_mask |= (student_max_abs < 0.1)
        prediction_mask |= (student_norm < 0.5)
    
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
    # Use more efficient accumulation
    prior_gen_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    prior_count = 0
    succ_gen_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
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
        
        # Update statistics
        total_students += bs
        if use_prediction:
            predicted_above_epsilon += predicted_above_mask.sum().item()
            actual_above_epsilon += (train_loss >= eps_train).sum().item()

        # Only clear cache every few batches to reduce overhead
        if batch % 10 == 0:
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


def train_gnc_sampling_based(
             seed: int,
             student_dim: int,
             device: torch.device,
             alpha_teacher: float,
             w_sequences: list,
             eps_train: float,
             num_samples: int,
             batch_size: int,
             sample_ratio: float = 0.1,  # Fraction of samples to actually compute
              ):
    """
    GNC training with sampling-based approach for speedup while maintaining accuracy.
    Computes losses for a random sample of students and scales the results.
    """
    # Accumulate losses on device
    prior_gen_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    prior_count = 0
    succ_gen_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    succ_count = 0
    
    # Convert w_sequences to tensor if it's a list
    if isinstance(w_sequences, list):
        w_sequences = [w.to(device) for w in w_sequences]
    
    # Calculate how many samples to actually compute
    actual_samples = int(num_samples * sample_ratio)
    logging.info(f"Sampling-based approach: computing {actual_samples} out of {num_samples} samples ({sample_ratio*100:.1f}%)")
    
    # Set random seed for reproducible sampling
    torch.manual_seed(seed)
    
    for batch in range(math.ceil(actual_samples / batch_size)):
        bs = min(batch_size, actual_samples - batch * batch_size)
        students = generate_students(student_dim, bs, device)
        
        # Calculate losses for sampled students
        train_loss, gen_loss = get_losses(students, w_sequences, alpha_teacher)
        
        succ_mask = train_loss < eps_train

        # Update accumulators
        finite_mask = torch.isfinite(gen_loss)
        if finite_mask.any():
            prior_gen_sum += gen_loss[finite_mask].sum()
            prior_count += finite_mask.sum().item()

        succ_mask = succ_mask.squeeze(-1)
        if succ_mask.any():
            succ_finite_mask = succ_mask & finite_mask
            if succ_finite_mask.any():
                succ_gen_sum += gen_loss[succ_finite_mask].sum()
                succ_count += succ_finite_mask.sum().item()

        # Only clear cache every few batches
        if batch % 10 == 0:
            torch.cuda.empty_cache()

    # Scale the results to account for sampling
    scale_factor = num_samples / actual_samples
    prior_gen_sum *= scale_factor
    prior_count = int(prior_count * scale_factor)
    succ_gen_sum *= scale_factor
    succ_count = int(succ_count * scale_factor)

    mean_prior = (prior_gen_sum / max(1, prior_count)).item()
    mean_gnc = (succ_gen_sum / succ_count).item() if succ_count > 0 else float("nan")
    if succ_count == 0:
        logging.warning(f"No GNC sensing losses for student dimension {student_dim} seed {seed}")
    
    logging.info(f"Sampling-based stats - Computed: {actual_samples}, Scaled to: {num_samples}, "
                f"Success rate: {succ_count/actual_samples*100:.2f}%")
    
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
             use_prediction: bool = True,
              ):
    """
    Original GNC training implementation for comparison.
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
                train_loss_candidates, gen_loss_candidates = get_losses_original(
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