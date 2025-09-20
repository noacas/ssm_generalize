import math
import logging
import time

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
    Analyze statistical characteristics of successful vs unsuccessful students
    to determine optimal threshold parameters for heuristic prediction.
    
    Args:
        students: Tensor of shape (batch_size, student_dim) with student parameters
        train_loss: Tensor of shape (batch_size,) with training losses
        eps_train: Epsilon threshold for training loss
        device: Device to run computations on
        
    Returns:
        dict: Statistics about successful vs unsuccessful students and recommended thresholds
    """
    with torch.no_grad():
        success_mask = train_loss < eps_train
        failure_mask = train_loss >= eps_train
        
        if success_mask.any() and failure_mask.any():
            successful_students = students[success_mask]
            failed_students = students[failure_mask]
            
            # Compute mean and variance for each student (key metrics for heuristic)
            success_means = torch.mean(successful_students, dim=1)  # (n_success,)
            success_vars = torch.var(successful_students, dim=1)    # (n_success,)
            failure_means = torch.mean(failed_students, dim=1)      # (n_failure,)
            failure_vars = torch.var(failed_students, dim=1)        # (n_failure,)
            
            # Compute statistics for successful students
            success_mean_stats = {
                'mean': success_means.mean().item(),
                'std': success_means.std().item(),
                'min': success_means.min().item(),
                'max': success_means.max().item(),
                'percentile_5': torch.quantile(success_means, 0.05).item(),
                'percentile_95': torch.quantile(success_means, 0.95).item(),
            }
            
            success_var_stats = {
                'mean': success_vars.mean().item(),
                'std': success_vars.std().item(),
                'min': success_vars.min().item(),
                'max': success_vars.max().item(),
                'percentile_5': torch.quantile(success_vars, 0.05).item(),
                'percentile_95': torch.quantile(success_vars, 0.95).item(),
            }
            
            # Compute statistics for failed students
            failure_mean_stats = {
                'mean': failure_means.mean().item(),
                'std': failure_means.std().item(),
                'min': failure_means.min().item(),
                'max': failure_means.max().item(),
                'percentile_5': torch.quantile(failure_means, 0.05).item(),
                'percentile_95': torch.quantile(failure_means, 0.95).item(),
            }
            
            failure_var_stats = {
                'mean': failure_vars.mean().item(),
                'std': failure_vars.std().item(),
                'min': failure_vars.min().item(),
                'max': failure_vars.max().item(),
                'percentile_5': torch.quantile(failure_vars, 0.05).item(),
                'percentile_95': torch.quantile(failure_vars, 0.95).item(),
            }
            
            # Recommend threshold parameters based on analysis
            # Use conservative bounds that capture most successful students
            recommended_mean_min = min(success_mean_stats['percentile_5'], failure_mean_stats['percentile_5']) - 0.1
            recommended_mean_max = max(success_mean_stats['percentile_95'], failure_mean_stats['percentile_95']) + 0.1
            recommended_var_min = min(success_var_stats['percentile_5'], failure_var_stats['percentile_5']) - 0.05
            recommended_var_max = max(success_var_stats['percentile_95'], failure_var_stats['percentile_95']) + 0.05
            
            # Ensure reasonable bounds
            recommended_mean_min = max(recommended_mean_min, -2.0)
            recommended_mean_max = min(recommended_mean_max, 2.0)
            recommended_var_min = max(recommended_var_min, 0.01)
            recommended_var_max = min(recommended_var_max, 5.0)
            
            # Calculate separation metrics
            mean_separation = abs(success_mean_stats['mean'] - failure_mean_stats['mean'])
            var_separation = abs(success_var_stats['mean'] - failure_var_stats['mean'])
            
            # Additional analysis for better understanding
            # Check parameter ranges and distributions
            all_means = torch.mean(students, dim=1)
            all_vars = torch.var(students, dim=1)
            all_norms = torch.norm(students, dim=1)
            all_max_abs = torch.max(torch.abs(students), dim=1)[0]
            
            # Analyze parameter statistics
            param_stats = {
                'all_mean_mean': all_means.mean().item(),
                'all_mean_std': all_means.std().item(),
                'all_var_mean': all_vars.mean().item(),
                'all_var_std': all_vars.std().item(),
                'all_norm_mean': all_norms.mean().item(),
                'all_norm_std': all_norms.std().item(),
                'all_max_abs_mean': all_max_abs.mean().item(),
                'all_max_abs_std': all_max_abs.std().item(),
            }
            
            # Check for alternative separation criteria
            norm_separation = abs(torch.mean(all_norms[success_mask]).item() - torch.mean(all_norms[failure_mask]).item())
            max_abs_separation = abs(torch.mean(all_max_abs[success_mask]).item() - torch.mean(all_max_abs[failure_mask]).item())
            
            # Determine best separation metric
            separations = {
                'mean': mean_separation,
                'variance': var_separation,
                'norm': norm_separation,
                'max_abs': max_abs_separation
            }
            best_separation_metric = max(separations, key=separations.get)
            best_separation_value = separations[best_separation_metric]
            
            # Improved separation quality assessment
            if best_separation_value > 0.2:
                separation_quality = 'excellent'
            elif best_separation_value > 0.1:
                separation_quality = 'good'
            elif best_separation_value > 0.05:
                separation_quality = 'fair'
            else:
                separation_quality = 'poor'
            
            stats = {
                'success_count': success_mask.sum().item(),
                'failure_count': failure_mask.sum().item(),
                'success_rate': success_mask.sum().item() / students.size(0),
                
                # Detailed statistics
                'success_mean_stats': success_mean_stats,
                'success_var_stats': success_var_stats,
                'failure_mean_stats': failure_mean_stats,
                'failure_var_stats': failure_var_stats,
                'param_stats': param_stats,
                
                # Recommended thresholds
                'recommended_mean_min': recommended_mean_min,
                'recommended_mean_max': recommended_mean_max,
                'recommended_var_min': recommended_var_min,
                'recommended_var_max': recommended_var_max,
                
                # Separation metrics
                'mean_separation': mean_separation,
                'var_separation': var_separation,
                'norm_separation': norm_separation,
                'max_abs_separation': max_abs_separation,
                'separations': separations,
                'best_separation_metric': best_separation_metric,
                'best_separation_value': best_separation_value,
                'separation_quality': separation_quality,
            }
            
            return stats
        else:
            return None


def determine_optimal_thresholds(seed, student_dim, device, alpha_teacher, w_sequences, eps_train, 
                                num_samples=10000, batch_size=1000):
    """
    Determine optimal threshold parameters for heuristic prediction by analyzing
    a sample of students and their training outcomes.
    
    Args:
        seed: Random seed for reproducibility
        student_dim: Dimension of student parameters
        device: Device to run computations on
        alpha_teacher: Teacher parameter
        w_sequences: List of w tensors
        eps_train: Epsilon threshold for training loss
        num_samples: Number of students to sample for analysis
        batch_size: Batch size for processing
        
    Returns:
        dict: Recommended threshold parameters and analysis results
    """
    logging.info(f"Determining optimal thresholds using {num_samples} sample students...")
    
    # Convert w_sequences to tensor if it's a list
    if isinstance(w_sequences, list):
        w_sequences = [w.to(device) for w in w_sequences]
    
    all_students = []
    all_train_losses = []
    
    # Sample students and compute their losses
    for batch in range(math.ceil(num_samples / batch_size)):
        bs = min(batch_size, num_samples - batch * batch_size)
        students = generate_students(student_dim, bs, device)
        train_loss, _ = get_losses(students, w_sequences, alpha_teacher)
        
        all_students.append(students)
        all_train_losses.append(train_loss)
    
    # Concatenate all results
    all_students = torch.cat(all_students, dim=0)
    all_train_losses = torch.cat(all_train_losses, dim=0)
    
    # Analyze characteristics
    analysis = analyze_student_characteristics(all_students, all_train_losses, eps_train, device)
    
    if analysis is not None:
        logging.info(f"Analysis complete:")
        logging.info(f"  Success rate: {analysis['success_rate']:.3f}")
        logging.info(f"  Mean separation: {analysis['mean_separation']:.3f}")
        logging.info(f"  Variance separation: {analysis['var_separation']:.3f}")
        logging.info(f"  Separation quality: {analysis['separation_quality']}")
        logging.info(f"  Recommended thresholds:")
        logging.info(f"    mean_min: {analysis['recommended_mean_min']:.3f}")
        logging.info(f"    mean_max: {analysis['recommended_mean_max']:.3f}")
        logging.info(f"    var_min: {analysis['recommended_var_min']:.3f}")
        logging.info(f"    var_max: {analysis['recommended_var_max']:.3f}")
        
        return analysis
    else:
        logging.warning("Could not determine optimal thresholds - no successful or failed students found")
        return {
            'recommended_mean_min': -1.0,
            'recommended_mean_max': 1.0,
            'recommended_var_min': 0.1,
            'recommended_var_max': 2.0,
            'separation_quality': 'unknown'
        }


def run_threshold_analysis_once(seed, student_dim, device, alpha_teacher, w_sequences, eps_train,
                               num_samples=50000, batch_size=1000, save_to_file=None):
    """
    Run threshold analysis once to determine optimal parameters for heuristic prediction.
    This is designed to be run once in advance since only a small fraction of students succeed.
    
    Args:
        seed: Random seed for reproducibility
        student_dim: Dimension of student parameters
        device: Device to run computations on
        alpha_teacher: Teacher parameter
        w_sequences: List of w tensors
        eps_train: Epsilon threshold for training loss
        num_samples: Number of students to sample for analysis (default: 50000)
        batch_size: Batch size for processing
        save_to_file: Optional file path to save results as JSON
        
    Returns:
        dict: Analysis results including recommended thresholds
    """
    logging.info(f"Running one-time threshold analysis with {num_samples} students...")
    logging.info("This analysis will determine optimal thresholds for heuristic prediction.")
    
    # Run the analysis
    analysis = determine_optimal_thresholds(
        seed, student_dim, device, alpha_teacher, w_sequences, eps_train,
        num_samples=num_samples, batch_size=batch_size
    )
    
    # Add metadata
    analysis['analysis_metadata'] = {
        'seed': seed,
        'student_dim': student_dim,
        'alpha_teacher': alpha_teacher,
        'eps_train': eps_train,
        'num_samples_analyzed': num_samples,
        'batch_size': batch_size,
        'timestamp': time.time()
    }
    
    # Save to file if requested
    if save_to_file:
        import json
        try:
            with open(save_to_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logging.info(f"Analysis results saved to {save_to_file}")
        except Exception as e:
            logging.warning(f"Could not save analysis to {save_to_file}: {e}")
    
    logging.info("Threshold analysis complete!")
    logging.info(f"Recommended thresholds for future use:")
    logging.info(f"  mean_min: {analysis['recommended_mean_min']:.3f}")
    logging.info(f"  mean_max: {analysis['recommended_mean_max']:.3f}")
    logging.info(f"  var_min: {analysis['recommended_var_min']:.3f}")
    logging.info(f"  var_max: {analysis['recommended_var_max']:.3f}")
    logging.info(f"  Separation quality: {analysis['separation_quality']}")
    
    return analysis


def load_threshold_analysis(file_path):
    """
    Load previously saved threshold analysis results.
    
    Args:
        file_path: Path to the saved analysis JSON file
        
    Returns:
        dict: Analysis results including recommended thresholds
    """
    import json
    try:
        with open(file_path, 'r') as f:
            analysis = json.load(f)
        logging.info(f"Loaded threshold analysis from {file_path}")
        return analysis
    except Exception as e:
        logging.error(f"Could not load analysis from {file_path}: {e}")
        return None


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
    
    # Statistics for prediction effectiveness
    total_students = 0
    predicted_above_epsilon = 0
    actual_above_epsilon = 0
    skipped_calculations = 0
    
    # Collect training losses for histogram if requested
    training_losses = [] if collect_training_losses else None

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
    
    # Log success count
    logging.info(f"GNC Total success count: {succ_count}")
    
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
    
    # Log success count
    logging.info(f"GNC Total success count: {succ_count}")
    
    return mean_prior, mean_gnc