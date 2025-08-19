import math
import torch
import logging
import numpy as np


def double_factorial(n):
    """Calculate double factorial n!! = n * (n-2) * (n-4) * ... * 1 for odd n"""
    if n == 1:
        return 1
    
    # Check for potential overflow
    if n > 20:  # math.factorial(20) is the largest safe value
        logging.warning(f"n={n} too large for double_factorial, may cause overflow")
        return float('inf')
    
    try:
        k = (n+1) // 2
        result = math.factorial(n) / (2**(k-1) * math.factorial(k-1))
        if math.isnan(result) or math.isinf(result):
            logging.error(f"double_factorial({n}) produced invalid result: {result}")
            return float('inf')
        return result
    except (OverflowError, ValueError) as e:
        logging.error(f"Error in double_factorial({n}): {e}")
        return float('inf')


def a_m_expectation(student_dim, m):
    # return E[a**m] for a ~ N(0, 1/d)
    return student_dim**(-m/2) * double_factorial(m-1)  if m % 2 == 0 else 0


def teacher_for_m(alpha_teacher, m):
    # Calculate teacher's m-th power response
    # This would need to be implemented based on the specific teacher structure
    # For now, using a placeholder that sums the teacher's parameters
    return alpha_teacher**m


def calc_mu(student_dim, alpha_teacher, sequence_length, device):
    # calculate the expected value of the student's output minus the teacher for the impulse response input
    # result is a vector of length sequence length - 1
    mu = torch.empty(sequence_length - 1, device=device)
    for m in range(1, sequence_length):
        mu[m-1] = student_dim * a_m_expectation(student_dim, m) - teacher_for_m(alpha_teacher, m)
    return mu


def calc_sigma(student_dim, sequence_length, device):
    # covariance matrix of the student's output minus the teacher for the impulse response input
    sigma = torch.empty((sequence_length - 1, sequence_length - 1), device=device)
    for m in range(1, sequence_length-1):
        for n in range(1, sequence_length-1):
            sigma[m-1, n-1] = student_dim * (a_m_expectation(student_dim, m+n) - a_m_expectation(student_dim, m) * a_m_expectation(student_dim, n))
    return sigma


def safe_divide(numerator, denominator, eps=1e-12):
    """Safely divide with numerical stability check"""
    # Handle both tensor and scalar inputs
    if isinstance(denominator, torch.Tensor):
        if torch.abs(denominator) < eps:
            logging.warning(f"Denominator too small: {denominator}, using eps={eps}")
            return numerator / eps
    else:
        if abs(denominator) < eps:
            logging.warning(f"Denominator too small: {denominator}, using eps={eps}")
            return numerator / eps
    return numerator / denominator


def is_reasonable_loss(loss_value, threshold=1e6):
    """Check if loss value is reasonable (not too large or too small)"""
    if torch.isnan(loss_value) or torch.isinf(loss_value):
        return False
    # Handle both tensor and scalar inputs
    if isinstance(loss_value, torch.Tensor):
        if torch.abs(loss_value) > threshold:
            return False
    else:
        if abs(loss_value) > threshold:
            return False
    return True


def calc_asymptotic_coefficients(alpha_teacher, w, sequence_length, device):
    """
    Calculate asymptotic coefficients A and B for the loss expansion:
    L(d) = A + B/d + O(1/d^2)
    
    Based on the asymptotic expansions of μ_S and Σ_S:
    μ_S(d) = μ_0 + μ_1/d + O(1/d^2)
    Σ_S(d) = Σ_0 + Σ_1/d + O(1/d^(3/2))
    """
    # Ensure vector shape for w
    w_vec = w.squeeze()
    
    # Check for numerical stability
    if torch.any(torch.isnan(w_vec)) or torch.any(torch.isinf(w_vec)):
        logging.error(f"Invalid w values detected: {w_vec}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    # Calculate μ_0 and μ_1
    mu_0 = torch.empty(sequence_length - 1, device=device)
    mu_1 = torch.empty(sequence_length - 1, device=device)
    
    for m in range(1, sequence_length):
        if m % 2 == 1:  # odd m
            mu_0[m-1] = -alpha_teacher**m
        elif m == 2:  # m = 2
            mu_0[m-1] = 1 - alpha_teacher**m
        else:  # m >= 4, even
            mu_0[m-1] = -alpha_teacher**m
        
        # μ_1 has non-zero term only for m = 4
        if m == 4:
            mu_1[m-1] = 3.0  # E[Z^4] = 3
        else:
            mu_1[m-1] = 0.0
    
    # Calculate Σ_0 and Σ_1
    sigma_0 = torch.zeros((sequence_length - 1, sequence_length - 1), device=device)
    sigma_1 = torch.zeros((sequence_length - 1, sequence_length - 1), device=device)
    
    # Σ_0: only (1,1) element is non-zero
    sigma_0[0, 0] = 1.0
    
    # Σ_1: non-zero elements for m+n=4
    for m in range(1, sequence_length-1):
        for n in range(1, sequence_length-1):
            if m == 2 and n == 2:
                sigma_1[m-1, n-1] = 2.0  # E[Z^4] - E[Z^2]^2 = 3 - 1 = 2
            elif (m == 1 and n == 3) or (m == 3 and n == 1):
                sigma_1[m-1, n-1] = 3.0  # E[Z^4] = 3
    
    # Calculate asymptotic conditional mean μ_{c,0}
    w_transpose_mu_0 = torch.dot(w_vec, mu_0)
    w_transpose_sigma_0_w = torch.dot(w_vec, sigma_0 @ w_vec)  # This should be w_1^2
    
    # Check for numerical stability
    if torch.abs(w_transpose_sigma_0_w) < 1e-12:
        logging.warning(f"w_transpose_sigma_0_w too small: {w_transpose_sigma_0_w}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    F_0 = safe_divide(w_transpose_mu_0, w_transpose_sigma_0_w)
    
    mu_c_0 = mu_0 - F_0 * (sigma_0 @ w_vec)
    
    # Calculate coefficient A
    A = torch.dot(mu_c_0, mu_c_0)
    
    # Check for numerical stability
    if torch.isnan(A) or torch.isinf(A):
        logging.error(f"Invalid A coefficient: {A}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    # Calculate coefficient B components
    # First component: 2 * μ_{c,0}^T * μ_{c,1}
    w_transpose_mu_1 = torch.dot(w_vec, mu_1)
    w_transpose_sigma_1_w = torch.dot(w_vec, sigma_1 @ w_vec)
    F_1 = safe_divide(w_transpose_mu_1, w_transpose_sigma_0_w) - safe_divide(w_transpose_mu_0 * w_transpose_sigma_1_w, w_transpose_sigma_0_w**2)
    
    mu_c_1 = mu_1 - F_0 * (sigma_1 @ w_vec) - F_1 * (sigma_0 @ w_vec)
    
    # Second component: Tr(Σ_1) - correction terms
    sigma_0_sigma_1_w = sigma_0 @ sigma_1 @ w_vec
    sigma_1_sigma_0_w = sigma_1 @ sigma_0 @ w_vec
    w_transpose_sigma_0_sigma_1_w = torch.dot(w_vec, sigma_0_sigma_1_w)
    w_transpose_sigma_1_sigma_0_w = torch.dot(w_vec, sigma_1_sigma_0_w)
    
    variance_component = torch.trace(sigma_1) - safe_divide(w_transpose_sigma_0_sigma_1_w + w_transpose_sigma_1_sigma_0_w - w_transpose_sigma_1_w, w_transpose_sigma_0_w)
    
    B = 2 * torch.dot(mu_c_0, mu_c_1) + variance_component
    
    # Check for numerical stability
    if torch.isnan(B) or torch.isinf(B):
        logging.error(f"Invalid B coefficient: {B}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)

    delta_l_infinity = 1 - 2 * alpha_teacher * safe_divide(w_transpose_mu_0, w_vec[0].item()) - safe_divide(w_transpose_mu_0**2, w_vec[0].item() **2)
    #logging.info(f"delta_l_infinity: {delta_l_infinity} for w={w}")
    
    return A, B, delta_l_infinity


def gnc_theoretical_loss(alpha_teacher, w, student_dim, device):
    # Get sequence length from dataset
    sequence_length = w.shape[0] + 1

    # Add debugging information
    logging.debug(f"Computing theoretical loss for student_dim={student_dim}, alpha_teacher={alpha_teacher}, w={w}")

    # Check for numerical stability in inputs
    if torch.any(torch.isnan(w)) or torch.any(torch.isinf(w)):
        logging.error(f"Invalid w values: {w}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    if math.isnan(alpha_teacher) or math.isinf(alpha_teacher):
        logging.error(f"Invalid alpha_teacher: {alpha_teacher}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)

    mu = calc_mu(student_dim, alpha_teacher, sequence_length, device)
    sigma = calc_sigma(student_dim, sequence_length, device)
    
    # Check for numerical stability in mu and sigma
    if torch.any(torch.isnan(mu)) or torch.any(torch.isinf(mu)):
        logging.error(f"Invalid mu values: {mu}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    if torch.any(torch.isnan(sigma)) or torch.any(torch.isinf(sigma)):
        logging.error(f"Invalid sigma values: {sigma}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    # Calculate the conditional expectation E[||S||^2 | w^T S = 0]
    # where S ~ N(mu_S, Sigma_S) and we condition on w^T S = 0
    
    # Prior Loss: μ_S^T μ_S + Tr(Σ_S)
    prior_loss = torch.dot(mu, mu) + torch.trace(sigma)
    
    # Mean Shift Term 1: -2 * (w^T μ_S) * (μ_S^T Σ_S w) / (w^T Σ_S w)
    w_transpose_mu = torch.dot(w.squeeze(), mu)  # w^T μ_S
    mu_transpose_sigma_w = torch.dot(mu, sigma @ w.squeeze())  # μ_S^T Σ_S w
    w_transpose_sigma_w = torch.dot(w.squeeze(), sigma @ w.squeeze())  # w^T Σ_S w
    
    # Check for numerical stability
    if torch.abs(w_transpose_sigma_w) < 1e-12:
        logging.warning(f"w_transpose_sigma_w too small: {w_transpose_sigma_w}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    mean_shift_term1 = -2 * w_transpose_mu * safe_divide(mu_transpose_sigma_w, w_transpose_sigma_w)
    
    # Mean Shift Term 2: (w^T μ_S)^2 * (w^T Σ_S^2 w) / (w^T Σ_S w)^2
    w_transpose_sigma_squared_w = torch.dot(w.squeeze(), sigma @ sigma @ w.squeeze())  # w^T Σ_S^2 w
    mean_shift_term2 = (w_transpose_mu**2) * safe_divide(w_transpose_sigma_squared_w, w_transpose_sigma_w**2)
    
    # Variance Reduction: -w^T Σ_S^2 w / (w^T Σ_S w)
    variance_reduction = -safe_divide(w_transpose_sigma_squared_w, w_transpose_sigma_w)
    
    # Total conditional expectation
    conditional_expectation = prior_loss + mean_shift_term1 + mean_shift_term2 + variance_reduction
    
    # Log intermediate values for debugging
    logging.debug(f"student_dim={student_dim}: prior_loss={prior_loss:.6f}, mean_shift_term1={mean_shift_term1:.6f}, mean_shift_term2={mean_shift_term2:.6f}, variance_reduction={variance_reduction:.6f}")
    logging.debug(f"student_dim={student_dim}: conditional_expectation={conditional_expectation:.6f}")
    
    # Check for numerical stability in final result
    if torch.isnan(conditional_expectation) or torch.isinf(conditional_expectation):
        logging.error(f"Invalid conditional_expectation: {conditional_expectation}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)

    # Calculate asymptotic coefficients
    A, B, delta_l_infinity = calc_asymptotic_coefficients(alpha_teacher, w, sequence_length, device)
    
    # Check for numerical stability in asymptotic coefficients
    if torch.isnan(A) or torch.isnan(B):
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    # Asymptotic conditional expectation: A + B/d
    asymptotic_conditional_expectation = A + safe_divide(B, student_dim)
    
    # Final check for numerical stability
    if torch.isnan(asymptotic_conditional_expectation) or torch.isinf(asymptotic_conditional_expectation):
        logging.error(f"Invalid asymptotic_conditional_expectation: {asymptotic_conditional_expectation}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    # Check if results are reasonable
    if not is_reasonable_loss(conditional_expectation):
        logging.warning(f"Unreasonable conditional_expectation: {conditional_expectation}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    if not is_reasonable_loss(asymptotic_conditional_expectation):
        logging.warning(f"Unreasonable asymptotic_conditional_expectation: {asymptotic_conditional_expectation}")
        return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
    
    return conditional_expectation, asymptotic_conditional_expectation, delta_l_infinity


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from generator import generate_teacher_alpha, generate_w
    for seed in range(100):
        torch.manual_seed(seed)
        alpha_teacher = generate_teacher_alpha(device)
        dataset = generate_w(5, device)
        exact_loss, asymptotic_loss, delta_l_infinity = gnc_theoretical_loss(alpha_teacher, dataset, 400, device)
        #print(f"seed={seed} alpha_teacher={alpha_teacher.item():.6f} dataset={dataset.squeeze().tolist()} asymptotic_loss={asymptotic_loss.item():.6f}")
        if delta_l_infinity.item() < -10:
            print(f"seed={seed} delta_l_infinity={delta_l_infinity.item():.6f}")
        #for d in range(100, 400, 10):
        #    exact_loss, asymptotic_loss = gnc_theoretical_loss(alpha_teacher, dataset, d, device)
        #    print(f"d={d}: Exact={exact_loss.item():.6f}, Asymptotic={asymptotic_loss.item():.6f}")
