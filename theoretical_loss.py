import math
import torch
import logging


def double_factorial(n):
    """Calculate double factorial n!! = n * (n-2) * (n-4) * ... * 1 for odd n"""
    if n == 1:
        return 1

    k = (n+1) // 2
    return math.factorial(n) / (2**(k-1) * math.factorial(k-1))


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


def calc_asymptotic_coefficients(alpha_teacher, w, sequence_length, device):
    """
    Calculate asymptotic coefficients A and B for the loss expansion:
    L(d) = A + B/d + O(1/d^2)
    
    Based on the asymptotic expansions of μ_S and Σ_S:
    μ_S(d) = μ_0 + μ_1/d + O(1/d^2)
    Σ_S(d) = Σ_0 + Σ_1/d + O(1/d^(3/2))
    """
    # Calculate μ_0 and μ_1 (initialize to zeros for safety)
    mu_0 = torch.zeros(sequence_length - 1, device=device)
    mu_1 = torch.zeros(sequence_length - 1, device=device)
    
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
    w_vec = w.reshape(-1)
    w_transpose_mu_0 = torch.dot(w_vec, mu_0)
    w_transpose_sigma_0_w = torch.dot(w_vec, sigma_0 @ w_vec)  # This should be w_1^2
    # Numerical safeguard: avoid division by very small w[0]^2
    denom0 = torch.clamp(w_transpose_sigma_0_w, min=1e-12)
    F_0 = w_transpose_mu_0 / denom0
    
    mu_c_0 = mu_0 - F_0 * (sigma_0 @ w_vec)
    
    # Calculate coefficient A
    A = torch.dot(mu_c_0, mu_c_0)
    
    # Calculate coefficient B components
    # First component: 2 * μ_{c,0}^T * μ_{c,1}
    w_transpose_mu_1 = torch.dot(w_vec, mu_1)
    w_transpose_sigma_1_w = torch.dot(w_vec, sigma_1 @ w_vec)
    F_1 = w_transpose_mu_1 / denom0 - (w_transpose_mu_0 * w_transpose_sigma_1_w) / (denom0**2)
    
    mu_c_1 = mu_1 - F_0 * (sigma_1 @ w_vec) - F_1 * (sigma_0 @ w_vec)
    
    # Second component: Tr(Σ_1) - correction terms
    sigma_0_sigma_1_w = sigma_0 @ sigma_1 @ w_vec
    sigma_1_sigma_0_w = sigma_1 @ sigma_0 @ w_vec
    w_transpose_sigma_0_sigma_1_w = torch.dot(w_vec, sigma_0_sigma_1_w)
    w_transpose_sigma_1_sigma_0_w = torch.dot(w_vec, sigma_1_sigma_0_w)
    
    variance_component = torch.trace(sigma_1) - (w_transpose_sigma_0_sigma_1_w + w_transpose_sigma_1_sigma_0_w - w_transpose_sigma_1_w) / denom0
    
    B = 2 * torch.dot(mu_c_0, mu_c_1) + variance_component

    # Guard against w[0] extremely small in logging helper as well
    w0 = float(w_vec[0].item())
    if abs(w0) < 1e-6:
        w0 = 1e-6 if w0 >= 0 else -1e-6
    delta_l_infinity = 1 - 2 * alpha_teacher * w_transpose_mu_0 / w0 - (w_transpose_mu_0**2) /  (w0 ** 2)
    logging.info(f"delta_l_infinity: {delta_l_infinity} for w={w}")
    
    return A, B


def gnc_theoretical_loss(alpha_teacher, w, student_dim, device):
    # Get sequence length from dataset
    sequence_length = w.shape[0] + 1

    mu = calc_mu(student_dim, alpha_teacher, sequence_length, device)
    sigma = calc_sigma(student_dim, sequence_length, device)
    
    # Calculate the conditional expectation E[||S||^2 | w^T S = 0]
    # where S ~ N(mu_S, Sigma_S) and we condition on w^T S = 0
    
    # Prior Loss: μ_S^T μ_S + Tr(Σ_S)
    prior_loss = torch.dot(mu, mu) + torch.trace(sigma)
    
    # Mean Shift Term 1: -2 * (w^T μ_S) * (μ_S^T Σ_S w) / (w^T Σ_S w)
    w_vec = w.reshape(-1)
    w_transpose_mu = torch.dot(w_vec, mu)  # w^T μ_S
    mu_transpose_sigma_w = torch.dot(mu, sigma @ w_vec)  # μ_S^T Σ_S w
    w_transpose_sigma_w = torch.dot(w_vec, sigma @ w_vec)  # w^T Σ_S w
    denom = torch.clamp(w_transpose_sigma_w, min=1e-12)
    mean_shift_term1 = -2 * w_transpose_mu * mu_transpose_sigma_w / denom
    
    # Mean Shift Term 2: (w^T μ_S)^2 * (w^T Σ_S^2 w) / (w^T Σ_S w)^2
    w_transpose_sigma_squared_w = torch.dot(w_vec, sigma @ sigma @ w_vec)  # w^T Σ_S^2 w
    mean_shift_term2 = (w_transpose_mu**2) * w_transpose_sigma_squared_w / (denom**2)
    
    # Variance Reduction: -w^T Σ_S^2 w / (w^T Σ_S w)
    variance_reduction = -w_transpose_sigma_squared_w / denom
    
    # Total conditional expectation
    conditional_expectation = prior_loss + mean_shift_term1 + mean_shift_term2 + variance_reduction

    # Calculate asymptotic coefficients
    A, B = calc_asymptotic_coefficients(alpha_teacher, w, sequence_length, device)
    
    # Asymptotic conditional expectation: A + B/d
    asymptotic_conditional_expectation = A + B / student_dim
    
    return conditional_expectation, asymptotic_conditional_expectation


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from generator import generate_teacher_alpha, generate_w
    for seed in range(1):
        torch.manual_seed(seed)
        alpha_teacher = generate_teacher_alpha(device)
        dataset = generate_w(5, device)
        print(f"alpha_teacher={alpha_teacher}")
        print(f"dataset={dataset}")
        for d in range(100, 400, 10):
            exact_loss, asymptotic_loss = gnc_theoretical_loss(alpha_teacher, dataset, d, device)
            print(f"d={d}: Exact={exact_loss.item():.6f}, Asymptotic={asymptotic_loss.item():.6f}")
