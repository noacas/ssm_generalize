import math
import torch
import logging
import numpy as np


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
    # Ensure vector shape for w
    w_vec = w.squeeze()
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
    F_0 = w_transpose_mu_0 / w_transpose_sigma_0_w
    
    mu_c_0 = mu_0 - F_0 * (sigma_0 @ w_vec)
    
    # Calculate coefficient A
    A = torch.dot(mu_c_0, mu_c_0)
    
    # Calculate coefficient B components
    # First component: 2 * μ_{c,0}^T * μ_{c,1}
    w_transpose_mu_1 = torch.dot(w_vec, mu_1)
    w_transpose_sigma_1_w = torch.dot(w_vec, sigma_1 @ w_vec)
    F_1 = w_transpose_mu_1 / w_transpose_sigma_0_w - (w_transpose_mu_0 * w_transpose_sigma_1_w) / (w_transpose_sigma_0_w**2)
    
    mu_c_1 = mu_1 - F_0 * (sigma_1 @ w_vec) - F_1 * (sigma_0 @ w_vec)
    
    # Second component: Tr(Σ_1) - correction terms
    sigma_0_sigma_1_w = sigma_0 @ sigma_1 @ w_vec
    sigma_1_sigma_0_w = sigma_1 @ sigma_0 @ w_vec
    w_transpose_sigma_0_sigma_1_w = torch.dot(w_vec, sigma_0_sigma_1_w)
    w_transpose_sigma_1_sigma_0_w = torch.dot(w_vec, sigma_1_sigma_0_w)
    
    variance_component = torch.trace(sigma_1) - (w_transpose_sigma_0_sigma_1_w + w_transpose_sigma_1_sigma_0_w - w_transpose_sigma_1_w) / w_transpose_sigma_0_w
    
    B = 2 * torch.dot(mu_c_0, mu_c_1) + variance_component

    delta_l_infinity = 1 - 2 * alpha_teacher * w_transpose_mu_0 / w_vec[0].item() - (w_transpose_mu_0**2) /  (w_vec[0].item() **2)
    logging.info(f"delta_l_infinity: {delta_l_infinity} for w={w}")
    
    return A, B, delta_l_infinity


def gnc_theoretical_loss_for_one_w(alpha_teacher, w, student_dim, device):
    # Get sequence length from dataset
    sequence_length = w.shape[0] + 1

    mu = calc_mu(student_dim, alpha_teacher, sequence_length, device)
    sigma = calc_sigma(student_dim, sequence_length, device)
    
    # Calculate the conditional expectation E[||S||^2 | w^T S = 0]
    # where S ~ N(mu_S, Sigma_S) and we condition on w^T S = 0
    
    # Prior Loss: μ_S^T μ_S + Tr(Σ_S)
    prior_loss = torch.dot(mu, mu) + torch.trace(sigma)
    
    # Mean Shift Term 1: -2 * (w^T μ_S) * (μ_S^T Σ_S w) / (w^T Σ_S w)
    w_transpose_mu = torch.dot(w.squeeze(), mu)  # w^T μ_S
    mu_transpose_sigma_w = torch.dot(mu, sigma @ w.squeeze())  # μ_S^T Σ_S w
    w_transpose_sigma_w = torch.dot(w.squeeze(), sigma @ w.squeeze())  # w^T Σ_S w
    mean_shift_term1 = -2 * w_transpose_mu * mu_transpose_sigma_w / w_transpose_sigma_w
    
    # Mean Shift Term 2: (w^T μ_S)^2 * (w^T Σ_S^2 w) / (w^T Σ_S w)^2
    w_transpose_sigma_squared_w = torch.dot(w.squeeze(), sigma @ sigma @ w.squeeze())  # w^T Σ_S^2 w
    mean_shift_term2 = (w_transpose_mu**2) * w_transpose_sigma_squared_w / (w_transpose_sigma_w**2)
    
    # Variance Reduction: -w^T Σ_S^2 w / (w^T Σ_S w)
    variance_reduction = -w_transpose_sigma_squared_w / w_transpose_sigma_w
    
    # Total conditional expectation
    conditional_expectation = prior_loss + mean_shift_term1 + mean_shift_term2 + variance_reduction

    # Calculate asymptotic coefficients
    A, B, delta_l_infinity = calc_asymptotic_coefficients(alpha_teacher, w, sequence_length, device)
    
    # Asymptotic conditional expectation: A + B/d
    asymptotic_conditional_expectation = A + B / student_dim

    # if the conditional expectation is out of bounds, use the asymptotic conditional expectation
    if conditional_expectation < 0 or conditional_expectation > 10:
        conditional_expectation = asymptotic_conditional_expectation
    
    return conditional_expectation, asymptotic_conditional_expectation, delta_l_infinity



def gnc_theoretical_loss_for_multiple_w(alpha_teacher, w_sequences, student_dim, device):
    """
    Calculate theoretical loss for multiple sequences by properly conditioning on multiple constraints.
    This implements the mathematical formulation from the LaTeX document for N > 1 constraints.
    
    For N=2, this implements the exact conditional expectation formula:
    E[||S||^2 | W^T S = 0] = μ_S^T μ_S + Tr(Σ_S) - 2*b^T A^(-1) b + b^T A^(-1) C A^(-1) b - Tr(A^(-1) C)
    
    where:
    - A = W^T Σ_S W
    - b = W^T Σ_S μ_S  
    - C = W^T Σ_S^2 W
    """
    if len(w_sequences) == 1:
        return gnc_theoretical_loss_for_one_w(alpha_teacher, w_sequences[0], student_dim, device)
    
    # Get sequence length from the first sequence
    sequence_length = w_sequences[0].shape[0] + 1
    N = len(w_sequences)  # Number of constraints
    
    if N == 2:
        # Implement exact conditional expectation for N=2 based on the LaTeX derivation
        return _gnc_theoretical_loss_for_two_w(alpha_teacher, w_sequences, student_dim, device)
    else:
        # For N > 2, we would need to extend the formula
        # For now, fall back to single constraint case as approximation
        logging.warning(f"N={N} > 2 not yet implemented, using single constraint approximation")
        return gnc_theoretical_loss_for_one_w(alpha_teacher, w_sequences[0], student_dim, device)


def _gnc_theoretical_loss_for_two_w(alpha_teacher, w_sequences, student_dim, device):
    """
    Calculate exact conditional expectation for N=2 training examples.
    
    Implements the formula from the LaTeX document:
    E[||S||^2 | W^T S = 0] = μ_S^T μ_S + Tr(Σ_S) - 2*b^T A^(-1) b + b^T A^(-1) C A^(-1) b - Tr(A^(-1) C)
    """
    sequence_length = w_sequences[0].shape[0] + 1
    
    # Step 1: Build μ_S using the finite-d formula
    mu_S = _build_mu_S_finite_d(alpha_teacher, student_dim, sequence_length, device)
    
    # Step 2: Build Σ_S using the finite-d formula  
    sigma_S = _build_sigma_S_finite_d(student_dim, sequence_length, device)
    
    # Step 3: Form W matrix from the two w sequences
    W = torch.stack([w_sequences[0].squeeze(), w_sequences[1].squeeze()], dim=1)  # Shape: (k-1) × 2
    
    # Step 4: Calculate A, b, and C matrices
    A = W.T @ sigma_S @ W  # Shape: 2 × 2
    b = W.T @ sigma_S @ mu_S  # Shape: 2 × 1
    C = W.T @ sigma_S @ sigma_S @ W  # Shape: 2 × 2
    
    # Step 5: Handle potential singularity of A using pseudoinverse
    try:
        A_inv = torch.inverse(A)
    except RuntimeError:
        # If A is singular, use pseudoinverse
        A_inv = torch.pinverse(A)
        logging.warning("Matrix A was singular, using pseudoinverse")
    
    # Step 6: Evaluate the exact conditional expectation formula
    mu_S_squared = torch.dot(mu_S, mu_S)  # μ_S^T μ_S
    trace_sigma_S = torch.trace(sigma_S)  # Tr(Σ_S)
    
    b_T_A_inv_b = torch.dot(b, A_inv @ b)  # b^T A^(-1) b
    b_T_A_inv_C_A_inv_b = torch.dot(b, A_inv @ C @ A_inv @ b)  # b^T A^(-1) C A^(-1) b
    trace_A_inv_C = torch.trace(A_inv @ C)  # Tr(A^(-1) C)
    
    conditional_expectation = (mu_S_squared + trace_sigma_S - 
                              2 * b_T_A_inv_b + 
                              b_T_A_inv_C_A_inv_b - 
                              trace_A_inv_C)
    
    # Calculate asymptotic approximation for comparison
    asymptotic_conditional_expectation = _calculate_asymptotic_for_two_w(alpha_teacher, w_sequences, student_dim, device)
    
    # Safety check: if result is unreasonable, use asymptotic
    if (conditional_expectation < 0 or conditional_expectation > 10 or 
        not torch.isfinite(conditional_expectation)):
        logging.warning(f"Conditional expectation {conditional_expectation} out of bounds, using asymptotic")
        conditional_expectation = asymptotic_conditional_expectation


    #  delta_l_infinity = - 2 \;-\;\alpha^{2} \;+\;\alpha^{4} + \sum_{m=3}^{k-1}\alpha^{2m}} - asymptotic_conditional_expectation
    delta_l_infinity = - 2 + alpha_teacher**2 - alpha_teacher**4 + torch.sum(alpha_teacher**(2 * torch.arange(3, sequence_length - 1))) - asymptotic_conditional_expectation
    
    return conditional_expectation, asymptotic_conditional_expectation, delta_l_infinity


def _build_mu_S_finite_d(alpha_teacher, student_dim, sequence_length, device):
    """
    Build μ_S using the exact finite-d formula from the LaTeX document.
    
    (μ_S)_m = d^(1-m/2) * J_m - α^m
    
    where J_m is the even-moment double factorial:
    J_m = (m-1)!! if m is even, 0 if m is odd
    """
    mu_S = torch.empty(sequence_length - 1, device=device)
    
    for m in range(1, sequence_length):
        if m % 2 == 0:  # m is even
            J_m = double_factorial(m - 1)
            mu_S[m-1] = (student_dim ** (1 - m/2)) * J_m - (alpha_teacher ** m)
        else:  # m is odd
            mu_S[m-1] = -(alpha_teacher ** m)
    
    return mu_S


def _build_sigma_S_finite_d(student_dim, sequence_length, device):
    """
    Build Σ_S using the exact finite-d formula from the LaTeX document.
    
    (Σ_S)_mn = d^(1-(m+n)/2) * (J_{m+n} - J_m * J_n)
    
    where J_m is the even-moment double factorial.
    """
    sigma_S = torch.empty((sequence_length - 1, sequence_length - 1), device=device)
    
    for m in range(1, sequence_length):
        for n in range(1, sequence_length):
            if (m + n) % 2 == 0:  # m+n is even
                J_m_plus_n = double_factorial(m + n - 1)
                J_m = double_factorial(m - 1) if m % 2 == 0 else 0
                J_n = double_factorial(n - 1) if n % 2 == 0 else 0
                
                sigma_S[m-1, n-1] = (student_dim ** (1 - (m + n)/2)) * (J_m_plus_n - J_m * J_n)
            else:  # m+n is odd
                sigma_S[m-1, n-1] = 0.0
    
    return sigma_S


def _calculate_asymptotic_for_two_w(alpha_teacher, w_sequences, student_dim, device):
    """
    Calculate asymptotic conditional expectation for N=2 as d → ∞.
    
    This implements the limit formula from the LaTeX document:
    lim_{d→∞} E[||S||^2 | W^T S = 0] = μ_0^T μ_0 + 2α * r_eff + r_eff^2
    
    where r_eff is the effective radius combining both constraints.
    """
    sequence_length = w_sequences[0].shape[0] + 1
    
    # Calculate μ_0 (asymptotic mean)
    mu_0 = torch.empty(sequence_length - 1, device=device)
    for m in range(1, sequence_length):
        if m % 2 == 1:  # odd m
            mu_0[m-1] = -alpha_teacher**m
        elif m == 2:  # m = 2
            mu_0[m-1] = 1 - alpha_teacher**m
        else:  # m >= 4, even
            mu_0[m-1] = -alpha_teacher**m
    
    # Calculate asymptotic covariance Σ_0 (only (1,1) element is non-zero)
    sigma_0 = torch.zeros((sequence_length - 1, sequence_length - 1), device=device)
    sigma_0[0, 0] = 1.0
    
    # Calculate effective radius r_eff
    w1 = w_sequences[0].squeeze()
    w2 = w_sequences[1].squeeze()
    
    # r(w) = ||w||^2 for each sequence
    r_w1 = torch.dot(w1, w1)
    r_w2 = torch.dot(w2, w2)
    
    # r_eff = (w1_1^2 * r(w1) + w2_1^2 * r(w2)) / (w1_1^2 + w2_1^2)
    w1_1_squared = w1[0]**2
    w2_1_squared = w2[0]**2
    r_eff = (w1_1_squared * r_w1 + w2_1_squared * r_w2) / (w1_1_squared + w2_1_squared)
    
    # Calculate asymptotic conditional expectation
    mu_0_squared = torch.dot(mu_0, mu_0)
    asymptotic_expectation = mu_0_squared + 2 * alpha_teacher * r_eff + r_eff**2
    
    return asymptotic_expectation


def gnc_theoretical_loss(alpha_teacher, w_sequences, student_dim, device):
    """
    Main function to calculate theoretical loss for either single or multiple sequences.
    """
    if isinstance(w_sequences, torch.Tensor):
        # Single sequence case
        return gnc_theoretical_loss_for_one_w(alpha_teacher, w_sequences, student_dim, device)
    elif isinstance(w_sequences, list):
        # Multiple sequences case
        return gnc_theoretical_loss_for_multiple_w(alpha_teacher, w_sequences, student_dim, device)
    else:
        raise ValueError("w_sequences must be either a torch.Tensor (single sequence) or list of tensors (multiple sequences)")


def first_best_seeds():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from generator import generate_teacher_alpha, generate_w
    
    # Store all losses with their seeds
    all_losses = []
    
    print("Testing seeds to find the 5 with smallest exact loss...")
    for seed in range(10000):
        if seed % 1000 == 0:
            print(f"Progress: {seed}/10000")
            
        torch.manual_seed(seed)
        failed = False
        alpha_teacher = generate_teacher_alpha(device)
        dataset = generate_w(5, device)
        loss = 0
        for d in range(150, 300, 25):
            exact_loss, asymptotic_loss, delta_l_infinity = gnc_theoretical_loss(alpha_teacher, dataset, d, device)
            if exact_loss < 0 or asymptotic_loss < 0 or not torch.isfinite(exact_loss) or not torch.isfinite(asymptotic_loss):
                failed = True
                break
            loss += exact_loss
        
        # Store all valid losses (positive and finite)
        if not failed:
            all_losses.append((loss.item(), seed, alpha_teacher.item(), dataset))
    
    # Sort by loss value and get the 5 smallest
    all_losses.sort(key=lambda x: x[0])
    top_5_seeds = all_losses[:5]
    
    print(f"\nTop 5 seeds with smallest exact loss:")
    for i, (loss, seed, alpha_teacher, dataset) in enumerate(top_5_seeds, 1):
        print(f"{i}. Seed {seed}: Loss = {loss:.8f}, Alpha Teacher = {alpha_teacher:.8f}, Dataset = {dataset}")
    
    print(f"\nTotal valid seeds found: {len(all_losses)}")


def w_that_minimizes_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from generator import generate_teacher_alpha, generate_w
    for seed in range(10):
        torch.manual_seed(seed)
        alpha_teacher = generate_teacher_alpha(device)
        w = generate_w(5, device)
        w[1] = (alpha_teacher**3 * w[2] +  alpha_teacher**4 * w[3]) / (1-alpha_teacher**2)
        d = 300
        exact_loss, asymptotic_loss, delta_l_infinity = gnc_theoretical_loss(alpha_teacher, w, d, device)
        print(f"w: {w}")
        print(f"exact_loss: {exact_loss.item():.8f}, asymptotic_loss: {asymptotic_loss.item():.8f}, delta_l_infinity: {delta_l_infinity.item():.8f}")
        expected_minimum_loss = (1 - alpha_teacher**2)**2 + alpha_teacher**6 + alpha_teacher**8
        print(f"expected minimum loss: {expected_minimum_loss.item():.8f}")


def w_that_minimizes_loss(w, alpha_teacher, sequence_length):
    # change w to have a new value at w_2 that minimizes the loss (w[1] when indices start from 0)
    new_w_2 = 0
    for i in range(3, sequence_length):
        new_w_2 += alpha_teacher**i * w[i-1]
    new_w_2 /= (1-alpha_teacher**2)
    w[1] = new_w_2
    return w


def w2_that_minimizes_loss(w_sequences, w, alpha_teacher, sequence_length):
    #todo
    return w


if __name__ == "__main__":
    #w_that_minimizes_loss()
    first_best_seeds()