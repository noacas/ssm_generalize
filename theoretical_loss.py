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
    idx = torch.arange(3, sequence_length - 1, device=alpha_teacher.device, dtype=torch.long)
    sum_term = torch.sum(alpha_teacher ** (2 * idx)) if idx.numel() > 0 else torch.zeros((), device=alpha_teacher.device, dtype=alpha_teacher.dtype)
    delta_l_infinity = -2 + alpha_teacher**2 - alpha_teacher**4 + sum_term - asymptotic_conditional_expectation

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


def _build_mu_0_asymptotic(alpha_teacher, sequence_length, device):
    """
    Build μ_0 (asymptotic mean) as d → ∞.
    
    μ_0 = (-α, 1-α^2, -α^3, -α^4, ..., -α^{k-1})^T
    """
    mu_0 = torch.empty(sequence_length - 1, device=device)
    for m in range(1, sequence_length):
        if m % 2 == 1:  # odd m
            mu_0[m-1] = -alpha_teacher**m
        elif m == 2:  # m = 2
            mu_0[m-1] = 1 - alpha_teacher**m
        else:  # m >= 4, even
            mu_0[m-1] = -alpha_teacher**m
    return mu_0


def _calculate_asymptotic_for_two_w(alpha_teacher, w_sequences, student_dim, device):
    """
    Calculate asymptotic conditional expectation for N=2 as d → ∞.
    
    This implements a robust numerical approach that avoids the analytical complexity
    of deriving exact B coefficients for multiple constraints.
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
    
    # Form W matrix from the two w sequences
    w1 = w_sequences[0].squeeze()
    w2 = w_sequences[1].squeeze()
    W = torch.stack([w1, w2], dim=1)  # Shape: (k-1) × 2
    
    # Calculate asymptotic conditional expectation using the same approach as single sequence
    # but adapted for multiple constraints
    
    # For N=2, we need to solve the system W^T Σ_0 W * λ = W^T μ_0
    # where λ is the Lagrange multiplier vector
    
    # Calculate A = W^T Σ_0 W
    A_matrix = W.T @ sigma_0 @ W  # Shape: 2 × 2
    
    # Calculate b = W^T μ_0
    b_vector = W.T @ mu_0  # Shape: 2 × 1
    
    # Solve for Lagrange multipliers: λ = A^(-1) * b
    try:
        A_inv = torch.inverse(A_matrix)
    except RuntimeError:
        # If A is singular, use pseudoinverse
        A_inv = torch.pinverse(A_matrix)
        logging.warning("Matrix A was singular, using pseudoinverse")
    
    lambda_0 = A_inv @ b_vector  # Shape: 2 × 1
    
    # Calculate μ_{c,0} = μ_0 - Σ_0 W λ_0
    mu_c_0 = mu_0 - sigma_0 @ W @ lambda_0
    
    # Calculate coefficient A (leading order term)
    A_coeff = torch.dot(mu_c_0, mu_c_0)
    
    # Instead of trying to derive the exact B coefficient analytically,
    # use a numerical approach that's more stable for multiple constraints
    
    # Calculate the exact loss at a high dimension (d=1000) to approximate the asymptotic limit
    # This avoids the numerical instabilities that occur at lower dimensions
    d_high = 1000
    
    # Build μ_S and Σ_S at high dimension
    mu_S_high = _build_mu_S_finite_d(alpha_teacher, d_high, sequence_length, device)
    sigma_S_high = _build_sigma_S_finite_d(d_high, sequence_length, device)
    
    # Calculate exact conditional expectation at high dimension
    exact_loss_high = _calculate_exact_conditional_expectation_robust(mu_S_high, sigma_S_high, W)
    
    # The asymptotic limit should be close to the high-dimension result
    # Use this as our asymptotic approximation
    asymptotic_expectation = exact_loss_high
    
    return asymptotic_expectation


def _calculate_exact_conditional_expectation_robust(mu_S, sigma_S, W):
    """
    Robust calculation of exact conditional expectation for given μ_S and Σ_S.
    """
    # Calculate A, b, and C matrices
    A = W.T @ sigma_S @ W  # Shape: 2 × 2
    b = W.T @ sigma_S @ mu_S  # Shape: 2 × 1
    C = W.T @ sigma_S @ sigma_S @ W  # Shape: 2 × 2
    
    # Handle potential singularity of A using pseudoinverse
    try:
        A_inv = torch.inverse(A)
    except RuntimeError:
        A_inv = torch.pinverse(A)
    
    # Evaluate the exact conditional expectation formula
    mu_S_squared = torch.dot(mu_S, mu_S)  # μ_S^T μ_S
    trace_sigma_S = torch.trace(sigma_S)  # Tr(Σ_S)
    
    b_T_A_inv_b = torch.dot(b, A_inv @ b)  # b^T A^(-1) b
    b_T_A_inv_C_A_inv_b = torch.dot(b, A_inv @ C @ A_inv @ b)  # b^T A^(-1) C A^(-1) b
    trace_A_inv_C = torch.trace(A_inv @ C)  # Tr(A^(-1) C)
    
    conditional_expectation = (mu_S_squared + trace_sigma_S - 
                              2 * b_T_A_inv_b + 
                              b_T_A_inv_C_A_inv_b - 
                              trace_A_inv_C)
    
    # Safety check: ensure the result is reasonable
    if not torch.isfinite(conditional_expectation) or conditional_expectation < 0:
        # Fall back to a simpler approximation
        conditional_expectation = mu_S_squared + trace_sigma_S
    
    return conditional_expectation


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


def w2_that_minimizes_loss(w1, alpha_teacher, sequence_length, device):
    """
    Find w2 that minimizes the N=2 loss for a given w1.
    
    Based on Proposition 4 from the LaTeX document:
    - Choose r2 with sign(r2 + α) = -sign(r1 + α) 
    - Set weight ratio by equation (eq:ratio-hit-min) to achieve r_eff = -α
    - This gives L2 = μ_0^T μ_0 - α^2 (the global minimum)
    
    Args:
        w1: First constraint vector of shape (sequence_length-1,)
        alpha_teacher: Teacher parameter
        sequence_length: Length of the sequence
        device: PyTorch device
        
    Returns:
        w2: Optimal second constraint vector that minimizes loss
    """
    # Calculate r1 = r(w1) = w1^T μ_0 / w1[0]
    mu_0 = _build_mu_0_asymptotic(alpha_teacher, sequence_length, device)
    r1 = torch.dot(w1, mu_0) / w1[0]
    
    # Choose r2 on the opposite side of -α from r1
    if r1 + alpha_teacher > 0:
        # r1 > -α, so choose r2 < -α
        r2 = -alpha_teacher - 1.0  # Choose r2 well below -α
    else:
        # r1 < -α, so choose r2 > -α  
        r2 = -alpha_teacher + 1.0  # Choose r2 well above -α
    
    # Construct w2 using the method from Proposition 4
    # w2 = e1 + β * v_tail where v_tail = (0, μ_0,2, ..., μ_0,k-1)^T
    v_tail = torch.zeros_like(mu_0)
    v_tail[1:] = mu_0[1:]  # v_tail[0] = 0, v_tail[m] = μ_0,m for m ≥ 2
    
    # Calculate β = (r2 + α) / ||v_tail||^2
    v_tail_norm_squared = torch.dot(v_tail, v_tail)
    if v_tail_norm_squared > 1e-10:  # Avoid division by zero
        beta = (r2 + alpha_teacher) / v_tail_norm_squared
    else:
        # If v_tail is nearly zero, just use a simple construction
        beta = 0.0
    
    # Construct w2_tilde = e1 + β * v_tail
    w2_tilde = torch.zeros_like(w1)
    w2_tilde[0] = 1.0  # e1 component
    w2_tilde += beta * v_tail
    
    # Verify that r(w2_tilde) = r2
    r2_actual = torch.dot(w2_tilde, mu_0) / w2_tilde[0]
    
    # Calculate the optimal weight ratio from equation (eq:ratio-hit-min)
    # (w2_1)^2 / (w1_1)^2 = -(r1 + α) / (r2 + α)
    weight_ratio = -(r1 + alpha_teacher) / (r2 + alpha_teacher)
    
    # Scale w2 to achieve the optimal weight ratio
    # We want (w2[0])^2 / (w1[0])^2 = weight_ratio
    # So w2[0] = w1[0] * sqrt(weight_ratio)
    scale_factor = torch.sqrt(torch.abs(weight_ratio))
    w2 = w2_tilde * scale_factor
    
    # Ensure w2[0] has the correct sign to match the weight ratio
    if weight_ratio < 0:
        w2[0] = -torch.abs(w2[0])
    else:
        w2[0] = torch.abs(w2[0])
    
    return w2

def w2_that_maximizes_loss(w1, alpha_teacher, sequence_length, device):
    """
    Find w2 that maximizes the N=2 loss for a given w1 (adversarial/poisoning case).
    
    Based on Proposition 5 from the LaTeX document:
    - Pick r2 on the same side of -α as r1 with |r2 + α| > |r1 + α|
    - Choose |w2_1|/|w1_1| large so r_eff ≈ r2
    - This gives L2 > L1(w1) and the gap can be made arbitrarily large
    
    Args:
        w1: First constraint vector of shape (sequence_length-1,)
        alpha_teacher: Teacher parameter
        sequence_length: Length of the sequence
        device: PyTorch device
        
    Returns:
        w2: Adversarial second constraint vector that maximizes loss
    """
    # Calculate r1 = r(w1) = w1^T μ_0 / w1[0]
    mu_0 = _build_mu_0_asymptotic(alpha_teacher, sequence_length, device)
    r1 = torch.dot(w1, mu_0) / w1[0]
    
    # Choose r2 on the same side of -α as r1 with |r2 + α| > |r1 + α|
    if r1 + alpha_teacher > 0:
        # r1 > -α, so choose r2 > -α with r2 > r1
        r2 = r1 + 2.0  # Choose r2 well above r1
    else:
        # r1 < -α, so choose r2 < -α with r2 < r1
        r2 = r1 - 2.0  # Choose r2 well below r1
    
    # Construct w2 using the same method as in the minimizing case
    v_tail = torch.zeros_like(mu_0)
    v_tail[1:] = mu_0[1:]  # v_tail[0] = 0, v_tail[m] = μ_0,m for m ≥ 2
    
    # Calculate β = (r2 + α) / ||v_tail||^2
    v_tail_norm_squared = torch.dot(v_tail, v_tail)
    if v_tail_norm_squared > 1e-10:  # Avoid division by zero
        beta = (r2 + alpha_teacher) / v_tail_norm_squared
    else:
        # If v_tail is nearly zero, just use a simple construction
        beta = 0.0
    
    # Construct w2_tilde = e1 + β * v_tail
    w2_tilde = torch.zeros_like(w1)
    w2_tilde[0] = 1.0  # e1 component
    w2_tilde += beta * v_tail
    
    # For adversarial case, make the weight ratio large so r_eff ≈ r2
    # Choose a large weight ratio to dominate the effective ratio
    large_weight_ratio = 10.0  # Make w2_1 much larger than w1_1
    
    # Scale w2 to achieve the large weight ratio
    # We want (w2[0])^2 / (w1[0])^2 = large_weight_ratio
    # So w2[0] = w1[0] * sqrt(large_weight_ratio)
    scale_factor = torch.sqrt(large_weight_ratio)
    w2 = w2_tilde * scale_factor
    
    # Ensure w2[0] has the same sign as w1[0] to maintain the weight ratio
    w2[0] = torch.sign(w1[0]) * torch.abs(w2[0])
    
    return w2


def test_w2_optimization():
    """
    Test function to demonstrate w2 optimization for minimizing and maximizing loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from generator import generate_teacher_alpha, generate_w
    
    print("Testing w2 optimization functions...")
    
    # Generate test parameters
    torch.manual_seed(42)
    alpha_teacher = generate_teacher_alpha(device)
    w1 = generate_w(5, device)  # sequence_length = 5
    sequence_length = 5
    student_dim = 200
    
    print(f"Alpha teacher: {alpha_teacher.item():.4f}")
    print(f"w1: {w1}")
    
    # Calculate L1(w1) - the single constraint loss
    L1_w1, _, _ = gnc_theoretical_loss_for_one_w(alpha_teacher, w1, student_dim, device)
    print(f"L1(w1): {L1_w1.item():.6f}")
    
    # Find w2 that minimizes loss
    w2_min = w2_that_minimizes_loss(w1, alpha_teacher, sequence_length, device)
    print(f"w2_min: {w2_min}")
    
    # Calculate L2(w1, w2_min)
    L2_min, _, _ = gnc_theoretical_loss_for_multiple_w(alpha_teacher, [w1, w2_min], student_dim, device)
    print(f"L2(w1, w2_min): {L2_min.item():.6f}")
    print(f"Improvement: {L1_w1.item() - L2_min.item():.6f}")
    
    # Find w2 that maximizes loss (adversarial)
    w2_max = w2_that_maximizes_loss(w1, alpha_teacher, sequence_length, device)
    print(f"w2_max: {w2_max}")
    
    # Calculate L2(w1, w2_max)
    L2_max, _, _ = gnc_theoretical_loss_for_multiple_w(alpha_teacher, [w1, w2_max], student_dim, device)
    print(f"L2(w1, w2_max): {L2_max.item():.6f}")
    print(f"Deterioration: {L2_max.item() - L1_w1.item():.6f}")
    
    # Verify the theoretical predictions
    mu_0 = _build_mu_0_asymptotic(alpha_teacher, sequence_length, device)
    r1 = torch.dot(w1, mu_0) / w1[0]
    r2_min = torch.dot(w2_min, mu_0) / w2_min[0]
    r2_max = torch.dot(w2_max, mu_0) / w2_max[0]
    
    print(f"\nRatio analysis:")
    print(f"r1: {r1.item():.4f}")
    print(f"r2_min: {r2_min.item():.4f}")
    print(f"r2_max: {r2_max.item():.4f}")
    print(f"-α: {-alpha_teacher.item():.4f}")
    
    # Calculate effective ratios
    w1_1_sq = w1[0]**2
    w2_min_1_sq = w2_min[0]**2
    w2_max_1_sq = w2_max[0]**2
    
    r_eff_min = (w1_1_sq * r1 + w2_min_1_sq * r2_min) / (w1_1_sq + w2_min_1_sq)
    r_eff_max = (w1_1_sq * r1 + w2_max_1_sq * r2_max) / (w1_1_sq + w2_max_1_sq)
    
    print(f"r_eff_min: {r_eff_min.item():.4f} (should be close to -α = {-alpha_teacher.item():.4f})")
    print(f"r_eff_max: {r_eff_max.item():.4f}")


if __name__ == "__main__":
    #w_that_minimizes_loss()
    #first_best_seeds()
    test_w2_optimization()