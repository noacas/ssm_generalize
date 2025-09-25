import math
import torch
import logging
import numpy as np


def double_factorial(n: int) -> int:
    """Calculate double factorial n!! = n * (n-2) * (n-4) * ... * 1 for odd n"""
    if n == 1:
        return 1

    k = (n+1) // 2
    return math.factorial(n) / (2**(k-1) * math.factorial(k-1))

def J(m: int) -> int:
    """J_m = (m-1)!! if m is even, else 0. (Note: (m-1)!! over odds.)"""
    if m % 2 == 1:
        return 0  # odd m → 0
    # m even: (m-1)!! = (2n-1)!! with n = m/2
    n = m // 2
    # closed form: (2n-1)!! = (2n)! / (2^n n!)
    return math.factorial(2*n) // (2**n * math.factorial(n))


def a_m_expectation(student_dim, m):
    # return E[a**m] for a ~ N(0, 1/d)
    return student_dim**(-m/2) * double_factorial(m-1)  if m % 2 == 0 else 0


def calc_mu(student_dim, alpha_teacher, sequence_length, device):
    # calculate the expected value of the student's output minus the teacher for the impulse response input
    # result is a vector of length sequence length - 1
    mu = torch.empty(sequence_length - 1, device=device)
    for m in range(1, sequence_length):
        mu[m-1] = student_dim * a_m_expectation(student_dim, m) - alpha_teacher**m
    return mu


def calc_sigma(student_dim, sequence_length, device):
    # covariance matrix of the student's output minus the teacher for the impulse response input
    sigma = torch.empty((sequence_length - 1, sequence_length - 1), device=device)
    for m in range(1, sequence_length-1):
        for n in range(1, sequence_length-1):
            sigma[m-1, n-1] = student_dim * (a_m_expectation(student_dim, m+n) - a_m_expectation(student_dim, m) * a_m_expectation(student_dim, n))
    return sigma


def build_mu_Sigma(alpha: float, d: int, k: int, device=None):
    """
    Build finite-d mean vector mu (size k-1) and covariance Sigma ((k-1)x(k-1))
    for S = (S_1,...,S_{k-1}) with a_i ~ N(0, 1/d).

    (mu)_m = d^(1 - m/2) * J_m - alpha^m
    (Sigma)_{mn} = d^(1 - (m+n)/2) * (J_{m+n} - J_m * J_n)
    for m,n in {1,...,k-1}.
    """
    mvals = list(range(1, k))
    # mean
    mu_list = [(d ** (1 - m / 2)) * J(m) - (alpha ** m) for m in mvals]
    mu = torch.tensor(mu_list, device=device)

    # covariance (k-1 is tiny, loops are fine and clearer)
    Sigma = torch.empty((k - 1, k - 1), device=device)
    for i, m in enumerate(mvals):
        Jm = J(m)
        for j, n in enumerate(mvals):
            Sigma[i, j] = (d ** (1 - (m + n) / 2)) * (J(m + n) - Jm * J(n))
    return mu, Sigma


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


def gnc_theoretical_loss_for_one_w(alpha_teacher, w, student_dim, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def gnc_theoretical_loss_for_multiple_w(alpha_teacher, W, student_dim, k, device):
    """
    Exact finite-d Gaussian hard-constraint predictor:
        S ~ N(mu, Sigma), condition on W^T S = 0
        mu_c   = mu   - Sigma W (W^T Sigma W)^{-1} W^T mu
        Sigma_c= Sigma- Sigma W (W^T Sigma W)^{-1} W^T Sigma
        Loss   = ||mu_c||^2 + Tr(Sigma_c)

    Args:
        alpha: teacher alpha
        d:     student dimension
        k:     sequence length (S has length k-1)
        W:     (k-1, N) tensor; columns are constraint vectors w^(n)

    Returns:
        scalar torch.Tensor: predicted conditional loss
    """
    device = W.device
    mu, Sigma = build_mu_Sigma(alpha_teacher, student_dim, k, device=device)

    # A = W^T Sigma W  (N x N), robust inverse via pinv
    A = W.T @ Sigma @ W
    Ainv = torch.linalg.pinv(A)

    # Conditional mean/covariance
    mu_c    = mu   - Sigma @ W @ (Ainv @ (W.T @ mu))
    Sigma_c = Sigma - Sigma @ W @ (Ainv @ (W.T @ Sigma))

    return mu_c @ mu_c + torch.trace(Sigma_c)


def gnc_theoretical_loss_for_multiple_w_asymptotic(alpha_teacher, W, student_dim, k, device):
    """
    Asymptotic calculation of theoretical loss for multiple sequences.
    """
    m0 = _build_mu_0_asymptotic(alpha_teacher, k, device)
    r = (W.T @ m0) / W[0]   # shape: (N,)
    lam = (W[0]**2)
    lam = lam / lam.sum()
    r_eff = (lam * r).sum()
    mu_0_squared = torch.dot(m0, m0)
    return mu_0_squared - alpha_teacher**2 + (r_eff + alpha_teacher)**2


def gnc_theoretical_loss(alpha_teacher, w_sequences, student_dim, device):
    """
    Main function to calculate theoretical loss for either single or multiple sequences.
    """
    if isinstance(w_sequences, torch.Tensor):
        # Single sequence case
        return gnc_theoretical_loss_for_one_w(alpha_teacher, w_sequences, student_dim, device)
    elif isinstance(w_sequences, list):
        # Multiple sequences case
        k = w_sequences[0].numel() + 1
        W = torch.stack([w.reshape(-1) for w in w_sequences], dim=1).to(device)  # (k-1, N)
        return gnc_theoretical_loss_for_multiple_w(alpha_teacher, W, student_dim, k, device), gnc_theoretical_loss_for_multiple_w_asymptotic(alpha_teacher, W, student_dim, k, device), None
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


# def w2_that_minimizes_loss(w1, alpha_teacher, sequence_length, device):
#     """
#     Find w2 that minimizes the N=2 loss for a given w1.
    
#     Based on Proposition 4 from the LaTeX document:
#     - Choose r2 with sign(r2 + α) = -sign(r1 + α) 
#     - Set weight ratio by equation (eq:ratio-hit-min) to achieve r_eff = -α
#     - This gives L2 = μ_0^T μ_0 - α^2 (the global minimum)
    
#     Args:
#         w1: First constraint vector of shape (sequence_length-1,)
#         alpha_teacher: Teacher parameter
#         sequence_length: Length of the sequence
#         device: PyTorch device
        
#     Returns:
#         w2: Optimal second constraint vector that minimizes loss
#     """
#     # Calculate r1 = r(w1) = w1^T μ_0 / w1[0]
#     mu_0 = _build_mu_0_asymptotic(alpha_teacher, sequence_length, device)
#     r1 = torch.dot(w1, mu_0) / w1[0]
    
#     # Choose r2 on the opposite side of -α from r1
#     if r1 + alpha_teacher > 0:
#         # r1 > -α, so choose r2 < -α
#         r2 = -alpha_teacher - 1.0  # Choose r2 well below -α
#     else:
#         # r1 < -α, so choose r2 > -α  
#         r2 = -alpha_teacher + 1.0  # Choose r2 well above -α
    
#     # Construct w2 using the method from Proposition 4
#     # w2 = e1 + β * v_tail where v_tail = (0, μ_0,2, ..., μ_0,k-1)^T
#     v_tail = torch.zeros_like(mu_0)
#     v_tail[1:] = mu_0[1:]  # v_tail[0] = 0, v_tail[m] = μ_0,m for m ≥ 2
    
#     # Calculate β = (r2 + α) / ||v_tail||^2
#     v_tail_norm_squared = torch.dot(v_tail, v_tail)
#     if v_tail_norm_squared > 1e-10:  # Avoid division by zero
#         beta = (r2 + alpha_teacher) / v_tail_norm_squared
#     else:
#         # If v_tail is nearly zero, just use a simple construction
#         beta = 0.0
    
#     # Construct w2_tilde = e1 + β * v_tail
#     w2_tilde = torch.zeros_like(w1)
#     w2_tilde[0] = 1.0  # e1 component
#     w2_tilde += beta * v_tail
    
#     # Verify that r(w2_tilde) = r2
#     r2_actual = torch.dot(w2_tilde, mu_0) / w2_tilde[0]
    
#     # Calculate the optimal weight ratio from equation (eq:ratio-hit-min)
#     # (w2_1)^2 / (w1_1)^2 = -(r1 + α) / (r2 + α)
#     weight_ratio = -(r1 + alpha_teacher) / (r2 + alpha_teacher)
    
#     # Scale w2 to achieve the optimal weight ratio
#     # We want (w2[0])^2 / (w1[0])^2 = weight_ratio
#     # So w2[0] = w1[0] * sqrt(weight_ratio)
#     if weight_ratio > 0:
#         scale_factor = torch.sqrt(weight_ratio)
#         w2 = w2_tilde * scale_factor
#         # Ensure w2[0] has the same sign as w1[0]
#         w2[0] = torch.sign(w1[0]) * torch.abs(w2[0])
#     else:
#         # If weight_ratio is negative, we need to handle this case
#         # This shouldn't happen if r1 and r2 are on opposite sides of -α
#         logging.warning(f"Negative weight ratio: {weight_ratio}, r1={r1}, r2={r2}, alpha={alpha_teacher}")
#         # Fall back to a simple scaling
#         w2 = w2_tilde * torch.abs(w1[0]) / torch.abs(w2_tilde[0])
    
#     # Verify the effective ratio is close to -α
#     r_eff_actual = (w1[0]**2 * r1 + w2[0]**2 * r2) / (w1[0]**2 + w2[0]**2)
#     logging.info(f"Target r_eff: {-alpha_teacher.item():.4f}, Actual r_eff: {r_eff_actual.item():.4f}")
    
#     return w2

# def w2_that_maximizes_loss(w1, alpha_teacher, sequence_length, device):
#     """
#     Find w2 that maximizes the N=2 loss for a given w1 (adversarial/poisoning case).
    
#     Based on Proposition 5 from the LaTeX document:
#     - Pick r2 on the same side of -α as r1 with |r2 + α| > |r1 + α|
#     - Choose |w2_1|/|w1_1| large so r_eff ≈ r2
#     - This gives L2 > L1(w1) and the gap can be made arbitrarily large
    
#     Args:
#         w1: First constraint vector of shape (sequence_length-1,)
#         alpha_teacher: Teacher parameter
#         sequence_length: Length of the sequence
#         device: PyTorch device
        
#     Returns:
#         w2: Adversarial second constraint vector that maximizes loss
#     """
#     # Calculate r1 = r(w1) = w1^T μ_0 / w1[0]
#     mu_0 = _build_mu_0_asymptotic(alpha_teacher, sequence_length, device)
#     r1 = torch.dot(w1, mu_0) / w1[0]
    
#     # Choose r2 on the same side of -α as r1 with |r2 + α| > |r1 + α|
#     if r1 + alpha_teacher > 0:
#         # r1 > -α, so choose r2 > -α with r2 > r1
#         r2 = r1 + 2.0  # Choose r2 well above r1
#     else:
#         # r1 < -α, so choose r2 < -α with r2 < r1
#         r2 = r1 - 2.0  # Choose r2 well below r1
    
#     # Construct w2 using the same method as in the minimizing case
#     v_tail = torch.zeros_like(mu_0)
#     v_tail[1:] = mu_0[1:]  # v_tail[0] = 0, v_tail[m] = μ_0,m for m ≥ 2
    
#     # Calculate β = (r2 + α) / ||v_tail||^2
#     v_tail_norm_squared = torch.dot(v_tail, v_tail)
#     if v_tail_norm_squared > 1e-10:  # Avoid division by zero
#         beta = (r2 + alpha_teacher) / v_tail_norm_squared
#     else:
#         # If v_tail is nearly zero, just use a simple construction
#         beta = 0.0
    
#     # Construct w2_tilde = e1 + β * v_tail
#     w2_tilde = torch.zeros_like(w1)
#     w2_tilde[0] = 1.0  # e1 component
#     w2_tilde += beta * v_tail
    
#     # For adversarial case, make the weight ratio large so r_eff ≈ r2
#     # Choose a large weight ratio to dominate the effective ratio
#     large_weight_ratio = 100.0  # Make w2_1 much larger than w1_1
    
#     # Scale w2 to achieve the large weight ratio
#     # We want (w2[0])^2 / (w1[0])^2 = large_weight_ratio
#     # So w2[0] = w1[0] * sqrt(large_weight_ratio)
#     scale_factor = torch.sqrt(torch.tensor(large_weight_ratio, device=device))
#     w2 = w2_tilde * scale_factor
    
#     # Ensure w2[0] has the same sign as w1[0] to maintain the weight ratio
#     w2[0] = torch.sign(w1[0]) * torch.abs(w2[0])
    
#     # Verify the effective ratio is close to r2
#     r_eff_actual = (w1[0]**2 * r1 + w2[0]**2 * r2) / (w1[0]**2 + w2[0]**2)
#     logging.info(f"Adversarial r_eff: {r_eff_actual.item():.4f} (should be close to r2 = {r2.item():.4f})")
    
#     return w2


def calculate_asymptotic_loss(r_eff, alpha_teacher, sequence_length, device):
    """
    Calculate the asymptotic loss using the formula: f(r) = μ_0^T μ_0 - α^2 + (r + α)^2
    """
    mu_0 = _build_mu_0_asymptotic(alpha_teacher, sequence_length, device)
    mu_0_squared = torch.dot(mu_0, mu_0)
    return mu_0_squared - alpha_teacher**2 + (r_eff + alpha_teacher)**2


# def test_w2_optimization():
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
    
    # Calculate theoretical asymptotic losses
    L1_theoretical = calculate_asymptotic_loss(r1, alpha_teacher, sequence_length, device)
    L2_min_theoretical = calculate_asymptotic_loss(r_eff_min, alpha_teacher, sequence_length, device)
    L2_max_theoretical = calculate_asymptotic_loss(r_eff_max, alpha_teacher, sequence_length, device)
    
    print(f"\nTheoretical asymptotic losses:")
    print(f"L1_theoretical: {L1_theoretical.item():.6f}")
    print(f"L2_min_theoretical: {L2_min_theoretical.item():.6f}")
    print(f"L2_max_theoretical: {L2_max_theoretical.item():.6f}")
    
    # Additional debugging information
    print(f"\nDebugging info:")
    print(f"w1[0]: {w1[0].item():.4f}")
    print(f"w2_min[0]: {w2_min[0].item():.4f}")
    print(f"w2_max[0]: {w2_max[0].item():.4f}")
    print(f"Weight ratio for min: {(w2_min[0]/w1[0])**2:.4f}")
    print(f"Weight ratio for max: {(w2_max[0]/w1[0])**2:.4f}")
    
    # Check if the conditions are met
    print(f"\nCondition checks:")
    print(f"r1 + α = {r1.item() + alpha_teacher.item():.4f}")
    print(f"r2_min + α = {r2_min.item() + alpha_teacher.item():.4f}")
    print(f"r2_max + α = {r2_max.item() + alpha_teacher.item():.4f}")
    print(f"Opposite signs for min: {(r1 + alpha_teacher) * (r2_min + alpha_teacher) < 0}")
    print(f"Same side for max: {(r1 + alpha_teacher) * (r2_max + alpha_teacher) > 0}")


if __name__ == "__main__":
    #w_that_minimizes_loss()
    #first_best_seeds()
    #test_w2_optimization()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = 5
    d = 70
    # seed 3
    alpha_teacher = 0.5426
    w1 = torch.tensor([-0.5162, -0.2217, -0.5594,  0.5542], device=device)
    w2 = torch.tensor([-0.7082, -2.1120,  0.3220, -0.9507], device=device)
    L2, L2_asymptotic, _ = gnc_theoretical_loss(alpha_teacher, [w1, w2], d, device)
    print("N=2 predicted loss:", L2.item(), L2_asymptotic.item())
    L2_large_d, _, _ = gnc_theoretical_loss(alpha_teacher, [w1, w2], 10000, device)
    print("for large d, the predicted loss is:", L2_large_d.item())
    # seed 2
    alpha_teacher = 0.5022
    w1 = torch.tensor([ 2.2669,  1.3477, -1.4438, -1.0484], device=device)
    w2 = torch.tensor([-0.1825,  1.7087,  0.1843, -0.6569], device=device)
    L2, L2_asymptotic, _ = gnc_theoretical_loss(alpha_teacher, [w1, w2], d, device)
    print("N=2 predicted loss:", L2.item(), L2_asymptotic.item())
