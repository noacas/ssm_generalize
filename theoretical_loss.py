import math
import torch


def double_factorial(n):
    """Calculate double factorial n!! = n * (n-2) * (n-4) * ... * 1 for odd n"""
    if n == 1:
        return 1

    k = (n+1) // 2
    return math.factorial(n) / (2**(k-1) * math.factorial(k-1))


def a_m_expectation(student_dim, m):
    # return E[a**m] for a ~ N(0, 1/d)
    return student_dim**(-m/2) * double_factorial(m-1)  if m % 2 == 0 else 0


def teacher_for_m(teacher, m):
    # teacher is a tuple of (A_teacher, B_teacher, C_teacher)
    A_teacher, B_teacher, C_teacher = teacher
    # Calculate teacher's m-th power response
    # This would need to be implemented based on the specific teacher structure
    # For now, using a placeholder that sums the teacher's parameters
    return (A_teacher**m).sum()


def calc_mu(student_dim, teacher, sequence_length, device):
    # calculate the expected value of the student's output minus the teacher for the impulse response input
    # result is a vector of length sequence length - 1
    mu = torch.empty(sequence_length - 1, device=device)
    for m in range(1, sequence_length):
        mu[m-1] = student_dim * a_m_expectation(student_dim, m) - teacher_for_m(teacher, m)
    return mu


def calc_sigma(student_dim, teacher, sequence_length, device):
    # covariance matrix of the student's output minus the teacher for the impulse response input
    sigma = torch.empty((sequence_length - 1, sequence_length - 1), device=device)
    for m in range(1, sequence_length-1):
        for n in range(1, sequence_length-1):
            sigma[m-1, n-1] = student_dim * (a_m_expectation(student_dim, m+n) - a_m_expectation(student_dim, m) * a_m_expectation(student_dim, n))
    return sigma


def calc_asymptotic_coefficients(teacher, w, sequence_length, device):
    """
    Calculate asymptotic coefficients A and B for the loss expansion:
    L(d) = A + B/d + O(1/d^2)
    
    Based on the asymptotic expansions of μ_S and Σ_S:
    μ_S(d) = μ_0 + μ_1/d + O(1/d^2)
    Σ_S(d) = Σ_0 + Σ_1/d + O(1/d^(3/2))
    """
    # Extract teacher parameters (assuming teacher is a tuple of (A_teacher, B_teacher, C_teacher))
    A_teacher, _, _ = teacher
    # for simplicity, we'll assume α = A_teacher[0] (teacher of rank 1)
    alpha = A_teacher[0]
    
    # Calculate μ_0 and μ_1
    mu_0 = torch.empty(sequence_length - 1, device=device)
    mu_1 = torch.empty(sequence_length - 1, device=device)
    
    for m in range(1, sequence_length):
        if m % 2 == 1:  # odd m
            mu_0[m-1] = -alpha**m
        elif m == 2:  # m = 2
            mu_0[m-1] = 1 - alpha**m
        else:  # m >= 4, even
            mu_0[m-1] = -alpha**m
        
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
    w_transpose_mu_0 = torch.dot(w.squeeze(), mu_0)
    w_transpose_sigma_0_w = torch.dot(w.squeeze(), sigma_0 @ w.squeeze())  # This should be w_1^2
    F_0 = w_transpose_mu_0 / w_transpose_sigma_0_w
    
    mu_c_0 = mu_0 - F_0 * (sigma_0 @ w.squeeze())
    
    # Calculate coefficient A
    A = torch.dot(mu_c_0, mu_c_0)
    
    # Calculate coefficient B components
    # First component: 2 * μ_{c,0}^T * μ_{c,1}
    w_transpose_mu_1 = torch.dot(w.squeeze(), mu_1)
    w_transpose_sigma_1_w = torch.dot(w.squeeze(), sigma_1 @ w.squeeze())
    F_1 = w_transpose_mu_1 / w_transpose_sigma_0_w - (w_transpose_mu_0 * w_transpose_sigma_1_w) / (w_transpose_sigma_0_w**2)
    
    mu_c_1 = mu_1 - F_0 * (sigma_1 @ w.squeeze()) - F_1 * (sigma_0 @ w.squeeze())
    
    # Second component: Tr(Σ_1) - correction terms
    sigma_0_sigma_1_w = sigma_0 @ sigma_1 @ w.squeeze()
    sigma_1_sigma_0_w = sigma_1 @ sigma_0 @ w.squeeze()
    w_transpose_sigma_0_sigma_1_w = torch.dot(w.squeeze(), sigma_0_sigma_1_w)
    w_transpose_sigma_1_sigma_0_w = torch.dot(w.squeeze(), sigma_1_sigma_0_w)
    
    variance_component = torch.trace(sigma_1) - (w_transpose_sigma_0_sigma_1_w + w_transpose_sigma_1_sigma_0_w - w_transpose_sigma_1_w) / w_transpose_sigma_0_w
    
    B = 2 * torch.dot(mu_c_0, mu_c_1) + variance_component
    
    return A, B


def gnc_theoretical_loss(teacher, dataset, student_dim, device):
    # only training set, without the last input of each sequence in reverse
    # When input_e1=False, dataset has shape (num_measurements, sequence_length, 1) where index 0 is random data and index 1 is impulse response
    # When input_e1=True, dataset has shape (1, sequence_length, 1) and is just the impulse response
    # In both cases, we want the first measurement without the last time step, reversed
    w = torch.flip(dataset[0, :-1, :], dims=[0])
    
    # Get sequence length from dataset
    sequence_length = dataset.shape[1]

    mu = calc_mu(student_dim, teacher, sequence_length, device)
    sigma = calc_sigma(student_dim, teacher, sequence_length, device)
    
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
    A, B = calc_asymptotic_coefficients(teacher, w, sequence_length, device)
    
    # Asymptotic conditional expectation: A + B/d
    asymptotic_conditional_expectation = A + B / student_dim
    
    return conditional_expectation, asymptotic_conditional_expectation


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from generator import generate_teacher, generate_dataset
    for d in range(15, 40):
        torch.manual_seed(0)
        teacher = generate_teacher(1, d, device)
        dataset = generate_dataset(1, 5, False, device)
        exact_loss, asymptotic_loss = gnc_theoretical_loss(teacher, dataset, d, device)
        print(f"d={d}: Exact={exact_loss.item():.6f}, Asymptotic={asymptotic_loss.item():.6f}")
