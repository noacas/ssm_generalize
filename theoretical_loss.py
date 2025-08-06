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
    
    return conditional_expectation 