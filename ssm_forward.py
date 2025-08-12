import torch

def ssm_forward(A_diag, x, alpha_teacher):
    # if no batch dimension is provided, add it
    if A_diag.dim() == 1:
        A_diag = A_diag.unsqueeze(0)  # (batch_size, state_dim)

    device = x.device
    # s should be a matrix of x.shape with the entry in the m-th position being tr(A_diag**(m+1)) - alpha_teacher**(m+1) keeping the batches separate
    result = torch.empty(x.shape[0], x.shape[1], device=x.device)
    temp_A = A_diag.clone(device=device)
    temp_alpha = alpha_teacher
    s[:, 0] = torch.trace(temp, dim1=-2, dim2=-1) - temp_alpha
    for m in range(1, x.shape[1]):
        # temp_A is a batch matrix of size (batch_size, state_dim) where temp_A[i, j] at time t is A_diag[i, j]**(t+1)
        temp_A = torch.einsum("bi,bj->bij", temp_A, A_diag)
        temp_alpha = temp_alpha * alpha_teacher
        s[:, m] = torch.trace(temp_A, dim1=-2, dim2=-1) - temp_alpha
    sx = torch.einsum("bi,mj->bmi", s, x)
    return 

