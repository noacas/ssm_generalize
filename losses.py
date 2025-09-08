import torch


def get_losses_original(A_diag: torch.Tensor, w: list[torch.Tensor], alpha_teacher: float):
    """
    Original implementation for comparison.
    """
    # Ensure batch dimension on A
    if A_diag.dim() == 1:
        A_diag = A_diag.unsqueeze(0)  # (batch_size, state_dim)

    device = A_diag.device
    dtype = A_diag.dtype

    M = w[0].size(-1)

    # Vectorized computation using cumulative products over the power dimension
    # X: (batch, state_dim, M) with every slice along M equal to A_diag
    X = A_diag.unsqueeze(-1).expand(-1, -1, M)
    # A_pows[:, :, k] = A_diag ** (k+1)
    A_pows = torch.cumprod(X, dim=2)
    # Sum over state dimensions → (batch, M)
    sum_A_pows = A_pows.sum(dim=1)

    # alpha powers: [alpha, alpha^2, ..., alpha^M]
    alpha_scalar = torch.as_tensor(alpha_teacher, device=device, dtype=dtype)
    alpha_base_vec = torch.full((M,), alpha_scalar.item(), device=device, dtype=dtype)
    alpha_pows = torch.cumprod(alpha_base_vec, dim=0)

    # s[:, m-1] = sum_j A_j^m - alpha^m
    s = sum_A_pows - alpha_pows.unsqueeze(0)

    train_loss = 0
    # Move w to the same device and expand to batch if needed
    for w_i in w:
        w_i = w_i.to(device)
        if w_i.dim() == 1:
            w_expanded = w_i.unsqueeze(0).expand(A_diag.size(0), -1)
        elif w_i.dim() == 2:
            if w.size(0) == 1:
                w_expanded = w.expand(A_diag.size(0), -1)
            elif w.size(0) == A_diag.size(0):
                w_expanded = w_i
            else:
                raise ValueError("w has incompatible batch dimension. Expected 1 or batch_size")
        else:
            raise ValueError("w must be 1D or 2D tensor")

        # Compute losses
        # train_loss_i = (<s_i, w>)**2
        dot_sw = (s * w_expanded.to(dtype)).sum(dim=1)
        train_loss += dot_sw.pow(2)

    train_loss /= len(w)

    # gen_loss_i = ||s_i||_2^2
    gen_loss = (s * s).sum(dim=1)

    return train_loss, gen_loss


def get_losses(A_diag: torch.Tensor, w: list[torch.Tensor], alpha_teacher: float):
    """
    Compute per-sample training and generalization losses for a diagonal SSM.
    Optimized version with reduced memory allocations and better vectorization.

    Args:
        A_diag: Tensor of shape (state_dim,) or (batch_size, state_dim) with diagonal entries of A.
        w:      list of Tensors of shape (sequence_length_minus_1,) or (batch_size, sequence_length_minus_1).
        alpha_teacher: Scalar (float or 0-dim tensor) teacher parameter.

    Returns:
        train_loss: Tensor of shape (batch_size,) with (<s_i, w>)^2 for each sample.
        gen_loss:   Tensor of shape (batch_size,) with ||s_i||_2^2 for each sample.
    """

    # Ensure batch dimension on A
    if A_diag.dim() == 1:
        A_diag = A_diag.unsqueeze(0)  # (batch_size, state_dim)

    device = A_diag.device
    dtype = A_diag.dtype
    batch_size, state_dim = A_diag.shape
    M = w[0].size(-1)

    # Optimized computation with reduced memory allocations
    # Instead of creating large intermediate tensors, compute cumprod in-place where possible
    
    # Compute A powers more efficiently
    # A_pows[:, :, k] = A_diag ** (k+1) for k in [0, M-1]
    A_pows = torch.empty(batch_size, state_dim, M, device=device, dtype=dtype)
    A_pows[:, :, 0] = A_diag  # A^1
    for k in range(1, M):
        A_pows[:, :, k] = A_pows[:, :, k-1] * A_diag  # A^(k+1)
    
    # Sum over state dimensions → (batch, M)
    sum_A_pows = A_pows.sum(dim=1)

    # Pre-compute alpha powers more efficiently
    alpha_pows = torch.empty(M, device=device, dtype=dtype)
    alpha_pows[0] = alpha_teacher
    for k in range(1, M):
        alpha_pows[k] = alpha_pows[k-1] * alpha_teacher

    # s[:, m-1] = sum_j A_j^m - alpha^m
    s = sum_A_pows - alpha_pows.unsqueeze(0)

    # Optimized train loss computation
    train_loss = torch.zeros(batch_size, device=device, dtype=dtype)
    
    # Pre-allocate w_expanded to avoid repeated allocations
    w_expanded = torch.empty(batch_size, M, device=device, dtype=dtype)
    
    for w_i in w:
        w_i = w_i.to(device, dtype=dtype)
        if w_i.dim() == 1:
            w_expanded[:] = w_i.unsqueeze(0)  # Broadcast to batch
        elif w_i.dim() == 2:
            if w_i.size(0) == 1:
                w_expanded[:] = w_i.expand(batch_size, -1)
            elif w_i.size(0) == batch_size:
                w_expanded[:] = w_i
            else:
                raise ValueError("w has incompatible batch dimension. Expected 1 or batch_size")
        else:
            raise ValueError("w must be 1D or 2D tensor")

        # Compute dot product more efficiently
        dot_sw = torch.sum(s * w_expanded, dim=1)
        train_loss += dot_sw.pow(2)

    train_loss /= len(w)

    # Optimized gen_loss computation
    gen_loss = torch.sum(s * s, dim=1)

    return train_loss, gen_loss


def test_get_losses():
    A_diag = torch.tensor([[0.55, 0.45, 0.4], [0.3, 0.4, 0.3]])
    w = [torch.tensor([0.5, 0.7, 0.3, 0.4]), torch.tensor([0.5, 0.7, 0.3, 0.4])]
    alpha_teacher = 0.5
    train_loss, gen_loss = get_losses(A_diag, w, alpha_teacher)
    s1 = torch.tensor([A_diag[0].sum()-0.5, (A_diag[0]**2).sum()-0.5**2, (A_diag[0]**3).sum()-0.5**3, (A_diag[0]**4).sum()-0.5**4])
    s2 = torch.tensor([A_diag[1].sum()-0.5, (A_diag[1]**2).sum()-0.5**2, (A_diag[1]**3).sum()-0.5**3, (A_diag[1]**4).sum()-0.5**4])
    assert torch.allclose(train_loss, torch.tensor([torch.dot(s1, w[0])**2, torch.dot(s2, w[0])**2]))
    assert torch.allclose(gen_loss, torch.tensor([torch.dot(s1, s1), torch.dot(s2, s2)]))

if __name__ == "__main__":
    test_get_losses()