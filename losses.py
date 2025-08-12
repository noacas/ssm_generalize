import torch


def get_losses(A_diag: torch.Tensor, w: torch.Tensor, alpha_teacher: float):
    """
    Compute per-sample training and generalization losses for a diagonal SSM.

    Args:
        A_diag: Tensor of shape (state_dim,) or (batch_size, state_dim) with diagonal entries of A.
        w:      Tensor of shape (sequence_length_minus_1,) or (batch_size, sequence_length_minus_1).
        alpha_teacher: Scalar (float or 0-dim tensor) teacher parameter.

    Returns:
        train_loss: Tensor of shape (batch_size,) with dot(s_i, w) for each sample.
        gen_loss:   Tensor of shape (batch_size,) with ||s_i||_2^2 for each sample.
    """

    # Ensure batch dimension on A
    if A_diag.dim() == 1:
        A_diag = A_diag.unsqueeze(0)  # (batch_size, state_dim)

    device = A_diag.device
    dtype = A_diag.dtype

    # Move w to the same device and expand to batch if needed
    w = w.to(device)
    if w.dim() == 1:
        M = w.numel()
        w_expanded = w.unsqueeze(0).expand(A_diag.size(0), -1)
    elif w.dim() == 2:
        M = w.size(1)
        if w.size(0) == 1:
            w_expanded = w.expand(A_diag.size(0), -1)
        elif w.size(0) == A_diag.size(0):
            w_expanded = w
        else:
            raise ValueError("w has incompatible batch dimension. Expected 1 or batch_size")
    else:
        raise ValueError("w must be 1D or 2D tensor")

    # Prepare sequence statistics: for each m, use the m-th (clamped) diagonal entry
    # s[:, m-1] = A_{min(m, state_dim)}^m - alpha_teacher^m for m=1..M
    s = torch.empty((A_diag.size(0), M), device=device, dtype=dtype)

    state_dim = A_diag.size(1)
    alpha_current = torch.as_tensor(alpha_teacher, device=device, dtype=dtype)

    for m in range(1, M + 1):
        col_index = min(m, state_dim) - 1
        base = A_diag[:, col_index]
        s[:, m - 1] = base.pow(m) - alpha_current

        # Update alpha term
        alpha_current = alpha_current * alpha_teacher

    # Compute losses
    # train_loss_i = <s_i, w>
    train_loss = (s * w_expanded.to(dtype)).sum(dim=1)

    # gen_loss_i = ||s_i||_2^2
    gen_loss = (s * s).sum(dim=1)

    return train_loss, gen_loss


def test_get_losses():
    A_diag = torch.tensor([[0.55, 0.45, 0.4], [0.3, 0.4, 0.3]])
    w = torch.tensor([0.5, 0.7, 0.3, 0.4])
    alpha_teacher = 0.5
    train_loss, gen_loss = get_losses(A_diag, w, alpha_teacher)
    s1 = torch.tensor([0.55-0.5, 0.45**2-0.5**2, 0.4**3-0.5**3, 0.4**4-0.5**4])
    s2 = torch.tensor([0.3-0.5, 0.4**2-0.5**2, 0.3**3-0.5**3, 0.3**4-0.5**4])
    assert torch.allclose(train_loss, torch.tensor([torch.dot(s1, w), torch.dot(s2, w)]))
    assert torch.allclose(gen_loss, torch.tensor([torch.dot(s1, s1), torch.dot(s2, s2)]))

if __name__ == "__main__":
    test_get_losses()