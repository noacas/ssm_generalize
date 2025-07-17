import torch

def ssm_forward(A_diag, B, C, x):
    """
    Forward pass through State Space Model.

    Args:
        A_diag: The Diagonal of state transition matrix (batch_size, state_dim)
        B: Input matrix (batch_size, input_dim, state_dim)
        C: Output matrix (batch_size, state_dim, output_dim)
        x: Input sequence (num_measurements, sequence_length, input_dim)

    Returns:
        y: Output sequence (batch_size, num_measurements, sequence_length, output_dim)
    """
    # if no batch dimension is provided, add it
    if A_diag.dim() == 1:
        A_diag = A_diag.unsqueeze(0)  # (1, state_dim)
    if B.dim() == 2:
        B = B.unsqueeze(0)
    if C.dim() == 2:
        C = C.unsqueeze(0)

    # x:           (num_measurements, seq_len, input_dim)
    # A_diag:      (batch_size, state_dim)                  – diagonal of A
    # B:           (batch_size, input_dim,  state_dim)      – input matrix
    # C:           (batch_size, state_dim,  output_dim)     – read‑out

    device = x.device
    num_measurements, seq_len, input_dim = x.shape
    state_dim = A_diag.size(1)
    output_dim = C.size(2)
    batch_size = A_diag.size(0)

    # Make A, B, C broadcastable over the measurement axis
    A_diag_unsqueeze = A_diag.unsqueeze(1)  # (B, 1, S)

    # Hidden state and output buffer
    h = torch.zeros(batch_size, num_measurements, state_dim, device=device)
    output = torch.empty(batch_size, num_measurements, seq_len, output_dim,
                         device=device)

    for t in range(seq_len):
        u_t = x[:, t, :]  # (M, I)
        Bu = torch.einsum('mi,biS->bmS', u_t, B)  # (B, M, S)
        h = h * A_diag_unsqueeze + Bu  # (B, M, S)

        # y_t = h @ C
        y_t = torch.einsum('bmS,bSo->bmo', h, C)  # (B, M, O)

        output[:, :, t, :] = y_t

    return output
