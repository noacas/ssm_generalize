from typing import Optional, Tuple

import torch
import torch.nn as nn

from losses import get_losses


class DiagonalSSM(nn.Module):
    def __init__(self, state_dim: int, input_dim: int = 1, output_dim: int = 1, init_scale: float = 1e-2):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.A_diag = nn.Parameter(init_scale * torch.randn(state_dim))

    def forward(self,
                w: torch.Tensor,
                alpha_teacher: float
                ) -> Tuple[torch.Tensor]:
        train_loss, gen_loss = get_losses(self.A_diag, w, alpha_teacher)
        return train_loss, gen_loss