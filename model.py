from typing import Optional, Tuple

import torch
import torch.nn as nn

from ssm_forward import ssm_forward


class DiagonalSSM(nn.Module):
    def __init__(self, state_dim: int, input_dim: int = 1, output_dim: int = 1, init_scale: float = 1e-2):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.A_diag = nn.Parameter(init_scale * torch.randn(state_dim))

    def forward(self,
                x: torch.Tensor,
                alpha_teacher: float
                ) -> Tuple[torch.Tensor]:
        return ssm_forward(self.A_diag, x, alpha_teacher)