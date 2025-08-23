from typing import Optional, Tuple

import torch
import torch.nn as nn

from losses import get_losses


class DiagonalSSM(nn.Module):
    def __init__(self, state_dim: int, input_dim: int = 1, output_dim: int = 1, init_scale: float = 1e-2, init_type: str = "regular"):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.A_diag = nn.Parameter(init_scale * torch.randn(state_dim))
        if init_type == "near_one":
            self.A_diag = self.A_diag + 1
        elif init_type == "double_max_A_j":
            max_A_j = torch.argmax(self.A_diag)
            self.A_diag[max_A_j] = 2 * self.A_diag[max_A_j]

    def forward(self,
                w: torch.Tensor,
                alpha_teacher: float
                ) -> Tuple[torch.Tensor]:
        train_loss, gen_loss = get_losses(self.A_diag, w, alpha_teacher)
        return train_loss, gen_loss