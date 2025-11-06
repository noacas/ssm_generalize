from typing import Optional, Tuple

import torch
import torch.nn as nn

from losses import get_losses_gd


class DiagonalSSM(nn.Module):
    def __init__(self, state_dim: int, input_dim: int = 1, output_dim: int = 1, init_scale: float = 1e-2, init_type: str = "regular"):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        if init_type == "near_one":
            self.A_diag = nn.Parameter(init_scale * torch.randn(state_dim) + 1)
        elif init_type == "double_max_A_j":
            initial_A = init_scale * torch.randn(state_dim)
            max_A_j = torch.argmax(initial_A)
            # Create mask for doubling the max value
            mask = torch.zeros_like(initial_A)
            mask[max_A_j] = 4
            final_A = initial_A + mask * initial_A
            self.A_diag = nn.Parameter(final_A)
        else:
            self.A_diag = nn.Parameter(init_scale * torch.randn(state_dim))

    def forward(self,
                w: list[torch.Tensor],
                alpha_teacher: float
                ) -> Tuple[torch.Tensor]:
        train_loss, gen_loss = get_losses_gd(self.A_diag, w, alpha_teacher)
        return train_loss, gen_loss