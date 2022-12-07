from typing import List, Tuple

import torch
from torch import nn

class CustomLoss(nn.Module):
    def __init__(self, l:float=5e-3) -> None:
        super(CustomLoss, self).__init__()
        self.l = l

    def forward(
        self, 
        # embs1: List[Tuple[int, torch.Tensor]], 
        # embs2: List[Tuple[int, torch.Tensor]]
        z_a: torch.Tensor, 
        z_b: torch.Tensor
    ) -> torch.Tensor:
        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD

        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss