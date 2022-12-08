import torch
from torch import nn

from sklearn.metrics import pairwise_distances

class CustomLoss(nn.Module):
    def __init__(self,  batch_size, l:float=5e-3) -> None:
        super(CustomLoss, self).__init__()
        self.lambda_param = l

        dists = pairwise_distances(torch.tensor(range(batch_size)).reshape(-1,1))
        self.J = torch.tensor(((dists == 1)).astype(int))

    def forward(
        self, 
        z_a: torch.Tensor
    ) -> torch.Tensor:
        device = z_a.device
        self.J = self.J.to(device)

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm, z_a_norm.T) / N # DxD

        # loss
        c_diff = (c - self.J[:N,:N]).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        a = ~self.J[:N,:N].bool()
        a[torch.eye(len(a)).bool()] = False
        c_diff[a] *= self.lambda_param
        loss = c_diff.sum()

        return loss

if __name__ == '__main__':
    criterion = CustomLoss(8)
    criterion(torch.rand((8,2048)))