import torch
from torch import nn
class AffineCouplingLayer(nn.Module):
    def __init__(
        self,
        theta: nn.Module,
        split,
        swap: int
    ):
        super().__init__()
        self.theta = theta
        self.split = split
        self.swap = swap

    def g(self, z: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        """g : z -> x. The inverse of f."""
        mask1=self.split[self.swap]
        mask2=self.split[(self.swap+1)%2]
        z1, z2 = z[:,mask1], z[:,mask2]
        z1 = torch.cat((z1,params),dim=-1)
        t, s = self.theta(z1)
        x2 = z2 * torch.exp(s) + t
        log_det = s.sum(-1) 
        z[:,mask2]=x2
        return z, log_det

    def f(self, x: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        mask1=self.split[self.swap]
        mask2=self.split[(self.swap+1)%2]
        x1, x2 = x[:,mask1], x[:,mask2]
        x1 = torch.cat((x1,params),dim=-1)
        t, s = self.theta(x1)
        z1, z2 = x1, torch.exp(-s)*(x2-t) 
        log_det = s.sum(-1) 
        x[:,mask2] = z2
        return x, log_det    