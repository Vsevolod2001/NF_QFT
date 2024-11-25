import torch 
from torch import nn
import numpy as np
pi=torch.tensor(np.pi)
from NFconstants import N_nod



class KL_with_S(nn.Module):
    
    def __init__(self,S,n_nod):
        super().__init__()
        self.S=S
        self.n_nod=n_nod
    
    def forward(self,x,log_abs_det):
        S=self.S(x)
        loss=torch.mean(S-log_abs_det)
        loss-=(self.n_nod / 2) * (1+torch.log( 2 * pi ))
        return loss

