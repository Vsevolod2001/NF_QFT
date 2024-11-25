import torch
from torch import nn
from typing import Callable, List, Tuple


from flows.theta import ThetaNetwork
from flows.Layers import AffineCouplingLayer
from transforms import get_split_masks
from transforms import get_pair_split_masks

class NormalizingFlow(nn.Module):
    
    def __init__(self, flows: List[nn.Module], O=torch.tensor([])):
        super().__init__()
        self.flows = flows
        self.O = O
        self.Ot = torch.t(O)
        self.ort = self.O.shape[0]>0
        
    def configure_flows(n_flows,num_hidden,hidden_dim,p_drop,dim,param_dim=0,mask_config = get_pair_split_masks):  # n_flows=8,...,12
        flows = []
        split_masks_d = mask_config(dim)
    
        for k in range(n_flows):
            theta = ThetaNetwork.configure_theta(num_hidden = num_hidden, 
                                                 hidden_dim = hidden_dim, 
                                                 p_drop = p_drop,
                                                 in_dim = dim//2+param_dim,
                                                 out_dim = dim//2)
            flows.append(AffineCouplingLayer(theta, split = split_masks_d, swap = k % 2))
   
        flows = nn.ModuleList(flows)
        return flows     

    

    def g(self, z: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        
        x, sum_log_abs_det = z, torch.zeros(z.size(0)).to(z.device)
        for flow in reversed(self.flows):
            x, log_abs_det = flow.g(x,params)
            sum_log_abs_det += log_abs_det
        
        if self.ort:
            x = torch.matmul(x,self.Ot.to(x.device))
            
        return x, sum_log_abs_det
    
        
    def __len__(self) -> int:
        return len(self.flows)
    
    def f(self, x: torch.Tensor,params=torch.tensor([])) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            if self.ort:
                x=torch.matmul(x,self.O.to(x.device))
        
            z, sum_log_abs_det = x, torch.zeros(x.size(0)).to(x.device)
        
            for flow in self.flows:
                z, log_abs_det = flow.f(z,params)
                sum_log_abs_det += log_abs_det
        
        return z, sum_log_abs_det