from System import System
import numpy as np
import torch
from LOSS import KL_with_S
from math import gamma

pi=torch.tensor(np.pi)

class Morse(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.normalizer = 0.5 * torch.log( 2 * pi * self.a )
        self.alpha=args["alpha"]
    
    def T(self,diff):
        t = (diff) ** 2 / (2 * self.a ** 2)
        return t
        
    def V(self,x):
        y=torch.exp(-self.alpha * x)-1
        return 0.5 * (y**2 - 1)
    
    
    def theor_Psi2(self,x):
        alpha = self.alpha
        N = 2**(2 / alpha - 1) * (2-alpha)/( (alpha ** (2/alpha - 1)) * gamma(2/alpha))
        power = (-2 / alpha) * np.exp(-alpha * x) - (2-alpha) * x
        P=N*np.exp(power)
        return P
    

      
        