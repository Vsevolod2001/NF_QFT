from System import System
import numpy as np
import torch
from LOSS import KL_with_S

pi=torch.tensor(np.pi)

class X4(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.normalizer = 0.5 * torch.log( 2 * pi * self.a )
        self.g=args["g"]
    
    def T(self,diff):
        t = (diff) ** 2 / (2 * self.a ** 2)
        return t
        
    def V(self,x):
        return x ** 2 / 2 + self.g * x ** 4
    
    

      
        