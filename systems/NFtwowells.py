from System import System
import numpy as np
import torch
from LOSS import KL_with_S

pi=torch.tensor(np.pi)

class Two_wells(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.normalizer = 0.5 * torch.log( 2 * pi * self.a )
        self.g = args["g"]
        self.x0 = args["x0"]
    
    def T(self,diff):
        t = (diff) ** 2 / (2 * self.a ** 2)
        return t
        
    def V(self,x):
        return self.g * ( x ** 2 - self.x0 ** 2 ) ** 2
    
    

      
        