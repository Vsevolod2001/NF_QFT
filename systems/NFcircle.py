from System import System
import numpy as np
import torch
from LOSS import KL_with_S

pi=torch.tensor(np.pi)

class Circle(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.normalizer = 0
        self.g = args["g"]
        self.x0 = args["x0"]
    
    def T(self,diff):
        return 0*diff
        
    def V(self,x):
        return 0.5 * self.g * ( x ** 2 + torch.roll(x,-1,1) ** 2 - self.x0 ** 2 ) ** 2
    
    

      
        