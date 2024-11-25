from systems.Nonrel.nonrel_system import nonrel_System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class X4(nonrel_System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.g=args["g"]
        
    def V(self,x):
        return x ** 2 / 2 + self.g * x ** 4
        
    def U(self,x):
        return -x - 4 * self.g * x ** 3
        
    

      
        