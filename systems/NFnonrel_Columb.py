from System import System
import numpy as np
import torch
pi=torch.tensor(np.pi)

class NonRel_Columb(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.alpha=args["alpha"]
        self.R=args["R"]
        a=torch.tensor(self.a)
        self.normalizer = 0.5 * torch.log( 2 * pi * self.a )
    
    def T(self,diff):
        t = (diff) ** 2 / (2 * self.a ** 2)
        return t
    
    def V(self,x):
        return  -(x ** 2 + self.R ** 2) ** (-0.5)
    


