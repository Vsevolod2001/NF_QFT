from systems.Nonrel.nonrel_system import nonrel_System
import numpy as np
import torch
pi=torch.tensor(np.pi)

class Columb(nonrel_System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.alpha=args["alpha"]
        self.R=args["R"]
    
    def V(self,x):
        return  -self.alpha * (x ** 2 + self.R ** 2) ** (-0.5)
    
    def U(self,x):
        return -self.alpha * x * (x ** 2 + self.R ** 2) ** (-1.5)
    


