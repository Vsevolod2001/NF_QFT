from systems.Nonrel.nonrel_system import nonrel_System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Two_wells(nonrel_System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.g = args["g"]
        self.x0 = args["x0"]
        
    def V(self,x):
        return self.g * ( x ** 2 - self.x0 ** 2 ) ** 2
        
    def U(self,x):
        return -4*self.g*x*(x**2-self.x0**2)
    

      
        