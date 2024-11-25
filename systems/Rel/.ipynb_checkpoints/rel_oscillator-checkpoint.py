import numpy as np
import torch
from systems.Rel.rel_system import rel_System
from systems.Nonrel.oscillator import Oscillator
from systems.Rel.dimensionlesser import get_coeffs
pi=torch.tensor(np.pi)

class Rel_Oscillator(rel_System):
    
    def __init__(self,n_nod,beta,**args):
        s1, s2, self.s3 = get_coeffs(args["sigma"])
        super().__init__(n_nod,beta,s1=s1,s2=s2)
        self.sigma=args["sigma"]
    
    def V(self,x):
        return  self.s3 * x ** 2 / 2

    def U(self,x):
        return Oscillator.U(self,x)
    #def U(self,x):
        #return -x


