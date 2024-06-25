from System import System
import numpy as np
import torch
from NFandist import calc_Z, get_A
from LOSS import KL_with_S
from torch.special import scaled_modified_bessel_k1 as Q1

pi=torch.tensor(np.pi)

class Rel_Morse(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.alpha=args["alpha"]
        self.m=args["m"]
        self.s1, self.s2 = self.m ** 0.5, self.m
        a=torch.tensor(self.a)
        self.normalizer = torch.log(pi * self.s1 / (self.s2 * Q1(self.s2 * a))) 
    
    def T(self,diff):
        a=torch.tensor(self.a)
        c=(self.s1) ** (-2)
        y=(1 + (diff/a)**2 * c)**0.5
        ln=torch.log(Q1( self.s2 * a * y )/(y * Q1(self.s2 * a)))
        t = self.s2 * (y-1) - ln / a
        return t

        
    def V(self,x):
        y=torch.exp(-self.alpha * x)-1
        return 0.5 * (y**2 - 1)
    
    
    

      
        