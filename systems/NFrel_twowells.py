from System import System
import numpy as np
import torch
from LOSS import KL_with_S
from torch.special import scaled_modified_bessel_k1 as Q1

pi=torch.tensor(np.pi)

class Rel_two_wells(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.g = args["g"]
        self.x0 = args["x0"]
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
        return self.g * ( x ** 2 - self.x0 ** 2 ) ** 2
    
    

      
        