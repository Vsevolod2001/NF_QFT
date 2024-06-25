from System import System
import numpy as np
import torch
from bessel import my_Q1 as Q1
from dimensionlesser import get_coeffs
pi=torch.tensor(np.pi)

class Rel_Oscillator(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.sigma=args["sigma"]
        self.s1, self.s2, self.s3 = get_coeffs(self.sigma)
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
        return  self.s3 * x ** 2 / 2
    


