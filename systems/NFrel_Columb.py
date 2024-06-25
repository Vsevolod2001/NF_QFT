from System import System
import numpy as np
import torch
from bessel import my_Q1 as Q1
pi=torch.tensor(np.pi)

class Rel_Columb(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.alpha=args["alpha"]
        self.R=args["R"]
        self.s1=(self.alpha)**(-2) 
        self.s2=(self.alpha)**(-4)
        a=torch.tensor(self.a)
        self.normalizer = torch.log(pi * self.s1 / (self.s2 * Q1(self.s2 * a))) 
    
    def T(self,diff):
        a=torch.tensor(self.a)
        y=( 1 + (diff/a)**2 * (self.s1) ** (-2))**0.5
        ln=torch.log(Q1( self.s2 * a * y )/(y * Q1(self.s2 * a)))
        t = self.s2 * (y-1) - ln / a
        return t
    
    def V(self,x):
        return  -(x ** 2 + self.R ** 2) ** (-0.5)
    


