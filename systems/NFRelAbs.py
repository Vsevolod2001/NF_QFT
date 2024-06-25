from System import system
from NFconstants import a
import numpy as np
import torch
from torch.special import scaled_modified_bessel_k1 as Q1

class rel_abs(system):
    
    def __init__(self,hbar=1,m=1,w=1):
        super().__init__(m, hbar)
        self.w=w
    
    def T(self,diff):
        y=( 1 + (diff/a)**2)**0.5
        ln=torch.log(Q1( self.m * a * y )/y)
        t = (y-1) * self.m - ln / a
        return t
    
    def V(self,x):
        return (self.w) * torch.abs(x) 
    
rel_abs=rel_abs(m=0.1,w=0.1)