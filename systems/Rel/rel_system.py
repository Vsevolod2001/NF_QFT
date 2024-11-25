from systems.System import System
import numpy as np
import torch
#from torch.special import scaled_modified_bessel_k1 as Q1
from systems.Rel.bessel import my_Q1 as Q1
from scipy.special import kv

pi=torch.tensor(np.pi)

class rel_System(System):
    
    def __init__(self,n_nod,beta,**args):
        super().__init__(n_nod,beta)
        self.s1 = args["s1"]
        self.s2 = args["s2"]
        a=torch.tensor(self.a)
        self.normalizer = torch.log(pi * self.s1 / (self.s2 * Q1(self.s2 * a))) 
    
    def T(self,diff):
        a=torch.tensor(self.a)
        c=(self.s1) ** (-2)
        y=(1 + (diff/a)**2 * c) ** 0.5
        ln=torch.log(Q1( self.s2 * a * y )/(y * Q1(self.s2 * a)))
        t = self.s2 * (y-1) - ln / a
        return t


    
    def Y(self,diff):
        r = diff/self.a
        eta = self.a * self.s2 * ( 1 + (r ** 2) * self.s1 ** (-2) ) ** 0.5
        f = (kv(0,eta)+kv(2,eta))/(2*kv(1,eta))
        f = (f+eta**(-1)) * eta**(-1)
        return (self.s2/self.s1)**2 * r * f
    
    


