from System import System
import numpy as np
import torch
from NFandist import calc_Z, get_A
from LOSS import KL_with_S

pi=torch.tensor(np.pi)

class Oscillator(System):
    
    def __init__(self,n_nod,beta):
        super().__init__(n_nod,beta)
        self.normalizer = 0.5 * torch.log( 2 * pi * self.a )
        self.Log_Z=torch.log(torch.tensor(calc_Z(n_nod,beta)))
        self.mat=torch.tensor(get_A(n_nod,beta)).float()
    
    def T(self,diff):
        t = (diff) ** 2 / (2 * self.a ** 2)
        return t
        
    def V(self,x):
        return x ** 2 / 2
    
    
    def U(self,x):
        return -x
    
    def Y(self,diff):
        return diff / (self.a ** 2)
        
    
    
    
    def mat_S(self,x):
        A=(self.mat).to(x.device)
        xt=(torch.t(x)).to(x.device)
        S=0.5 * torch.matmul(x,torch.matmul(A,xt))
        S=torch.trace(S)/(x.shape[0])
        S+=self.n_nod*self.normalizer
        S+=self.Log_Z
        return S
    
    def get_mat_S(self):
        def S(x):
            return self.mat_S(x)
        return S
    
    def get_mat_KL(self):
        mat_S=self.get_mat_S()
        KL=KL_with_S(mat_S,self.n_nod)
        return KL
      
    def theor_Psi2(self,z):
        return (pi) ** (-0.5) * np.exp(-z ** 2)     