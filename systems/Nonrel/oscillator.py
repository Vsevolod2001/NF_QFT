from systems.Nonrel.nonrel_system import nonrel_System
import numpy as np
import torch
from systems.Nonrel.osc_andist import calc_Z, get_A
pi=torch.tensor(np.pi)

class Oscillator(nonrel_System):
    
    def __init__(self,n_nod,beta):
        super().__init__(n_nod,beta)
        self.Log_Z=torch.log(torch.tensor(calc_Z(n_nod,beta)))
        
    def V(self,x):
        return x ** 2 / 2
    
    
    def U(self,x):
        return -x
        
    
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