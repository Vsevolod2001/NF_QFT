from systems.Ultrarel.ur_system import ur_System
from systems.Nonrel.oscillator import Oscillator
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Ur_Oscillator(ur_System):
    
    def __init__(self,n_nod,beta):
        super().__init__(n_nod,beta)
    
    def V(self,x):
        return Oscillator.V(self,x)
    
    def U(self,x):
        return Oscillator.U(self,x)
    

