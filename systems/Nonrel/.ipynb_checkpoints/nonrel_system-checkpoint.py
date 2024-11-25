#import sys
#sys.path.append('./systems')
#sys.path.append("C:\\Users\\SEVA1\\PathIntegralMonte-Carlo\\Normalizingflow\\systems")
from systems.System import System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class nonrel_System(System):
    
    def __init__(self,n_nod,beta):
        super().__init__(n_nod,beta)
        self.normalizer = 0.5 * torch.log( 2 * pi * self.a )
    
    def T(self,diff):
        t = (diff) ** 2 / (2 * self.a ** 2)
        return t
    
    def Y(self,diff):
        return diff / (self.a ** 2)   