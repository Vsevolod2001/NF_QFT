from systems.Rel.rel_system import rel_System
from systems.Nonrel.Morse import Morse
import numpy as np
import torch

class Rel_Morse(rel_System):
    
    def __init__(self,n_nod,beta,**args):
        self.alpha=args["alpha"]
        self.m=args["m"]
        super().__init__(n_nod,beta, s1=self.m ** 0.5, s2=self.m) 

        
    def V(self,x):
        return Morse.V(self,x)
    
    def U(self,x):
        return Morse.U(self,x)
    

      
        