from systems.Rel.rel_system import rel_System
from systems.Nonrel.Columb import Columb
import numpy as np
import torch

class Rel_Columb(rel_System):
    
    def __init__(self,n_nod,beta,**args):
        self.alpha=args["alpha"]
        self.R=args["R"]
        super().__init__(n_nod,beta,s1=(self.alpha)**(-2),s2=(self.alpha)**(-4))
    
    
    def V(self,x):
        return  Columb.V(self,x)

    def U(self,x):
        return Columb.U(self,x)
        
    


