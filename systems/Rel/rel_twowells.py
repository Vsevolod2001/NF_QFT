from systems.Rel.rel_system import rel_System
from systems.Nonrel.twowells import Two_wells
import numpy as np
import torch

class Rel_two_wells(rel_System):
    
    def __init__(self,n_nod,beta,**args):
        self.g = args["g"]
        self.x0 = args["x0"]
        self.m=args["m"]
        super().__init__(n_nod,beta,s1=self.m ** 0.5 , s2=self.m)
        
    def V(self,x):
        return Two_wells.V(self,x)
    
    def U(self,x):
        return Two_wells.U(self,x)

      
        