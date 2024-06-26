import numpy as np
import torch
pi=torch.tensor(np.pi)
C=torch.tensor(0.5772156649)
def my_Q1_small(z,M=10):
    a=z/2
    b=0
    psi=-C
    k=1/z
    for m in range(M):
        b=(torch.log(z/2)-1/(2*m+2)-psi)/(m+1)
        k+=a*b
        
        psi+=1/(m+1)
        a=a * (z/(2*m+2)) ** 2
    return k*torch.exp(z)     
def my_Q1_big(z,M=10):
    b = 1.0
    k=0
    for m in range(M):
        k += b * (1/(2*z)) ** m
        b *= (1-(2*m+1)/4)/(m+1)
    k*=(np.pi/(2*z))**0.5    
    return k 

def my_Q1(z,M=10,z0=10.5):
    mask1=z<=z0
    mask2=~mask1
    y1=mask1*z+mask2
    y2=mask2*z+mask1
    q=mask1*my_Q1_small(y1)+mask2*my_Q1_big(y2)
    return q

