import numpy as np
import torch
pi=torch.tensor(np.pi)
C=torch.tensor(0.5772156649)
coeff = np.genfromtxt("systems/Rel/coeff_bessel.txt")[::-1]

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

def my_Q1_mid(z):
    L = 1
    k = 0
    for _ in range(len(coeff)):
        k+=L*coeff[_]
        L*=z
    return k


def my_Q1_big(z,M=50):
    b = 1.0
    k=0
    for m in range(M):
        k += b * (1/(2*z)) ** m
        b *= (1-(2*m+1)/4)/(m+1)
    k*=(np.pi/(2*z))**0.5    
    return k 

def my_Q1(z,M=10):
    z1=4
    z2=20
    mask1 = z<=z1
    mask3 = z>=z2
    mask2 = ~(mask1+mask3)
    y1 = mask1*z+mask2+mask3
    y2 = mask2*z+mask1+mask3
    y3 = mask3*z+mask1+mask2
    q = mask1 * my_Q1_small(y1) + mask2 * my_Q1_mid(y2) + mask3 * my_Q1_big(y3)
    return q

