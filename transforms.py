import numpy as np
import torch
from NFconstants import N_nod
def get_O(N):
    O=np.zeros((N,N))
    for k in range(N):
        O[k][0]=1
    
    if N%2==0:
        for k in range(N):
            O[k][1]=(-1)**k
    
    ev=int(N%2==0)
    Smax=(N-1-ev)//2    
    
    for s in range(1,Smax+1):
        for k in range(N):
            O[k][2*s-1+ev]=(2**0.5)*np.cos(2*np.pi*s*k/N)
            O[k][2*s+ev]=(2**0.5)*np.sin(2*np.pi*s*k/N)

    O=O / (N ** 0.5)
    return O

def t_get_O(N):
    return torch.tensor(get_O(N)).float()

def get_split_masks(dim=N_nod):
    mask1=list(range(0,dim,2))
    mask2=list(range(1,dim,2))
    split_masks=[mask1,mask2]
    return split_masks 

def get_pair_split_masks(dim=N_nod):
    mask1=list(range(0,dim,4))
    mask2=list(range(2,dim,4))
    mask1=(mask1+list(map(lambda x:x+1,mask1)))
    mask2=(mask2+list(map(lambda x:x+1,mask2)))
    mask1.sort()
    mask2.sort()
    split_masks=[mask1,mask2]
    return split_masks


def get_points(start,stop,step):
    if step == 0:
        print("step = 0 error")
        return 1
    x=start
    Y=[]
    while x < stop:
        Y.append(x)
        Y.append(x+1)
        x += step
    return Y


def config_RG_masks(m,n_nod):
    masks = []
    start = 0
    step = n_nod // (2 ** (m-1))
    tmp = get_points(start,n_nod,step)
    masks.append(tmp)
    if n_nod == 2 ** m:
        return masks
    start = step//2
    masks.append(get_points(start,n_nod,step))
    while step > 4:
        step //= 2
        start = step//2
        masks.append(get_points(start,n_nod,step))
    return masks
    