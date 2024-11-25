from tqdm import tqdm
import numpy as np
import torch

def calc_Loss(system,model,dataloader,n_batch=8):
    kl = []
    Loss = system.get_KL()
    for i in tqdm(range(n_batch)):
        batch = next(iter(dataloader))
        x,lad = model(batch)
        kl.append(Loss(x,lad))
    kl=torch.tensor(kl)
    return torch.mean(kl), torch.std(kl)

def calc_psi2(trajs,n_bins = 1001):
    n_traj = trajs.shape[0]
    n_nod = trajs.shape[1]
    hst = torch.histogram(trajs.cpu(),bins = n_bins)
    h = (hst[1][-1]-hst[1][0])/n_bins
    x = hst[1][:-1]
    psi2 = hst[0]/(n_nod*n_traj*h)
    return x.numpy(), psi2.numpy()

def G(X,n_p="all"):
    n_traj = X.shape[0]
    n_nod = X.shape[1]
    if n_p == "all":
        n_p = X.shape[1]
    G = np.zeros((n_p))
    Y = X.clone()
    Xt = torch.t(X)
    for s in tqdm(range(n_p)):
        G[s] = torch.trace(torch.matmul(Y,Xt))
        Y = torch.roll(Y,-1,1)
    return G / (n_traj*n_nod)

class gen_fun_W:
    
    def __init__(self,start,stop,step):
        self.k = np.arange(start,stop,step)
        self.W = []
        self.p = []

    def calc_W(self,trajs):
        f=[]
        t=0
        for i in range(len(self.k)):
            t=torch.mean(torch.exp(self.k[i]*trajs))
            f.append(t)
        f = torch.tensor(f).numpy()
        f = np.log(f)
        self.W = f
        return self.W
    
    def poly(self,deg):
        self.p = np.polyfit(self.k,self.W,deg)
        return self.p