import time
import torch
from NFconstants import N_nod
class HMC:
    def __init__(self,system,model="none",n_samp=1,eps=0.001,n_steps=10,sigma=1,N_sweep=100,log_per=1000,gamma=0,auto=True,imp=False):
        self.system = system
        self.S = 0
        self.model = model
        self.eps = eps
        self.n_steps = n_steps
        self.N_sweep = N_sweep
        self.log_per = log_per
        self.ar = 0
        self.sigma=sigma
        self.n_samp=n_samp
        self.shift_dist = torch.distributions.Normal(torch.zeros(N_nod), sigma * torch.ones(N_nod))
        self.un = torch.distributions.Uniform(0, 1)
        self.auto = auto
        self.imp = imp    
        self.res = []
        self.times = []
    
    def leapfrog_step(self,p,x,eps):
        if self.auto:
            F = self.calc_F(x)
        else:
            F = self.system.F(x)
            
        p_half = p + 0.5 * eps * F  # F=-dS/dx
        y = x + eps * p_half
        q = p_half + 0.5 * eps * F 
        return q, y

    def hmc_new(self,P,X):
        for i in range(self.n_steps): 
            P , X = self.leapfrog_step(P,X,self.eps)
        return P,X       


    def hmc_sweep(self,x):
        p = self.shift_dist.sample((self.n_samp,)).to(x.device)
        K0 = torch.sum(p**2/2,axis=1)
        
        q , y = self.hmc_new(p.clone(),x.clone())
        S_new = self.calc_S(y)
        dS = S_new-self.S + torch.sum(q**2/2,axis=1)-K0
        
        prob = torch.exp(-dS)
        ind = self.un.sample((self.n_samp,)).to(x.device)<prob
        mask = ind.nonzero()
        
        if len(mask)>1:
            mask = mask.squeeze()
        if  len(mask)>0:
            self.S[mask] = S_new[mask]
            x[mask,:] = y[mask,:]
        
        self.ar = torch.mean(ind.type('torch.FloatTensor'))    
    
    
    def log(self,z):
        if self.model=="none" or self.imp:
            x1=z
        else:    
            x1,_=self.model(z)
        print(torch.mean(x1**2),self.ar)
        self.res.append(torch.mean(x1**2).cpu())
        self.Times.append(time.time()-self.start)
    
    def calc_S(self,z,grads=False):
        
        if self.model=="none":
            S=self.system.Full_S(z).to(z.device) 
        
        else:
            if self.imp==False:
                if grads :
                    x , lad = self.model.g_samp(z)
                else:
                    x , lad = self.model(z)
                S = self.system.Full_S(x).to(z.device)-lad
            else:
                S = self.system.Full_S(z)+self.model.log_prob(z)
        return S
    
    def calc_F(self,z):
        t = z.clone()
        t.requires_grad = True
        s = torch.sum(self.calc_S(t,grads=True))
        s.backward()
        F =-(t.grad).detach()
        del(t)
        return F
    
    def init_state(self,z):
        self.start=time.time()
        self.S=self.calc_S(z)
        self.res=[]
        self.Times=[]

    def run(self,x):
        self.init_state(x)
        for i in range(self.N_sweep):
            if i%self.log_per==0:
                self.log(x)
            self.hmc_sweep(x)
        return x        