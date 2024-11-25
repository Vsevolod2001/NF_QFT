import time
import torch
class Langevin:
    def __init__(self,system,N_samp,eps=0.001,N_sweep=10,log_per=1000):
        self.system = system
        self.eps = eps
        self.N_sweep = N_sweep
        self.log_per = log_per
        self.n_nod = self.system.n_nod
        self.n_samp = N_samp
        self.shift_dist = torch.distributions.Normal(torch.zeros(self.n_nod), torch.ones(self.n_nod))
        self.res = []
        self.times = []
    

    def sweep(self,x):
        dw = self.shift_dist.sample((self.n_samp,)).to(x.device)
        x+= 0.5*self.eps*self.system.F(x) + self.eps**0.5 * dw
        
    def log(self,z):
        print(torch.mean(z**2))
        self.res.append(torch.mean(z**2).cpu())
        self.Times.append(time.time()-self.start)
    
    
    def init_state(self):
        self.start=time.time()
        self.res=[]
        self.Times=[]

    def run(self,x):
        self.init_state()
        for i in range(self.N_sweep):
            if i%self.log_per==0:
                self.log(x)
            self.sweep(x)
        return x        