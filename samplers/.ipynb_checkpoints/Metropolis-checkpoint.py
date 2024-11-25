import time
import torch
class Metropolis:
    def __init__(self,system,N_samp,d,model="none",N_sweep=100,log_per=1000):
        
        self.system = system
        self.n_nod = system.n_nod
        self.n_samp = N_samp
        self.S = 0
        self.model = model
        self.d = d
        self.N_sweep = N_sweep
        self.log_per = log_per
        self.ar = 0
        self.shift_dist = torch.distributions.Uniform(torch.zeros(self.n_nod), torch.ones(self.n_nod))
        self.un = torch.distributions.Uniform(0, 1)
        self.res=[]
        self.times=[]
        
    def sweep(self,z):
        shift = self.shift_dist.sample((self.n_samp,)).to(z.device)
        y = z+self.d*(2*shift-1)
    
        S_new=self.calc_S(y)
    
        dS=S_new-self.S
        prob = torch.exp(-dS)
        ind = self.un.sample((self.n_samp,)).to(z.device)<prob
    
        mask=ind.nonzero()
        if len(mask)>1:
            mask=mask.squeeze()
    
        if  len(mask)>0:
            self.S[mask]=S_new[mask]
            z[mask,:]=y[mask,:]  
        self.ar = torch.mean(ind.type('torch.FloatTensor'))           
    
    
    def log(self,z):
        if self.model=="none":
            x1=z
        else:    
            x1,_=self.model(z)
        print(torch.mean(x1**2),self.ar)
        self.res.append(torch.mean(x1**2).cpu())
        self.Times.append(time.time()-self.start)
    
    def calc_S(self,z):
        if self.model=="none":
            S=self.system.Full_S(z).to(z.device)
        else:
            x , lad = self.model(z)
            S=self.system.Full_S(x).to(z.device)-lad   
        return S
    
    def init_state(self,z):
        self.start=time.time()
        self.S=self.calc_S(z)
        self.res=[]
        self.Times=[]
        
    
    def run(self,z):
        
        self.init_state(z)
    
        for i in range(self.N_sweep):
            if i%self.log_per==0:
                self.log(z)
            self.sweep(z)
        
        return z    
    