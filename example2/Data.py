import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import random
from NFconstants import N_nod
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_random_seed(42)

class MY_Dataset(Dataset):
    def __init__(self, distribution,n_nod):
        super().__init__()
        self.distribution = distribution
        self.n_nod = n_nod
        self.features = torch.tensor(0)
        self.n_sample = 2**18

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.distribution.sample((1,))[0]
    
    def sample(self,n_sample):
        self.features=self.distribution.sample((n_sample,))
        

#normal_dist=torch.distributions.Exponential(torch.ones(N_nod))
#normal_dist=torch.distributions.Uniform(torch.zeros(N_nod), torch.ones(N_nod))
normal_dist=torch.distributions.Normal(loc=torch.zeros(N_nod), scale=torch.ones(N_nod))
DS=MY_Dataset(normal_dist,N_nod)
train_loader = DataLoader(DS, batch_size=2**10, shuffle=True,num_workers=3)
