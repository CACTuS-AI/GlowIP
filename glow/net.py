import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .actnorm import ActNorm


# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

class NN(nn.Module):
    def __init__(self, channels_in, channels_out, device, init_last_zeros=False):
        super(NN, self).__init__()
        self.conv1    = nn.Conv2d(channels_in,512,kernel_size=(3,3),stride=1,padding=1,bias=True)
        self.actnorm1 = ActNorm(512,device)
                                
        self.conv2    = nn.Conv2d(512,512,kernel_size=(1,1),stride=1,padding=0,bias=True)
        self.actnorm2 = ActNorm(512,device)
        
        self.conv3    = nn.Conv2d(512,channels_out,kernel_size=(3,3),stride=1,padding=1,bias=True)        
        self.logs     = nn.Parameter(torch.zeros(channels_out, 1, 1))
        
        
        
        # initializing
        with torch.no_grad():
            
            nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv1.bias)
            
            nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv2.bias) 
            
            if init_last_zeros:
                nn.init.zeros_(self.conv3.weight) # last layer initialized with zeros
            else:
                nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
            nn.init.zeros_(self.conv3.bias)
            
        # to device
        self.to(device)
            
    def forward(self, x):
        x = self.conv1(x)
        x, _, _ = self.actnorm1(x, logdet=0, reverse=False) 
        x = F.relu(x)
        x = self.conv2(x)
        x, _, _ = self.actnorm2(x, logdet=0, reverse=False) 
        x = F.relu(x)
        x = self.conv3(x)
        x = x * torch.exp(self.logs * 3)
        return x
    
    
    
if __name__ == "__main__":
    size = (16,64,16,16)
    net = NN(channels_in=64,channels_out=64, device=device,init_last_zeros=True)
    opt = torch.optim.Adam(net.parameters(), lr=0.0001)
    
    for i in range(5000):
        opt.zero_grad()
        x = torch.tensor(np.random.normal(0,1,size),dtype=torch.float,device=device)
        y_true = x*2 + 2
        y = net(x)
        loss = torch.norm(y - y_true)
        mu   = y.mean().item()
        std  = y.std().item()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print("\r loss=%0.3f| mu=%0.3f | std=%0.3f"%(loss.item(),mu,std), end="\r")
            
    
