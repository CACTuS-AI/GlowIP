import torch
import torch.nn as nn
import numpy as np

# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    


class ActNorm(nn.Module):
    def __init__(self, channels,device):
        super(ActNorm, self).__init__()
        size       = (1,channels,1,1)
        self.logs  = torch.nn.Parameter(torch.zeros(size,dtype=torch.float,device=device,requires_grad=True))
        self.b     = torch.nn.Parameter(torch.zeros(size,dtype=torch.float,device=device,requires_grad=True))
        self.initialized = False
        
    def initialize(self, x):
        if not self.training:
            return
        with torch.no_grad():
            b_    = x.clone().mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
            s_    = ((x.clone() - b_)**2).mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
            b_    = -1 * b_
            logs_ = -1 * torch.log(torch.sqrt(s_)) 
            self.logs.data.copy_(logs_.data)
            self.b.data.copy_(b_.data)
            self.initialized = True

    def apply_bias(self, x, logdet, reverse):
        if not reverse:
            x = x + self.b
            assert not np.isnan(x.mean().item()), "nan after apply_bias in forward: x=%0.3f, b=%0.3f"%(x.mean().item(), self.b.mean().item())
            assert not np.isinf(x.mean().item()), "inf after apply_bias in forward: x=%0.3f, b=%0.3f"%(x.mean().item(), self.b.mean().item())
            return x, logdet
        if reverse:
            x =  x - self.b
            return x
        
    def apply_scale(self,x, logdet, reverse ):
        if not reverse:
            n,c,h,w = x.size()
            x = x * torch.exp(self.logs)
            logdet = logdet + h*w*self.logs.view(-1).sum()
            assert not np.isnan(x.mean().item()), "nan after apply_scale in forward: x=%0.3f, logs=%0.3f"%(x.mean().item(), self.logs.mean().item())
            assert not np.isinf(x.mean().item()), "inf after apply_scale in forward: x=%0.3f, logs=%0.3f"%(x.mean().item(), self.logs.mean().item())
            assert not np.isnan(logdet.sum().item()), "nan in log after apply_scale in forward: logdet=%0.3f, logs=%0.3f"%(logdet.mean().item(), self.logs.mean().item())
            assert not np.isinf(logdet.sum().item()), "inf in log after apply_scale in forward: logdet=%0.3f, logs=%0.3f"%(logdet.mean().item(), self.logs.mean().item())
            return x, logdet
        if reverse:
            x = x * torch.exp(-1 * self.logs)
            return x
        
    def forward(self, x, logdet=None, reverse=False):
        if not self.initialized:
            self.initialize(x)
        if not reverse:
            x, logdet   = self.apply_bias(x,  logdet, reverse)
            x, logdet   = self.apply_scale(x, logdet, reverse)
            
            loss_mean   = x.mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True).mean()
            loss_std    = ((x - loss_mean)**2).mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True).mean()
            actnormloss = torch.abs(loss_mean) + torch.abs(1 - loss_std)
            
            return x, logdet, actnormloss
        if reverse:
            x = self.apply_scale(x, logdet, reverse)
            assert not np.isnan(x.mean().item()), "nan after apply_scale in reverse"
            assert not np.isinf(x.mean().item()), "inf after apply_scale in forward"
            
            x = self.apply_bias(x,  logdet, reverse)
            assert not np.isnan(x.mean().item()), "nan after apply_bias in reverse"
            assert not np.isinf(x.mean().item()), "inf after apply_bias in reverse"
            return x
    
    
    
if __name__ == "__main__":
    size = (16,3,16,16)
    actnorm = ActNorm(channels=size[1],device=device)
    logdet = torch.tensor(0, dtype=torch.float, device=device, requires_grad=True)
    opt = torch.optim.Adam(actnorm.parameters(), lr=0.01)
    for i in range(5000):
        opt.zero_grad()
        x = torch.tensor(np.random.normal(5,10,size),dtype=torch.float,device=device)
        y,logdet2, loss = actnorm(x, logdet, reverse=False)
        x_rev           = actnorm(y, reverse=True)
        rev_loss        = torch.norm(x_rev - x).item()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print("\r loss=%0.3f | logdet=%0.3f | b=%0.3f | s=%0.3f | rev=%0.3f "%(loss, -logdet2.mean().item(), actnorm.b.mean().item(), np.exp(actnorm.logs.mean().item()),rev_loss),end="\r")