import torch
import torch.nn as nn
import numpy as np
from .actnorm import ActNorm
from .invertibe_conv import InvertibleConvolution
from .coupling import CouplingLayer


# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Flow(nn.Module):
    
    def __init__(self,channels, coupling, device, nn_init_last_zeros=False):
        super(Flow, self).__init__()
        self.actnorm  = ActNorm(channels,device)
        self.coupling = CouplingLayer(channels, coupling, device, nn_init_last_zeros)
        self.invconv  = InvertibleConvolution(channels, device)
        self.to(device)        

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            x, logdet, actnormloss = self.actnorm(x, logdet=logdet, reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after actnorm in forward"
            assert not np.isinf(x.mean().item()), "inf after actnorm in forward"
            assert not np.isnan(logdet.sum().item()), "nan in log after actnorm in forward"
            assert not np.isinf(logdet.sum().item()), "inf in log after actnorm in forward"
            
            x, logdet = self.invconv(x, logdet=logdet, reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after invconv in forward"
            assert not np.isinf(x.mean().item()), "inf after invconv in forward"
            assert not np.isnan(logdet.sum().item()), "nan in log after invconv"
            assert not np.isinf(logdet.sum().item()), "inf in log after invconv"
            
            x, logdet = self.coupling(x, logdet=logdet, reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after coupling in forward"
            assert not np.isinf(x.mean().item()), "inf after coupling in forward"
            assert not np.isnan(logdet.sum().item()), "nan in log after coupling"
            assert not np.isinf(logdet.sum().item()), "inf in log after coupling"

            return x, logdet, actnormloss
        
        if reverse:
            x = self.coupling(x, reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after coupling in reverse"
            assert not np.isinf(x.mean().item()), "inf after coupling in reverse"
            
            x = self.invconv(x,  reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after invconv in reverse"
            assert not np.isinf(x.mean().item()), "inf after invconv in reverse"
                        
            x = self.actnorm(x,  reverse=reverse)
            assert not np.isnan(x.mean().item()), "nan after actnorm in reverse"
            assert not np.isinf(x.mean().item()), "inf after actnorm in reverse"
            return x

    
if __name__ == "__main__":
    size  = (16,4,32,32)
    flow  = Flow(channels=4,coupling="affine",device=device,nn_init_last_zeros=False)
    opt   = torch.optim.Adam(flow.parameters(),lr=0.01)
    for i in range(5000):
        opt.zero_grad()
        x         = torch.tensor(np.random.normal(1,1,size),dtype=torch.float,device=device)
        logdet    = torch.tensor(0,dtype=torch.float,device=device,requires_grad=True)
        y_true    = x*2+1
        y, logdet, actloss = flow(x,logdet=logdet,reverse=False)
        x_rev     = flow(y,reverse=True)
        mse       = torch.norm(y_true - y)
        mse.backward()
        opt.step()
        mu        = y.mean().item()
        std       = y.std().item()
        loss_rev  = torch.norm(x_rev-x).item()
        print("\rmse=%0.3f | mu=%0.3f | std=%0.3f | rloss=%0.3f | logdet=%0.3f"
              %(mse.item(),mu,std,loss_rev,logdet.mean().item()), end="\r")
