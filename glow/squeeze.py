import torch
import torch.nn as nn
import numpy as np

# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    


class Squeeze(nn.Module):
    
    
    def __init__(self, factor):
        super(Squeeze, self).__init__()
        self.factor = factor
    
        
    def forward(self, x, logdet=None, reverse=False):
        n,c,h,w  = x.size()
        
        if not reverse:
            if self.factor == 1:
                return x, logdet
            # squeezing is done in one line unlike the original code
            assert h % self.factor == 0 and w % self.factor == 0, "h,w not divisible by factor: h=%d, factor=%d"%(h,self.factor)
            x = x.view(n, c*self.factor*self.factor, h//self.factor, w//self.factor)
            return x, logdet
        
        if reverse: 
            if self.factor == 1:
                return x
            assert c % self.factor**2 == 0, "channels not divisible by factor squared"
            # unsqueezing is also done in one line unlike the original code
            x = x.view(n, c //(self.factor**2), h*self.factor, w* self.factor)
            return x
    
    
    
if __name__ == "__main__":
    size = (16,64,16,16)
    x = torch.tensor(np.random.normal(5,10,size),dtype=torch.float,device=device)
    logdet = torch.tensor(0, dtype=torch.float, device=device, requires_grad=True)
    squeeze = Squeeze(factor=2)
    y, logdet = squeeze(x, logdet=logdet, reverse=False)
    x_rev = squeeze(y, reverse=True)
    print(y.size())
    print(x_rev.size())
    print(torch.norm(x_rev - x))
    
        