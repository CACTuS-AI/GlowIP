import torch
import torch.nn as nn


# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()
    
    def forward(self, x, y = None, reverse=False):
        n,c,h,w = x.size()
        if not reverse:
            x1 = x[:,:c//2,:,:]
            x2 = x[:,c//2:,:,:]
            return x1, x2
        if reverse:
            assert y is not None, "y must be given"
            x = torch.cat([x, y], dim=1)
            return x
            
            
            
    
    
if __name__ == "__main__":
    size = (16,64,32,32)
    x = np.random.normal(0,1,size)
    x = torch.tensor(x, device=device, dtype=torch.float, requires_grad=True)
    split = Split()
    x1, x2 = split(x)
    y      = split(x1,x2,reverse=True)
    print((x-y).norm().item())
    
    
    