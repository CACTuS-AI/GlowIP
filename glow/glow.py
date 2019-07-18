import torch
import torch.nn as nn
from .flow import Flow
from .squeeze import Squeeze
from .split import Split
import numpy as np
import skimage.io as sio
from skimage.transform import resize
import torch.utils.checkpoint as checkpoint
# device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Glow(nn.Module):
    
    def __init__(self, image_shape, K, L, coupling, device, n_bits_x=8, nn_init_last_zeros=False):
        super(Glow, self).__init__()
        self.image_shape = image_shape
        self.K            = K
        self.L            = L
        self.coupling     = coupling
        self.n_bits_x     = n_bits_x
        self.device       = device
        self.init_resizer = False
        self.nn_init_last_zeros = nn_init_last_zeros
        
        # setting up layers
        c,w,h = image_shape
        self.glow_modules = nn.ModuleList()
        
        for l in range(L-1):
            # step of flows
            squeeze = Squeeze(factor=2)
            c = c * 4
            self.glow_modules.append(squeeze)
            for k in range(K):
                flow = Flow(c,self.coupling,device,nn_init_last_zeros)                
                self.glow_modules.append(flow)
            split = Split()
            c = c // 2
            self.glow_modules.append(split)
        # L-th flow 
        squeeze = Squeeze(factor=2)
        c = c * 4
        self.glow_modules.append(squeeze)
        flow =Flow(c,self.coupling,device,nn_init_last_zeros)
        self.glow_modules.append(flow)
        
        # at the end
        self.to(device)
    
    def forward(self, x, logdet=None, reverse=False, reverse_clone=True):
        if not reverse:
            n,c,h,w = x.size()
            Z = []
            if logdet is None:
                logdet = torch.tensor(0.0,requires_grad=False,device=self.device,dtype=torch.float)
            for i in range( len(self.glow_modules) ):
                module_name = self.glow_modules[i].__class__.__name__
                if  module_name == "Squeeze":
                    x, logdet = self.glow_modules[i](x, logdet=logdet, reverse=False)
                elif  module_name == "Flow":
                    x, logdet, actloss = self.glow_modules[i](x, logdet=logdet, reverse=False)
                elif  module_name == "Split":
                    x, z = self.glow_modules[i](x, reverse=False)
                    Z.append(z)
                else:
                    raise "Unknown Layer"

            Z.append(x)
            
            if not self.init_resizer:
                self.sizes = [t.size() for t in Z]
                self.init_resizer = True
            return Z, logdet, actloss
        
        if reverse:
            if reverse_clone:
                x     = [x[i].clone().detach() for i in range(len(x))]
            else:
                x     = [x[i] for i in range(len(x))]
            x_rev = x[-1] # here x is z -> latent vector
            k = len(x)-2
            for i in range(len(self.glow_modules)-1,-1,-1 ):
                module_name = self.glow_modules[i].__class__.__name__
                if  module_name == "Split":
                    x_rev = self.glow_modules[i](x_rev,x[k], reverse=True)
                    k = k - 1
                elif  module_name == "Flow":
                    x_rev = self.glow_modules[i](x_rev, reverse=True)
                elif  module_name == "Squeeze":
                    x_rev = self.glow_modules[i](x_rev, reverse=True)
                else:
                    raise "Unknown Layer"
            return x_rev
        
    def nll_loss(self, x, logdet=None):
        n,c,h,w = x.size()
        z, logdet, actloss = self.forward(x,logdet=logdet,reverse=False)
        if not self.init_resizer:
            self.sizes = [t.size() for t in z]
            self.init_resizer = True
        z_ = [ z_.view(n,-1) for z_ in z]
        z_ = torch.cat(z_, dim=1)
        mean  = 0; logs = 0
        logdet += float(-np.log(256.) * h*w*c)
        logpz = -0.5*(logs*2. + ((z_- mean)**2)/np.exp(logs*2.) + float(np.log(2 * np.pi))).sum(-1)
        nll   = -(logdet + logpz).mean()
        nll   = nll / float(np.log(2.)*h*w*c)
        return nll, -logdet.mean().item(),-logpz.mean().item(), z_.mean().item(), z_.std().item()
    
    def preprocess(self, x, clone=False):
        if clone:
            x = x.detach().clone()
        n_bins = 2 ** self.n_bits_x
        x = torch.floor(x / 2 ** (8 - self.n_bits_x))
        x = x / n_bins - .5
        x = x + torch.tensor(np.random.uniform(0,1/n_bins,x.size()),dtype=torch.float,device=self.device)
        return x
    
    def postprocess(self, x, floor_clamp=True):
        n_bins = 2 ** self.n_bits_x
        if floor_clamp:
            x = torch.floor((x + 0.5)*n_bins)*(1./n_bins)
            x = torch.clamp(x, 0,1)
        else:
            x = x + 0.5
        return x
    
    def generate_z(self,n, mu=0,std=1,to_torch=True):
        # a function to reshape z so that it can be fed to the reverse method
        z_np = [np.random.normal(mu,std,[n]+list(size)[1:]) for size in self.sizes]
        if to_torch:
            z_t  = [torch.tensor(t,dtype=torch.float,device=self.device,requires_grad=False) for t in z_np]
            return z_np, z_t
        else:
            return z_np
        
    def flatten_z(self, z):
        n  = z[0].size()[0]
        z = [ z_.view(n,-1) for z_ in z]
        z = torch.cat(z, dim=1)
        return z
        
    def unflatten_z(self, z, clone=True):
        # z must be torch tensor
        n_elements = [np.prod(s[1:]) for s in self.sizes]
        z_unflatten = []
        start = 0
        for n, size in zip(n_elements, self.sizes):
            end = start + n
            z_   = z[:,start:end].view([-1]+list(size)[1:])
            if clone:
                z_   = z_.clone().detach()
            z_unflatten.append(z_)
            start = end
        return z_unflatten
    
    def set_actnorm_init(self):
        # a method to set actnorm to True to prevent re-initializing for resuming training
        for i in range( len(self.glow_modules) ):
                module_name = self.glow_modules[i].__class__.__name__
                if module_name == "Flow":
                    self.glow_modules[i].actnorm.initialized = True
                    self.glow_modules[i].coupling.net.actnorm1.initialized = True
                    self.glow_modules[i].coupling.net.actnorm2.initialized = True
                    

if __name__ == "__main__":
    size = (16,3,64,64)
    images = sio.imread_collection("./images/*.png")
    x = np.array([ img.astype("float")/255 for img in images ]).transpose([0,3,1,2])
    x = torch.tensor(x, device=device, dtype=torch.float, requires_grad=True)
    logdet = torch.tensor(0.0,requires_grad=False,device=device,dtype=torch.float)
    
    with torch.no_grad():
        glow = Glow((3,64,64),K=32,L=4,
                    coupling="affine",nn_init_last_zeros=True,
                    device=device)
        z,logdet, actloss = glow(x, logdet=logdet, reverse=False)
        x_rev = glow(z, reverse=True)
    print(torch.norm(x_rev - x).item())       
    reconstructed = x_rev.data.cpu().numpy().transpose([0,2,3,1])
    sio.imshow_collection(images)
    sio.imshow_collection(reconstructed)
