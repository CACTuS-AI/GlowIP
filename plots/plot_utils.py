import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def histZNoisy(noise_std, max_images, glow, dataloader,batch_size,size):
    z_norm   = {"clean":[],"noisy":[]}
    n_images = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x        = data[0].cuda()*255
            x        = glow.preprocess(x)
            n        = x.size()[0]
            n_images = n_images + n
            noise = np.random.normal(0,noise_std,size=(batch_size,3,size,size))
            noise = torch.tensor(noise,dtype=torch.float,requires_grad=False).cuda()        
            x_noisy     = x + noise
            if i == 0:
                _ = glow(glow.preprocess(torch.zeros_like(x),clone=True))        
            z_clean, _, _ = glow.forward(x,logdet=None,reverse=False)
            z_clean       = glow.flatten_z(z_clean)
            z_noisy, _, _ = glow.forward(x_noisy,logdet=None,reverse=False)
            z_noisy       = glow.flatten_z(z_noisy)
            z_clean       = z_clean.norm(dim=-1).data.cpu().numpy()
            z_noisy       = z_noisy.norm(dim=-1).data.cpu().numpy()
            z_norm["clean"].extend(z_clean)
            z_norm["noisy"].extend(z_noisy)
            if max_images is not None:
                if n_images >= max_images:
                    break
    sns.distplot(z_norm["clean"], label="clean", hist=True, norm_hist=False, bins=15, 
                 hist_kws=dict(edgecolor="k", linewidth=0.5, linestyle="--"), color="slateblue")
    sns.distplot(z_norm["noisy"], label="noisy", hist=True, norm_hist=False, bins=15, 
                 hist_kws=dict(edgecolor="k", linewidth=0.5, linestyle="--"), color="darkseagreen")
    plt.xlim(100,240)
    plt.legend(fontsize=24)
    plt.xlabel("$|| z ||$", size=24)
    plt.ylabel("Frequency",size=24)
    plt.xticks(size=24)
    plt.yticks(size=24)
    plt.title("$\sigma = %0.2f$"%noise_std, fontsize=24)
    plt.tight_layout()
    
    
def getDxDzNaturalDirection(len_dz, glow, dataloader, size, max_images, batch_size):
    n    = size*size*3 
    Dx   = []
    Dz   = []
    n_images = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x        = data[0].cuda()*255
            n        = x.size()[0]
            n_images = n_images + n       
            x        = glow.preprocess(x)
            z, _, _ = glow(x,logdet=None,reverse=False)
            z       = glow.flatten_z(z)
            if i == 0:
                x_last = x
                z_last = z
                continue
            dz = z - z_last
            DDx = []; DDz=[]
            for dl in len_dz:
                ddz = dz / dz.norm(dim=-1, keepdim=True) * dl
                z_perturbed = z + ddz
                z_perturbed_unflat = glow.unflatten_z(z_perturbed)
                try:
                    x_perturbed = glow(z_perturbed_unflat,logdet=None,reverse=True)
                except:
                    print("\nAll NaNs at len_dz = %0.4f ... Moving to Next len_dz\n"%dl)
                    continue
                ddx          = (x.view(n,-1) - x_perturbed.view(n,-1)).norm(dim=-1)
                ddz          = ddz.norm(dim=-1)
                ddx          = ddx.data.cpu().numpy()
                ddz          = ddz.data.cpu().numpy()
                DDx.extend(ddx)
                DDz.extend(ddz)
            Dx.extend(DDx)
            Dz.extend(DDz)
            if max_images is not None:
                if n_images >= max_images:
                    break
    Dz = np.array(Dz)
    Dx = np.array(Dx)
    idx = np.argsort(Dz)
    Dz  = Dz[idx]
    Dx  = Dx[idx]
    Dz  = np.round(Dz, decimals=2)
    df = pd.DataFrame(data={"Dx":Dx, "Dz":Dz, "method":"natural-direction"})
    return df



def getDxDzRandomDirection(len_dz, n_rand_directions, glow, dataloader, size, max_images, batch_size,mean_per_lendz=False):
    n                  = size*size*3    
    Dx   = []
    Dz   = []
    LenDz = []
    with torch.no_grad():
        for ldz in len_dz:
            n_images = 0; DDx = []; DDz = []; LenDDz = []
            for i, data in enumerate(dataloader):
                x        = data[0].cuda()*255
                n        = x.size()[0]
                n_images = n_images + n       
                x        = glow.preprocess(x)
                z, _, _ = glow(x,logdet=None,reverse=False)
                z       = glow.flatten_z(z)
                dx = []; dz = []
                for r in range(n_rand_directions):
                    z_perturbed  = torch.rand(z.size(),device="cuda")
                    z_perturbed  = z_perturbed / z_perturbed.norm(dim=-1,keepdim=True) * ldz
                    z_perturbed  = z + z_perturbed
                    z_perturbed_unflat = glow.unflatten_z(z_perturbed)
                    try:
                        x_perturbed = glow(z_perturbed_unflat,logdet=None,reverse=True)
                    except:
                        continue
                    ddx          = (x.view(n,-1) - x_perturbed.view(n,-1)).norm(dim=-1)
                    ddz          = (z - z_perturbed).norm(dim=-1)
                    ddx          = ddx.data.cpu().numpy()
                    ddz          = ddz.data.cpu().numpy()
                    dx.append(ddx)
                    dz.append(ddz)
                if len(dx) == 0:
                    print("\nAll NaNs at len_dz = %0.4f ... Moving to Next len_dz\n"%ldz)
                    break
                dx = np.mean(dx, axis=0)
                dz = np.mean(dz, axis=0)
                DDx.extend(dx)
                DDz.extend(dz)
                LenDDz.extend([ldz]*n)
                if max_images is not None:
                    if n_images >= max_images:
                        break
            if mean_per_lendz:
                DDx    = [np.mean(DDx)]
                DDz    = [np.mean(DDz)]
                LenDDz = [np.mean(LenDDz)]
            Dx.extend(DDx)
            Dz.extend(DDz)
            LenDz.extend(LenDDz)
    Dx  = np.array(Dx)
    Dz  = np.array(Dz)
    LenDz = np.array(LenDz)
    idx = np.argsort(Dz)
    Dz  = Dz[idx]
    Dx  = Dx[idx]
    Dz  = np.round(Dz, decimals=2)
    df = pd.DataFrame(data={"Dz":Dz, "Dx":Dx, "method":"random-direction"})
    return df    



def glowLandscapeRandomDirection(which_image, m, dataloader, glow, gamma=0, n_grid=50, size=64, include_orig=False):
    n            = 64*64*3
    batch_size   = 1 # code support only 1 image for now
    device       = "cuda"
    # getting test images
    for i, data in enumerate(dataloader):
            if i == which_image:
                x_test = data[0]
                x_test = x_test.clone().cuda()
                n_test = x_test.shape[0]
                break
    # sensing matrix
    A = np.random.normal(0,1/np.sqrt(m), size=(n,m))
    A = torch.tensor(A,dtype=torch.float, requires_grad=False).cuda()
    # regularizor
    gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=device)
    # measurements from true 'z'
    y = torch.matmul(x_test.view([-1,n]),  A)
    # getting true minimizers
    with torch.no_grad():
        z_true,_,_  = glow(x_test - 0.5)    
        z_true      = glow.flatten_z(z_true)
    # choosing two random directions
    rand_dir_x = torch.randn_like(z_true)
    rand_dir_y = torch.randn_like(z_true)
    # normalizing the random directions to have same norm as z_true
    rand_dir_x = rand_dir_x / rand_dir_x.norm() * z_true.norm()
    rand_dir_y = rand_dir_y / rand_dir_y.norm() * z_true.norm()
    # finding point in x-y grid closest to origin
    if include_orig:
        orig_x = torch.dot(-z_true.view(-1), rand_dir_x.view(-1)) / rand_dir_x.norm()
        orig_y = torch.dot(-z_true.view(-1), rand_dir_y.view(-1)) / rand_dir_y.norm()
        orig_x = orig_x.item()
        orig_y = orig_y.item()
    # forward model loss
    def compute_loss(A,y,z,gamma):
        with torch.no_grad():
            z_unflat    = glow.unflatten_z(z, clone=False)
            x_gen       = glow(z_unflat, reverse=True, reverse_clone=False)
            x_gen       = glow.postprocess(x_gen,floor_clamp=False)
            x_gen_flat  = x_gen.view([-1,n])
            y_gen       = torch.matmul(x_gen_flat, A) 
            residual    = ((y_gen - y)**2).sum(dim=1)
            reg_loss    = gamma*z.norm(dim=1)
            loss        = residual + reg_loss
            return loss
    # Setting up Grid
    if include_orig:
        if orig_x < 0:
            lim    = -0.6 if (orig_x > -0.6) else (orig_x-0.2)
            grid_x = np.linspace(lim,0.6,n_grid)
        else:
            lim    = 0.6 if (orig_x < 0.6) else (orig_x+0.2)
            grid_x = np.linspace(-0.6,lim,n_grid)
        if orig_y < 0:
            lim    = -0.6 if (orig_y > -0.6) else (orig_y-0.2)
            grid_y = np.linspace(lim,0.6,n_grid)
        else:
            lim    = 0.6 if (orig_y < 0.6) else (orig_y+0.2)
            grid_y = np.linspace(-0.6,lim,n_grid)
    else:
        grid_x = np.linspace(-0.6,0.6,n_grid)
        grid_y = np.linspace(-0.6,0.6,n_grid)
    gridX,gridY    = np.meshgrid(grid_x,grid_y)
    gridZ          = np.zeros_like(gridX)
    # flattening grid to batch wise operation
    gridX  = gridX.reshape(-1,n_grid)
    gridY  = gridY.reshape(-1,n_grid)
    gridZ  = gridZ.reshape(-1,n_grid)
    # populating grid with loss 
    for i in range(len(gridX)):
        dx = [dx * rand_dir_x for dx in gridX[i]]
        dy = [dy * rand_dir_y for dy in gridY[i]]
        dx = torch.cat(dx)
        dy = torch.cat(dy)
        z  = z_true.repeat(n_grid,1)
        z  = z_true + dx + dy
        l  = compute_loss(A,y,z,gamma)
        l  = l.data.cpu().numpy()
        gridZ[i] = l
    # reshaping grid back to square matrix
    gridX = gridX.reshape(n_grid,n_grid)
    gridY = gridY.reshape(n_grid,n_grid)
    gridZ = gridZ.reshape(n_grid,n_grid)
    # plotting contours
    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    contours = plt.contour(gridX, gridY, gridZ, levels=20)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(gridZ, extent=[min(grid_x), max(grid_x), min(grid_y), max(grid_y)], origin='lower',cmap='RdGy', alpha=0.5)
    plt.colorbar()
    plt.scatter(0,0,c="k",s=4)
    if include_orig:
        plt.scatter(orig_x, orig_y)
    plt.xlabel("alpha",fontsize=15)
    plt.ylabel("beta",fontsize=15)
    plt.title("Glow Landscape in Random Directions")
    plt.tight_layout()
    # plotting 3D surface 
    ax = fig.add_subplot(122, projection="3d")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    surf = ax.plot_surface(gridX, gridY, gridZ, linewidth=0,cmap=plt.cm.coolwarm)
    plt.title("Glow Landscape in Random Directions")
    plt.tight_layout()
    plt.show()





def glowLandscapeNaturalDirection(which_image, m, dataloader, glow, gamma=0, n_grid=50, size=64, include_orig=False):
    n            = 64*64*3
    batch_size   = 1 # code support only 1 image for now
    device       = "cuda"    # getting test images
    x_test = []
    for i, data in enumerate(dataloader):
                x = data[0]
                x = x.clone().cuda()
                x_test.append(x)
    x_test = torch.cat(x_test,dim=0)
    x_test = x_test[np.random.choice(x_test.size(0),3)]
    A = np.random.normal(0,1/np.sqrt(m), size=(n,m))
    A = torch.tensor(A,dtype=torch.float, requires_grad=False).cuda()
    # regularizor
    gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=device)
    # measurements from true 'z'
    y      = torch.matmul(x_test[0].view([-1,n]),  A)
    # getting true minimizers
    with torch.no_grad():
        z,_,_  = glow(x_test - 0.5)    
        z      = glow.flatten_z(z)
    # choosing two random directions
    z0 = z[0:1]
    z1 = z[1:2]
    z2 = z[2:3]
    z_true        = z0
    natural_dir_x = z1
    natural_dir_y = z2
    # normalizing the random directions to have same norm as z_true
    natural_dir_x = natural_dir_x / natural_dir_x.norm() * z_true.norm()
    natural_dir_y = natural_dir_y / natural_dir_y.norm() * z_true.norm()
    # finding point in x-y grid closest to origin
    if include_orig:
        orig_x = torch.dot(-z_true.view(-1), natural_dir_x.view(-1)) / natural_dir_x.norm()
        orig_y = torch.dot(-z_true.view(-1), natural_dir_y.view(-1)) / natural_dir_y.norm()
        orig_x = orig_x.item()
        orig_y = orig_y.item()
    # forward model loss
    def compute_loss(A,y,z,gamma):
        with torch.no_grad():
            z_unflat    = glow.unflatten_z(z, clone=False)
            x_gen       = glow(z_unflat, reverse=True, reverse_clone=False)
            x_gen       = glow.postprocess(x_gen,floor_clamp=False)
            x_gen_flat  = x_gen.view([-1,n])
            y_gen       = torch.matmul(x_gen_flat, A) 
            residual    = ((y_gen - y)**2).sum(dim=1)
            reg_loss    = gamma*z.norm(dim=1)
            loss        = residual + reg_loss
            return loss
    # Setting up Grid
    x_lim  = 0.2
    y_lim  = 0.2
    if include_orig:
        if orig_x < 0:
            lim    = -x_lim if (orig_x > -x_lim) else (orig_x-0.2)
            grid_x = np.linspace(lim,x_lim,n_grid)
        else:
            lim    = x_lim if (orig_x < x_lim) else (orig_x+0.2)
            grid_x = np.linspace(-x_lim,lim,n_grid)
        if orig_y < 0:
            lim    = -y_lim if (orig_y > -y_lim) else (orig_y-0.2)
            grid_y = np.linspace(lim,y_lim,n_grid)
        else:
            lim    = y_lim if (orig_y < y_lim) else (orig_y+0.2)
            grid_y = np.linspace(-y_lim,lim,n_grid)
    else:
        grid_x = np.linspace(-x_lim,x_lim,n_grid)
        grid_y = np.linspace(-y_lim,y_lim,n_grid)
    gridX,gridY    = np.meshgrid(grid_x,grid_y)
    gridZ          = np.zeros_like(gridX)
    # flattening grid to batch wise operation
    gridX  = gridX.reshape(-1,n_grid)
    gridY  = gridY.reshape(-1,n_grid)
    gridZ  = gridZ.reshape(-1,n_grid)
    # populating grid with loss 
    for i in range(len(gridX)):
        dx = [dx * natural_dir_x for dx in gridX[i]]
        dy = [dy * natural_dir_y for dy in gridY[i]]
        dx = torch.cat(dx)
        dy = torch.cat(dy)
        z  = z_true.repeat(n_grid,1)
        z  = z_true + dx + dy
        l  = compute_loss(A,y,z,gamma)
        l  = l.data.cpu().numpy()
        gridZ[i] = l
    # reshaping grid back to square matrix
    gridX = gridX.reshape(n_grid,n_grid)
    gridY = gridY.reshape(n_grid,n_grid)
    gridZ = gridZ.reshape(n_grid,n_grid)
    # plotting contours
    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    contours = plt.contour(gridX, gridY, gridZ, levels=20)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(gridZ, extent=[min(grid_x), max(grid_x), min(grid_y), max(grid_y)], origin='lower',cmap='RdGy', alpha=0.5)
    plt.colorbar()
    plt.scatter(0,0,c="k",s=4)
    if include_orig:
        plt.scatter(orig_x, orig_y)
    plt.xlabel("alpha",fontsize=15)
    plt.ylabel("beta",fontsize=15)
    plt.title("Glow Landscape in Natural Directions")
    # plotting 3D surface 
    ax = fig.add_subplot(122, projection='3d')
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    surf = ax.plot_surface(gridX, gridY, gridZ, linewidth=0,cmap=plt.cm.coolwarm)
    plt.title("Glow Landscape in Natural Directions")
    plt.tight_layout()
    plt.show()
    
    
    
def dcganLossLandscape(which_image, m, dataloader, generator, gamma=0, n_grid=200, size=64, include_orig=False):
    n            = 64*64*3
    batch_size   = 1 # code support only 1 image for now
    device       = "cuda"
    # getting test images
    for i, data in enumerate(dataloader):
            if i == which_image:
                x_test = data[0]
                x_test = x_test.clone().cuda()
                n_test = x_test.shape[0]
                break
    # sensing matrix
    A = np.random.normal(0,1/np.sqrt(m), size=(n,m))
    A = torch.tensor(A,dtype=torch.float, requires_grad=False).cuda()
    # regularizor
    gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device="cuda")
    # measurements from true 'z'
    y = torch.matmul(x_test.view([-1,n]),  A)                
    # a function to compute loss
    def compute_loss(A,y,z,gamma):
        x_gen       = generator(z)
        x_gen       = (x_gen + 1)/2
        x_gen_flat  = x_gen.view([-1,64*64*3])
        y_gen       = torch.matmul(x_gen_flat, A) 
        residual    = ((y_gen - y)**2).sum(dim=1)
        reg_loss    = gamma*z.norm(dim=1)[:,0,0]
        loss        = residual + reg_loss
        return loss
    # solving cs to get z_hat
    z_hat = np.random.normal(0,0.1,[n_test,100,1,1])
    z_hat = torch.tensor(z_hat,requires_grad=True,dtype=torch.float,device="cuda")
    optimizer = torch.optim.LBFGS([z_hat], lr=1,)
    for i in range(20):
        def closure():
            optimizer.zero_grad()
            l = compute_loss(A,y,z_hat,gamma)
            l = l.mean()
            l.backward()
            return l
        optimizer.step(closure)
    x_hat = generator(z_hat)
    x_hat = (x_hat + 1 )/2
    x_hat = x_hat.data.cpu().numpy().transpose(0,2,3,1).squeeze()
    # choosing two random directions
    rand_dir_x = torch.randn_like(z_hat)
    rand_dir_y = torch.randn_like(z_hat)
    # normalizing the random directions to have same norm as z_true
    rand_dir_x = rand_dir_x / rand_dir_x.norm() * z_hat.norm()
    rand_dir_y = rand_dir_y / rand_dir_y.norm() * z_hat.norm()
    # finding point in x-y grid closest to origin
    if include_orig:
        orig_x = torch.dot(-z_hat.view(-1), rand_dir_x.view(-1)) / rand_dir_x.norm()
        orig_y = torch.dot(-z_hat.view(-1), rand_dir_y.view(-1)) / rand_dir_y.norm()
        orig_x = orig_x.item()
        orig_y = orig_y.item()
    # setting up grid
    if include_orig:
        if orig_x < 0:
            lim    = -0.6 if (orig_x > -0.6) else (orig_x-0.2)
            grid_x = np.linspace(lim,0.6,n_grid)
        else:
            lim    = 0.6 if (orig_x < 0.6) else (orig_x+0.2)
            grid_x = np.linspace(-0.6,lim,n_grid)
        if orig_y < 0:
            lim    = -0.6 if (orig_y > -0.6) else (orig_y-0.2)
            grid_y = np.linspace(lim,0.6,n_grid)
        else:
            lim    = 0.6 if (orig_y < 0.6) else (orig_y+0.2)
            grid_y = np.linspace(-0.6,lim,n_grid)
    else:
        grid_x = np.linspace(-0.6,0.6,n_grid)
        grid_y = np.linspace(-0.6,0.6,n_grid)
    gridX,gridY    = np.meshgrid(grid_x,grid_y)
    gridZ          = np.zeros_like(gridX)
    # flattening grid to batch wise operation
    gridX  = gridX.reshape(-1,n_grid)
    gridY  = gridY.reshape(-1,n_grid)
    gridZ  = gridZ.reshape(-1,n_grid)
    # populating grid with loss 
    for i in range(len(gridX)):
        dx = [dx * rand_dir_x for dx in gridX[i]]
        dy = [dy * rand_dir_y for dy in gridY[i]]
        dx = torch.cat(dx)
        dy = torch.cat(dy)
        z  = z_hat.repeat(50,1,1,1)
        z  = z_hat + dx + dy
        l  = compute_loss(A,y,z,gamma)
        l  = l.data.cpu().numpy()
        gridZ[i] = l
    # reshaping grid back to square matrix
    gridX = gridX.reshape(n_grid,n_grid)
    gridY = gridY.reshape(n_grid,n_grid)
    gridZ = gridZ.reshape(n_grid,n_grid)
    # plotting contours
    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    contours = plt.contour(gridX, gridY, gridZ, levels=20)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(gridZ,extent=[min(grid_x), max(grid_x), min(grid_y), max(grid_y)], origin='lower',cmap='RdGy', alpha=0.5)
    plt.colorbar()
    plt.scatter(0,0,c="k",s=4)
    if include_orig:
        plt.scatter(orig_x, orig_y)
    plt.xlabel("alpha",fontsize=15)
    plt.ylabel("beta",fontsize=15)
    plt.title("DCGAN Landscape")
    # plotting 3D surface 
    ax = fig.add_subplot(122, projection='3d')
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    surf = ax.plot_surface(gridX, gridY, gridZ, linewidth=0,cmap=plt.cm.coolwarm)
    plt.title("DCGAN Landscape")
    plt.tight_layout()
    plt.show()