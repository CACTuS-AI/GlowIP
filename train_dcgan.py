# the code for DCGAN was sourced from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import argparse
from dcgan.dcgan import *


def trainDCGAN(args):
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    dataroot = "./data/%s_preprocessed/train"%args.dataset
    save_path= "./trained_models/%s/dcgan"%args.dataset
    workers = 2
    batch_size = args.batchsize
    image_size = args.size
    nz = 100
    num_epochs = args.epochs
    lr = args.lr
    beta1 = 0.5
    ngpu = 1
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    device = args.device
    real_batch = next(iter(dataloader))
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)    
    if os.path.exists(save_path+"/dcgan_G.pt"):
        netG.load_state_dict(torch.load(save_path+"/dcgan_G.pt"))
        print("Loading weights of G")
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if os.path.exists(save_path+"/dcgan_D.pt"):
        netD.load_state_dict(torch.load(save_path+"/dcgan_D.pt"))
        print("Loading weights of D")
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
    
            netG.zero_grad()
            label.fill_(real_label)  
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1
            
        if i%5:            
            torch.save(netG.state_dict(), save_path+"/dcgan_G.pt")    
            torch.save(netD.state_dict(), save_path+"/dcgan_D.pt")    
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    
    HTML(ani.to_jshtml())
    
    real_batch = next(iter(dataloader))
    
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    
    
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

    torch.save(netG.state_dict(),  save_path+"/dcgan_G.pt")    
    torch.save(netD.state_dict(),  save_path+"/dcgan_D.pt")    
    
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train glow network')
    parser.add_argument('-dataset',type=str,help='the dataset to train the model on', default='celeba')
    parser.add_argument('-batchsize',type=bool,help='batch size for training',default=6)
    parser.add_argument('-size',type=int,help='images will be resized to this dimension',default=64)
    parser.add_argument('-lr',type=float,help='learning rate for training',default=0.0002)
    parser.add_argument('-epochs',type=int,help='epochs to train for',default=500)
    parser.add_argument('-device',type=str,help='device to use',default="cuda")    
    args = parser.parse_args()
    trainDCGAN(args)
    

