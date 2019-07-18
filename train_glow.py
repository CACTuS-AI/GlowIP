import torch 
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from glow.glow import Glow
import numpy as np
import skimage.io as sio
import matplotlib.pyplot as plt
import os
import json
import argparse


def trainGlow(args):
    save_path   = "./trained_models/%s/glow"%args.dataset
    training_folder = "./data/%s_preprocessed/train"%args.dataset
    
    # setting up configs as json
    config_path = save_path+"/configs.json"
    configs     = {"K":args.K,
                   "L":args.L,
                   "coupling":args.coupling,
                   "last_zeros":args.last_zeros,
                   "batchsize":args.batchsize,
                   "size":args.size,
                   "lr": args.lr,
                   "n_bits_x":args.n_bits_x,
                   "warmup_iter":args.warmup_iter}
    
    if not os.path.exists(save_path):
        print("creating directory to save model weights")
        os.makedirs(save_path)
    
    
    # loading pre-trained model to resume training
    if os.path.exists(save_path+"/glowmodel.pt"):
        print("loading previous model and saved configs to resume training ...")
        with open(config_path, 'r') as f:
            configs = json.load(f)
        glow = Glow((3,configs["size"],configs["size"]),
                    K=configs["K"],L=configs["L"],
                    coupling=configs["coupling"],
                    n_bits_x=configs["n_bits_x"],
                    nn_init_last_zeros=configs["last_zeros"],
                    device=args.device)
        glow.load_state_dict(torch.load(save_path+"/glowmodel.pt"))
        print("pre-trained model and configs loaded successfully")
        glow.set_actnorm_init()
        print("actnorm initialization flag set to True to avoid data dependant re-initialization")
        glow.train()
    
    else:
        # creating and initializing glow model
        print("creating and initializing model for training")
        glow = Glow((3,args.size,args.size),
                    K=args.K,L=args.L,coupling=args.coupling,n_bits_x=args.n_bits_x,
                    nn_init_last_zeros=args.last_zeros,
                    device=args.device)
        glow.train()
        print("saving configs as json file")
        with open(config_path, 'w') as f:
            json.dump(configs, f, sort_keys=True, indent=4, ensure_ascii=False)
            
    
    # setting up dataloader
    print("setting up dataloader for the training data")
    trans      = transforms.Compose([transforms.Resize((args.size,args.size)), 
                                     transforms.ToTensor()])
    dataset    = datasets.ImageFolder(training_folder, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize,
                                                drop_last=True, shuffle=True)
    
    
    
    # setting up optimizer and learning rate scheduler
    opt          = torch.optim.Adam(glow.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",
                                                              factor=0.5,
                                                              patience=1000,
                                                              verbose=True,
                                                              min_lr=1e-8)
    
    # starting training code here
    print("+-"*10,"starting training","-+"*10)
    global_step = 0
    global_loss = []
    warmup_completed = False
    for i in range(args.epochs):
        Loss_epoch = []
        for j, data in enumerate(dataloader):
            opt.zero_grad()
            glow.zero_grad()
            # loading batch
            x = data[0].to(device=args.device)*255
            # pre-processing data
            x = glow.preprocess(x)
            # computing loss: "nll"
            n,c,h,w = x.size()
            nll,logdet,logpz,z_mu,z_std = glow.nll_loss(x)
            # skipping first batch due to data dependant initialization (if not initialized)
            if global_step == 0:
                global_step += 1
                continue
            # backpropogating loss and gradient clipping
            nll.backward()
            torch.nn.utils.clip_grad_value_(glow.parameters(), 5)
            grad_norm = torch.nn.utils.clip_grad_norm_(glow.parameters(), 100)
            # linearly increase learning rate till warmup_iter upto args.lr
            if global_step <= args.warmup_iter:
                warmup_lr = args.lr / args.warmup_iter * global_step
                for params in opt.param_groups:
                    params["lr"] = warmup_lr
            # taking optimizer step                        
            opt.step()
            # learning rate scheduling after warm up iterations
            if global_step > args.warmup_iter:
                lr_scheduler.step(nll)
                if not warmup_completed:
                    if args.warmup_iter == 0:
                        print("no model warming...")
                    else:
                        print("\nwarm up completed")
                warmup_completed = True
            # printing training metrics 
            print("\repoch=%0.2d..nll=%0.2f..logdet=%0.2f..logpz=%0.2f..mu=%0.2f..std=%0.2f..gradnorm=%0.2f"
                  %(i,nll.item(),logdet,logpz,z_mu,z_std,grad_norm),end="\r")
            # saving generated samples during training
            try:
                if j % args.sample_freq == 0:
                    plt.plot(global_loss)
                    plt.xlabel("iterations",size=15)
                    plt.ylabel("nll",size=15)
                    plt.savefig(save_path+"/nll_training_curve.jpg")
                    plt.close()
                    with torch.no_grad():
                        z_sample, z_sample_t = glow.generate_z(n=10,mu=0,std=0.7,to_torch=True)
                        x_gen = glow(z_sample_t, reverse=True)
                        x_gen = glow.postprocess(x_gen)
                        x_gen = make_grid(x_gen,nrow=int(np.sqrt(len(x_gen))))
                        x_gen = x_gen.data.cpu().numpy()
                        x_gen = x_gen.transpose([1,2,0])
                        if x_gen.shape[-1] == 1:
                            x_gen = x_gen[...,0]
                        if not os.path.exists(save_path+"/samples_training"):
                            os.makedirs(save_path+"/samples_training")
                        x_gen = (x_gen * 255).astype("uint8")
                        sio.imsave(save_path+"/samples_training/%0.6d.jpg"%global_step, x_gen )
            except:
                print("\n failed to sample from glow at global step = %d"%global_step)
            global_step = global_step + 1
            global_loss.append(nll.item())
            if global_step % args.save_freq == 0:
                torch.save(glow.state_dict(), save_path+"/glowmodel.pt")
        
#    # model visualization 
#    temperature = [0.1,0.3,0.4,0.5,0.7,0.8, 0.9]
#    for temp in temperature:
#        with torch.no_grad():
#            glow.eval()
#            z_sample, z_sample_t = glow.generate_z(n=10,mu=0,std=temp,to_torch=True)
#            x_gen = glow(z_sample_t, reverse=True)
#            x_gen = glow.postprocess(x_gen)
#            x_gen = make_grid(x_gen,nrow=int(np.sqrt(len(x_gen))))
#            x_gen = x_gen.data.cpu().numpy()
#            x_gen = x_gen.transpose([1,2,0])
#            if x_gen.shape[-1] == 1:
#                x_gen = x_gen[...,0]
#            plt.figure()
#            plt.title("temperature = %0.1f"%temp,fontsize=15)
#            plt.axis("off")
#            plt.imshow(x_gen)
            
    # saving model weights
    torch.save(glow.state_dict(), save_path+"/glowmodel.pt")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train glow network')
    parser.add_argument('-dataset',type=str,help='the dataset to train the model on', default='celeba')
    parser.add_argument('-K',type=int,help='no. of steps of flow',default=48)
    parser.add_argument('-L',type=int,help='no. of time squeezing is performed',default=4)
    parser.add_argument('-coupling',type=str,help='type of coupling layer to use',default='affine')
    parser.add_argument('-last_zeros',type=bool,help='whether to initialize last layer ot NN with zeros',default=True)
    parser.add_argument('-batchsize',type=bool,help='batch size for training',default=6)
    parser.add_argument('-size',type=int,help='images will be resized to this dimension',default=64)
    parser.add_argument('-lr',type=float,help='learning rate for training',default=1e-4)
    parser.add_argument('-n_bits_x',type=int,help='requantization of training images',default=5)
    parser.add_argument('-epochs',type=int,help='epochs to train for',default=1000)
    parser.add_argument('-warmup_iter',type=int,help='no. of warmup iterations',default=10000)
    parser.add_argument('-sample_freq',type=int,help='sample after every save_freq',default=50)
    parser.add_argument('-save_freq',type=int,help='save after every save_freq',default=1000)
    parser.add_argument('-device',type=str,help='whether to use',default="cuda")    
    args = parser.parse_args()
    trainGlow(args)
    


