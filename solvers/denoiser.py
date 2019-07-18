import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
import skimage.io as sio
from glow.glow import Glow
from dcgan.dcgan import Generator
import json
import os
import warnings
warnings.filterwarnings("ignore")



def solveDenoising(args):
    if args.prior == 'glow':
        GlowDenoiser(args)
    elif args.prior == 'dcgan':
        GANDenoiser(args)
    else:
        raise "prior not defined correctly"




def GlowDenoiser(args):
    loopOver = zip(args.gamma)
    for gamma in loopOver:
        skip_to_next  = False # flag to skip to next loop if recovery is fails due to instability
        n             = args.size*args.size*3
        modeldir      = "./trained_models/%s/glow"%args.model
        test_folder   = "./test_images/%s"%args.dataset
        save_path     = "./results/%s/%s"%(args.dataset,args.experiment)
        
        # loading dataset
        trans           = transforms.Compose([transforms.Resize((args.size,args.size)),transforms.ToTensor()])
        test_dataset    = datasets.ImageFolder(test_folder, transform=trans)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,drop_last=False,shuffle=False)
            
        # loading glow configurations
        config_path = modeldir+"/configs.json"
        with open(config_path, 'r') as f:
            configs = json.load(f)
            
        # regularizor
        gamma = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)
        
        # getting test images
        Original  = []
        Recovered = []
        Noisy     = []
        Residual_Curve = []
        for i, data in enumerate(test_dataloader):
            # getting batch of data
            x_test = data[0]
            x_test = x_test.clone().to(device=args.device)
            n_test = x_test.size()[0]
            assert n_test == args.batchsize, "please make sure that no. of images are evenly divided by batchsize"

            # noise to be added
            if args.noise == "gaussian":
                noise = np.random.normal(0,args.noise_std, size=(n_test,3,args.size,args.size))
                noise = torch.tensor(noise,dtype=torch.float,requires_grad=False, device=args.device)
            elif args.noise == "laplacian":
                noise = np.random.laplace(scale=args.noise_std, size=(n_test,3,args.size,args.size))
                noise = torch.tensor(noise,dtype=torch.float,requires_grad=False, device=args.device)
                raise "code only supports gaussian for now" #-> no noise type tag in the folder name
            else:
                raise "noise type not defined"
                
            # loading glow model
            glow = Glow((3,args.size,args.size),
                        K=configs["K"],L=configs["L"],
                        coupling=configs["coupling"],
                        n_bits_x=configs["n_bits_x"],
                        nn_init_last_zeros=configs["last_zeros"],
                        device=args.device)
            glow.load_state_dict(torch.load(modeldir+"/glowmodel.pt"))
            glow.eval()            
            
            # making a forward to record shapes of z's for reverse pass
            _ = glow(glow.preprocess(torch.zeros_like(x_test)))

            # initializing z from Gaussian
            if args.init_strategy == "random":
                z_sampled = np.random.normal(0,args.init_std,[n_test,n])
                z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
            # initializing z from noisy image
            elif args.init_strategy == "from-noisy":
                x_noisy     = x_test + noise
                z, _, _     = glow(glow.preprocess(x_noisy*255,clone=True))
                z           = glow.flatten_z(z)
                z_sampled   = z.clone().detach().requires_grad_(True)        
            else:
                raise "Initialization strategy not defined"
            
            # selecting optimizer
            if args.optim == "adam":
                optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
            elif args.optim == "lbfgs":
                optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)
            
            # to be recorded over iteration
            psnr_t    = torch.nn.MSELoss().to(device=args.device)
            residual  = []
    
            # running optimizer steps
            for t in range(args.steps):
                def closure():
                    optimizer.zero_grad()
                    z_unflat    = glow.unflatten_z(z_sampled, clone=False)
                    x_gen       = glow(z_unflat, reverse=True, reverse_clone=False)
                    x_gen       = glow.postprocess(x_gen,floor_clamp=False)
                    x_noisy     = x_test + noise
                    global residual_t
                    residual_t  = ((x_gen - x_noisy)**2).view(len(x_noisy),-1).sum(dim=1).mean()                        
                    if args.z_penalty_squared:
                        z_reg_loss_t= gamma*(z_sampled.norm(dim=1)**2).mean()
                    else:
                        z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
                    loss_t      = residual_t + z_reg_loss_t
                    psnr        = psnr_t(x_test, x_gen)
                    psnr        = 10 * np.log10(1 / psnr.item())
                    print("\rAt step=%0.3d|loss=%0.4f|residual=%0.4f|z_reg=%0.5f|psnr=%0.3f"%(t,loss_t.item(),residual_t.item(),z_reg_loss_t.item(), psnr),end="\r")
                    loss_t.backward()
                    return loss_t
                try:
                    optimizer.step(closure)
                    residual.append(residual_t.item())
                except:
                    skip_to_next = True
                    break
            
            if skip_to_next:
                break
            
            # getting recovered and true images
            x_test_np  = x_test.data.cpu().numpy().transpose(0,2,3,1)
            z_unflat   = glow.unflatten_z(z_sampled, clone=False)
            x_gen      = glow(z_unflat, reverse=True, reverse_clone=False)
            x_gen      = glow.postprocess(x_gen,floor_clamp=False)
            x_gen_np   = x_gen.data.cpu().numpy().transpose(0,2,3,1)
            x_gen_np   = np.clip(x_gen_np,0,1)
            x_noisy    = x_test + noise
            x_noisy_np = x_noisy.data.cpu().numpy().transpose(0,2,3,1)
            x_noisy_np = np.clip(x_noisy_np,0,1)
    
            Original.append(x_test_np)
            Recovered.append(x_gen_np)
            Noisy.append(x_noisy_np)
            Residual_Curve.append(residual)
            
            # freeing up memory for second loop
            glow.zero_grad()
            optimizer.zero_grad()
            del x_test, x_gen, optimizer, psnr_t, z_sampled, glow, noise,
            torch.cuda.empty_cache()
            print("\nbatch completed")
        
        if skip_to_next:
            print("\nskipping current loop due to instability or user triggered quit")
            continue
            
    
        # metric evaluations
        Original  = np.vstack(Original)
        Recovered = np.vstack(Recovered)
        Noisy     = np.vstack(Noisy)
        psnr      = [compare_psnr(x, y) for x,y in zip(Original, Recovered)]
    
        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test     = %d\n"%len(Recovered)
        printout = printout + "\t noise_std  = %0.4f\n"%args.noise_std
        printout = printout + "\t gamma      = %0.6f\n"%gamma
        printout = printout + "\t PSNR       = %0.3f\n"%np.mean(psnr)
        print(printout)
        if args.save_metrics_text:
            with open("%s_denoising_glow_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)
        
        # saving images
        if args.save_results:
            gamma = gamma.item()
            file_names = [name[0].split("/")[-1].split(".")[0] for name in test_dataset.samples]
            save_path = save_path + "/denoising_noisestd_%0.4f_gamma_%0.6f_steps_%d_lr_%0.3f_init_std_%0.2f_optim_%s"            
            save_path = save_path%(args.noise_std, gamma, args.steps, args.lr, args.init_std, args.optim)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                save_path_1 = save_path + "_1"
                if not os.path.exists(save_path_1):
                    os.makedirs(save_path_1)
                    save_path = save_path_1
                else:
                    save_path_2 = save_path + "_2"
                    if not os.path.exists(save_path_2):
                        os.makedirs(save_path_2)
                        save_path = save_path_2
            _ = [sio.imsave(save_path+"/"+name+"_recov.jpg", x) for x,name in zip(Recovered,file_names)]
            _ = [sio.imsave(save_path+"/"+name+"_noisy.jpg", x) for x,name in zip(Noisy,file_names)]
            Residual_Curve = np.array(Residual_Curve).mean(axis=0)
            np.save(save_path+"/residual_curve.npy", Residual_Curve)
            np.save(save_path+"/original.npy", Original)
            np.save(save_path+"/recovered.npy", Recovered)
            np.save(save_path+"/noisy.npy", Noisy)
            
            
            
            
def GANDenoiser(args):
    assert args.noise == "gaussian", "only Gaussian noise is supported in GANDenoiser"
    loopOver = zip(args.gamma)
    for gamma in loopOver:
        n             = 100
        modeldir      = "./trained_models/%s/dcgan"%args.model
        test_folder   = "./test_images/%s"%args.dataset
        save_path     = "./results/%s/%s"%(args.dataset,args.experiment)
        # loading dataset
        trans           = transforms.Compose([transforms.Resize((args.size,args.size)),transforms.ToTensor()])
        test_dataset    = datasets.ImageFolder(test_folder, transform=trans)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,drop_last=False,shuffle=False)    
        # regularizor
        gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)
        # getting test images
        Original  = []
        Recovered = []
        Noisy     = []
        Residual_Curve = []
        for i, data in enumerate(test_dataloader):
            # getting batch of data
            x_test = data[0]
            x_test = x_test.clone().to(device=args.device)
            n_test = x_test.size()[0]
            assert n_test == args.batchsize, "please make sure that no. of images are evenly divided by batchsize"

            # noise to be added
            noise = np.random.normal(0,args.noise_std,size=(n_test,3,args.size,args.size))
            noise = torch.tensor(noise,dtype=torch.float,requires_grad=False, device=args.device)        
                            
            # loading dcgan model
            generator = Generator(ngpu=1).to(device=args.device)
            generator.load_state_dict(torch.load(modeldir+'/dcgan_G.pt'))
            generator.eval()
                
            # initializing z's
            z_sampled = np.random.normal(0,args.init_std,[n_test,n,1,1])
            z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
            
            # selecting optimizer
            if args.optim == "adam":
                optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
            elif args.optim == "lbfgs":
                optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)
            
            # metrics to record over training
            psnr_t    = torch.nn.MSELoss().to(device=args.device)
            residual  = []
    
            # running optimizer steps
            for t in range(args.steps):
                def closure():
                    optimizer.zero_grad()
                    x_gen       = generator(z_sampled)
                    x_gen       = (x_gen + 1)/2
                    x_noisy     = x_test + noise
                    global residual_t
                    residual_t  = ((x_gen - x_noisy)**2).view(len(x_noisy),-1).sum(dim=1).mean()
                    z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
                    loss_t      = residual_t + z_reg_loss_t
                    psnr        = psnr_t(x_test, x_gen)
                    psnr        = 10 * np.log10(1 / psnr.item())
                    print("\rAt step=%0.3d|loss=%0.4f|residual=%0.4f|z_reg=%0.5f|psnr=%0.3f"%(t,loss_t.item(),residual_t.item(),z_reg_loss_t.item(), psnr),end="\r")
                    loss_t.backward()
                    return loss_t
                optimizer.step(closure)
                residual.append(residual_t.item())
                        
            # getting recovered and true images
            x_test_np  = x_test.data.cpu().numpy().transpose(0,2,3,1)
            x_gen      = generator(z_sampled)
            x_gen      = (x_gen + 1)/2
            x_gen_np   = x_gen.data.cpu().numpy().transpose(0,2,3,1)
            x_gen_np   = np.clip(x_gen_np,0,1)
            x_noisy    = x_test + noise
            x_noisy_np = x_noisy.data.cpu().numpy().transpose(0,2,3,1)
            x_noisy_np = np.clip(x_noisy_np,0,1)
    
            Original.append(x_test_np)
            Recovered.append(x_gen_np)
            Noisy.append(x_noisy_np)
            Residual_Curve.append(residual)
            
            # freeing up memory for second loop
            generator.zero_grad()
            optimizer.zero_grad()
            del x_test, x_gen, optimizer, psnr_t, z_sampled, generator, noise,
            torch.cuda.empty_cache()
            print("\nbatch completed")
    
        # metric evaluations
        Original  = np.vstack(Original)
        Recovered = np.vstack(Recovered)
        Noisy     = np.vstack(Noisy)
        psnr      = [compare_psnr(x, y) for x,y in zip(Original, Recovered)]
    
        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test     = %d\n"%len(Recovered)
        printout = printout + "\t noises_std = %0.4f\n"%args.noise_std
        printout = printout + "\t gamma      = %0.6f\n"%gamma
        printout = printout + "\t PSNR       = %0.3f\n"%np.mean(psnr)
        print(printout)
        if args.save_metrics_text:
            with open("%s_denoising_dcgan_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)
        
        # saving images
        if args.save_results:
            gamma = gamma.item()
            file_names = [name[0].split("/")[-1].split(".")[0] for name in test_dataset.samples]
            save_path = save_path + "/denoising_noisestd_%0.4f_gamma_%0.6f_steps_%d_lr_%0.3f_init_std_%0.2f_optim_%s"            
            save_path = save_path%(args.noise_std, gamma, args.steps, args.lr, args.init_std, args.optim)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                save_path_1 = save_path + "_1"
                if not os.path.exists(save_path_1):
                    os.makedirs(save_path_1)
                    save_path = save_path_1
                else:
                    save_path_2 = save_path + "_2"
                    if not os.path.exists(save_path_2):
                        os.makedirs(save_path_2)
                        save_path = save_path_2
                    
                    
            _ = [sio.imsave(save_path+"/"+name+"_recov.jpg", x) for x,name in zip(Recovered,file_names)]
            _ = [sio.imsave(save_path+"/"+name+"_noisy.jpg", x) for x,name in zip(Noisy,file_names)]
            Residual_Curve = np.array(Residual_Curve).mean(axis=0)
            np.save(save_path+"/"+"residual_curve.npy", Residual_Curve)
            np.save(save_path+"/original.npy", Original)
            np.save(save_path+"/recovered.npy", Recovered)
            np.save(save_path+"/noisy.npy", Noisy)
        