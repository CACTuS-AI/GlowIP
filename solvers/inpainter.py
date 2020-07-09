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



def solveInpainting(args):
    if args.prior == 'glow':
        GlowInpaint(args)
    elif args.prior == 'dcgan':
        GANInpaint(args)
    else:
        raise "prior not defined correctly"

               

def GlowInpaint(args):
    loopOver = zip(args.gamma)
    for gamma in loopOver:
        skip_to_next = False # flag to skip to next loop if recovery is fails due to instability
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
        gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)
        
        # getting test images
        Original  = []
        Recovered = []
        Masked    = []
        Mask      = []
        Residual_Curve = []
        for i, data in enumerate(test_dataloader):
            # getting batch of data
            x_test = data[0]
            x_test = x_test.clone().to(device=args.device)
            n_test = x_test.size()[0]
            assert n_test == args.batchsize, "please make sure that no. of images are evenly divided by batchsize"
            
            # generate mask 
            mask  = gen_mask(args.inpaint_method,args.size,args.mask_size)
            mask  = np.array([mask for i in range(n_test)])
            mask  = mask.reshape([n_test,1,args.size,args.size])
            mask  = torch.tensor(mask, dtype=torch.float, requires_grad=False, device=args.device)
                   
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
            # initializing z from image with noise filled only in masked region
            elif args.init_strategy == "noisy_filled":
                x_noisy_filled = x_test.clone().detach()
                noise          = np.random.normal(0,0.2, x_noisy_filled.size())
                noise          = torch.tensor(noise,dtype=torch.float,device=args.device)
                noise          = noise * (1-mask)
                x_noisy_filled = x_noisy_filled + noise
                x_noisy_filled = torch.clamp(x_noisy_filled, 0, 1)
                z, _, _        = glow(x_noisy_filled - 0.5)
                z              = glow.flatten_z(z).clone().detach()
                z_sampled      = z.clone().detach().requires_grad_(True)
            # initializing z from image with masked region inverted
            elif args.init_strategy == "inverted_filled":
                x_inverted_filled = x_test.clone().detach()
                missing_x         = x_inverted_filled.clone()
                missing_x         = missing_x.data.cpu().numpy()
                missing_x         = missing_x[:,:,::-1,::-1]
                missing_x         = torch.tensor(missing_x.copy(),dtype=torch.float,device=args.device)
                missing_x         = (1-mask)*missing_x
                x_inverted_filled = x_inverted_filled * mask
                x_inverted_filled = x_inverted_filled + missing_x
                z, _, _           = glow(x_inverted_filled - 0.5)
                z                 = glow.flatten_z(z).clone().detach()
                z_sampled         = z.clone().detach().requires_grad_(True)
            # initializing z from masked image ( masked region as zeros )
            elif args.init_strategy == "black_filled":
                x_black_filled = x_test.clone().detach()
                x_black_filled = mask * x_black_filled
                x_black_filled = x_black_filled * mask
                z, _, _        = glow(x_black_filled - 0.5)
                z              = glow.flatten_z(z).clone().detach()
                z_sampled      = z.clone().detach().requires_grad_(True)
            # initializing z from noisy complete image
            elif args.init_strategy == "noisy":
                x_noisy  = x_test.clone().detach()
                noise    = np.random.normal(0,0.05, x_noisy.size())
                noise    = torch.tensor(noise,dtype=torch.float,device=args.device)
                x_noisy  = x_noisy + noise
                x_noisy  = torch.clamp(x_noisy, 0, 1)
                z, _, _        = glow(x_noisy - 0.5)
                z              = glow.flatten_z(z).clone().detach()
                z_sampled      = z.clone().detach().requires_grad_(True)
            # initializing z from image with only noise in masked region
            elif args.init_strategy == "only_noise_filled":
                x_noisy_filled = x_test.clone().detach()
                noise          = np.random.normal(0,0.2, x_noisy_filled.size())
                noise          = torch.tensor(noise,dtype=torch.float,device=args.device)
                noise          = noise * (1-mask)
                x_noisy_filled = mask * x_noisy_filled + noise
                x_noisy_filled = torch.clamp(x_noisy_filled, 0, 1)
                z, _, _        = glow(x_noisy_filled - 0.5)
                z              = glow.flatten_z(z).clone().detach()
                z_sampled      = z.clone().detach().requires_grad_(True)
            else:
                raise "Initialization strategy not defined"
            
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
                    z_unflat       = glow.unflatten_z(z_sampled, clone=False)
                    x_gen          = glow(z_unflat, reverse=True, reverse_clone=False)
                    x_gen          = glow.postprocess(x_gen,floor_clamp=False)
                    x_masked_test  = x_test * mask
                    x_masked_gen   = x_gen  * mask
                    global residual_t
                    residual_t  = ((x_masked_gen - x_masked_test)**2).view(len(x_masked_test),-1).sum(dim=1).mean()
                    if args.z_penalty_unsquared:
                        z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
                    else:
                        z_reg_loss_t= gamma*(z_sampled.norm(dim=1)**2).mean()
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
            mask_np    = mask.data.cpu().numpy()
            x_masked_test  = x_test * mask
            x_masked_test_np = x_masked_test.data.cpu().numpy().transpose(0,2,3,1)
            x_masked_test_np = np.clip(x_masked_test_np,0,1)
            
            Original.append(x_test_np)
            Recovered.append(x_gen_np)
            Masked.append(x_masked_test_np)
            Residual_Curve.append(residual)
            Mask.append(mask_np)
            
            # freeing up memory for second loop
            glow.zero_grad()
            optimizer.zero_grad()
            del x_test, x_gen, optimizer, psnr_t, z_sampled, glow, mask,
            torch.cuda.empty_cache()
            print("\nbatch completed")
        
        if skip_to_next:
            print("\nskipping current loop due to instability or user triggered quit")
            continue
            
    
        # metric evaluations
        Original  = np.vstack(Original)
        Recovered = np.vstack(Recovered)
        Masked    = np.vstack(Masked)
        Mask      = np.vstack(Mask)
        psnr      = [compare_psnr(x, y) for x,y in zip(Original, Recovered)]
    
        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test         = %d\n"%len(Recovered)
        printout = printout + "\t inpaint_method = %s\n"%args.inpaint_method
        printout = printout + "\t mask_size      = %0.3f\n"%args.mask_size
        printout = printout + "\t gamma          = %0.6f\n"%gamma
        printout = printout + "\t PSNR           = %0.3f\n"%np.mean(psnr)
        print(printout)
        if args.save_metrics_text:
            with open("%s_inpaint_glow_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)
    
        
        # saving images
        if args.save_results:
            gamma = gamma.item()
            file_names = [name[0].split("/")[-1].split(".")[0] for name in test_dataset.samples]
            if args.init_strategy == 'random':
                save_path = save_path + "/inpaint_%s_masksize_%0.4f_gamma_%0.6f_steps_%d_lr_%0.3f_init_std_%0.2f_optim_%s"            
                save_path = save_path%(args.inpaint_method,args.mask_size,gamma,args.steps,args.lr,args.init_std,args.optim)
            else:
                save_path = save_path + "/inpaint_%s_masksize_%0.4f_gamma_%0.6f_steps_%d_lr_%0.3f_init_%s_optim_%s"
                save_path = save_path%(args.inpaint_method,args.mask_size,gamma,args.steps,args.lr,args.init_strategy,args.optim)
            
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
            _ = [sio.imsave(save_path+"/"+name+"_masked.jpg", x) for x,name in zip(Masked,file_names)]
            Residual_Curve = np.array(Residual_Curve).mean(axis=0)
            np.save(save_path+"/"+"residual_curve.npy", Residual_Curve)
            np.save(save_path+"/original.npy", Original)
            np.save(save_path+"/recovered.npy", Recovered)
            np.save(save_path+"/mask.npy", Mask)
            np.save(save_path+"/masked.npy", Masked)
        





def GANInpaint(args):
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
        Masked    = []
        Mask      = []
        Residual_Curve = []
        for i, data in enumerate(test_dataloader):
            # getting batch of data
            x_test = data[0]
            x_test = x_test.clone().to(device=args.device)
            n_test = x_test.size()[0]
            assert n_test == args.batchsize, "please make sure that no. of images are evenly divided by batchsize"

            # generate mask 
            mask  = gen_mask(args.inpaint_method,args.size,args.mask_size)
            mask  = np.array([mask for i in range(n_test)])
            mask  = mask.reshape([n_test,1,args.size,args.size])
            mask  = torch.tensor(mask,dtype=torch.float,requires_grad=False, device=args.device)
            
            # loading dcgan model
            generator = Generator(ngpu=1).to(device=args.device)
            generator.load_state_dict(torch.load(modeldir+'/dcgan_G.pt'))
            generator.eval()
            
            # initializing latent code z from Gaussian
            if args.init_strategy == "random":
                z_sampled = np.random.normal(0,args.init_std,[n_test,n,1,1])
                z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
            else:
                raise "only random initialization strategy is supported for inpainting in dcgan"
            
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
                    x_masked_test  = x_test * mask
                    x_masked_gen   = x_gen  * mask
                    global residual_t
                    residual_t  = ((x_masked_gen - x_masked_test)**2).view(len(x_masked_test),-1).sum(dim=1).mean()
                    if args.z_penalty_unsquared:
                        z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
                    else:
                        z_reg_loss_t= gamma*(z_sampled.norm(dim=1)**2).mean()
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
            mask_np    = mask.data.cpu().numpy()
            x_masked_test  = x_test * mask
            x_masked_test_np = x_masked_test.data.cpu().numpy().transpose(0,2,3,1)
            x_masked_test_np = np.clip(x_masked_test_np,0,1)
    
            Original.append(x_test_np)
            Recovered.append(x_gen_np)
            Masked.append(x_masked_test_np)
            Residual_Curve.append(residual)
            Mask.append(mask_np)
            # freeing up memory for second loop
            generator.zero_grad()
            optimizer.zero_grad()
            del x_test, x_gen, optimizer, psnr_t, z_sampled, generator, mask,
            torch.cuda.empty_cache()
            print("\nbatch completed")
            
    
        # metric evaluations
        Original  = np.vstack(Original)
        Recovered = np.vstack(Recovered)
        Masked    = np.vstack(Masked)
        Mask      = np.vstack(Mask)
        psnr      = [compare_psnr(x, y) for x,y in zip(Original, Recovered)]
    
        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test         = %d\n"%len(Recovered)
        printout = printout + "\t inpaint_method = %s\n"%args.inpaint_method
        printout = printout + "\t mask_size      = %0.3f\n"%args.mask_size
        printout = printout + "\t gamma          = %0.6f\n"%gamma
        printout = printout + "\t PSNR           = %0.3f\n"%np.mean(psnr)
        print(printout)
        if args.save_metrics_text:
            with open("%s_inpaint_dcgan_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)
    
        
        # saving images
        if args.save_results:
            gamma = gamma.item()
            file_names = [name[0].split("/")[-1].split(".")[0] for name in test_dataset.samples]
            save_path = save_path + "/inpaint_%s_masksize_%0.4f_gamma_%0.6f_steps_%d_lr_%0.3f_init_std_%0.2f_optim_%s"            
            save_path = save_path%(args.inpaint_method,args.mask_size,gamma,args.steps,args.lr,args.init_std,args.optim)
            
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
            _ = [sio.imsave(save_path+"/"+name+"_masked.jpg", x) for x,name in zip(Masked,file_names)]
            Residual_Curve = np.array(Residual_Curve).mean(axis=0)
            np.save(save_path+"/"+"residual_curve.npy", Residual_Curve)                
            np.save(save_path+"/original.npy", Original)
            np.save(save_path+"/recovered.npy", Recovered)
            np.save(save_path+"/mask.npy", Mask)
            
            

# a function to generate masks
def gen_mask(maskType, imgSize, masksize=0.25):
    # the larger the masksize, the bigger the mask
    image_shape = [imgSize, imgSize]
    if maskType == 'random':
        mask = np.ones(image_shape)
        mask[np.random.random(image_shape[:2]) < masksize] = 0.0
    elif maskType == 'center':
        center_scale = -(masksize - 1)/2 
        assert(center_scale <= 0.5)
        mask = np.ones(image_shape)
        l = int(imgSize*center_scale)
        u = int(imgSize*(1.0-center_scale))
        mask[l:u, l:u] = 0.0
    elif maskType == 'left':
        mask = np.ones(image_shape)
        c = imgSize #// 2
        masksize = 1 - masksize
        c = int(c * masksize)
        mask[:, c:] = 0.0
    elif maskType == 'bottom':
        mask = np.ones(image_shape)
        c = imgSize# // 2
        masksize = 1 - masksize 
        c = int(c * masksize)
        mask[c:, :] = 0.0
    else:
        assert(False)
    return mask