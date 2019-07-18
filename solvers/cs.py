import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from skimage.transform import resize
import PIL
import skimage.io as sio
from glow.glow import Glow
from dcgan.dcgan import Generator
from .lasso_utils import celebA_estimators
import json
import os
from glob import glob
import easydict
from scipy.linalg import null_space
import warnings
warnings.filterwarnings("ignore")

def solveCS(args):
    if args.prior == 'glow':
        GlowCS(args)
    elif args.prior == 'dcgan':
        GANCS(args)
    elif args.prior == 'wavelet':
        WVTCS(args)
    elif args.prior == 'dct':
        DCTCS(args)
    else:
        raise "prior not defined correctly"


def GlowCS(args):
    if args.init_norms == None:
        args.init_norms = [None]*len(args.m)
    else:
        assert args.init_strategy == "random_fixed_norm", "init_strategy should be random_fixed_norm if init_norms is used"
    assert len(args.m) == len(args.gamma) == len(args.init_norms), "length of either m, gamma or init_norms are not same"
    loopOver = zip(args.m, args.gamma, args.init_norms)

    for m, gamma, init_norm in loopOver:
        skip_to_next = False # flag to skip to next loop if recovery is fails due to instability
        n                  = args.size*args.size*3
        modeldir           = "./trained_models/%s/glow"%args.model
        test_folder        = "./test_images/%s"%args.dataset
        save_path          = "./results/%s/%s"%(args.dataset,args.experiment)

        # loading dataset
        trans           = transforms.Compose([transforms.Resize((args.size,args.size)),transforms.ToTensor()])
        test_dataset    = datasets.ImageFolder(test_folder, transform=trans)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,drop_last=False,shuffle=False)
        
        # loading glow configurations
        config_path = modeldir+"/configs.json"
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        # sensing matrix
        A = np.random.normal(0,1/np.sqrt(m), size=(n,m))
        A = torch.tensor(A,dtype=torch.float, requires_grad=False, device=args.device)
        
        # regularizor
        gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)
    
        # adding noise
        if  args.noise == "random_bora":
            noise = np.random.normal(0,1,size=(args.batchsize,m))
            noise = noise * 0.1/np.sqrt(m)
            noise = torch.tensor(noise,dtype=torch.float,requires_grad=False, device=args.device)
        else:
            noise = np.random.normal(0,1,size=(args.batchsize,m))
            noise = noise / (np.linalg.norm(noise,2,axis=-1, keepdims=True)) * float(args.noise)
            noise = torch.tensor(noise, dtype=torch.float, requires_grad=False, device=args.device)
        
        # start solving over batches
        Original = []; Recovered = []; Z_Recovered = []; Residual_Curve = []; Recorded_Z = []
        for i, data in enumerate(test_dataloader):
            x_test = data[0]
            x_test = x_test.clone().to(device=args.device)
            n_test = x_test.size()[0]
            assert n_test == args.batchsize, "please make sure that no. of images are evenly divided by batchsize"
            
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
            
            # initializing z from Gaussian with std equal to init_std
            if args.init_strategy == "random":
                z_sampled = np.random.normal(0,args.init_std,[n_test,n])
                z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
            # intializing z from Gaussian and scaling its norm to init_norm
            elif args.init_strategy == "random_fixed_norm":
                z_sampled = np.random.normal(0,1,[n_test,n])
                z_sampled = z_sampled / np.linalg.norm(z_sampled, axis=-1, keepdims=True)
                z_sampled = z_sampled * init_norm
                z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
                print("z intialized with a norm equal to = %0.1f"%init_norm)
            # initializing z from pseudo inverse
            elif args.init_strategy == "pseudo_inverse":
                x_test_flat = x_test.view([-1,n])
                y_true      = torch.matmul(x_test_flat, A) + noise
                A_pinv      = torch.pinverse(A)
                x_pinv      = torch.matmul(y_true, A_pinv)
                x_pinv      = x_pinv.view([-1,3,args.size,args.size])
                x_pinv      = torch.clamp(x_pinv,0,1)
                z, _, _     = glow(glow.preprocess(x_pinv*255,clone=True))
                z           = glow.flatten_z(z).clone().detach()
                z_sampled   = torch.tensor(z, requires_grad=True, dtype=torch.float, device=args.device)
            # initializing z from a solution of lasso-wavelet 
            elif args.init_strategy == "lasso_wavelet":
                new_args    = {"batch_size":n_test, "lmbd":0.01,"lasso_solver":"sklearn"}
                new_args    = easydict.EasyDict(new_args)   
                estimator   = celebA_estimators.lasso_wavelet_estimator(new_args)
                x_ch_last   = x_test.permute(0,2,3,1)
                x_ch_last   = x_ch_last.contiguous().view([-1,n])
                y_true      = torch.matmul(x_ch_last, A) + noise
                x_lasso     = estimator(np.sqrt(2*m)*A.data.cpu().numpy(), np.sqrt(2*m)*y_true.data.cpu().numpy(), new_args)
                x_lasso     = np.array(x_lasso)
                x_lasso     = x_lasso.reshape(-1,64,64,3)
                x_lasso     = x_lasso.transpose(0,3,1,2)
                x_lasso     = torch.tensor(x_lasso, dtype=torch.float, device=args.device)
                z, _, _     = glow(x_lasso - 0.5)
                z           = glow.flatten_z(z).clone().detach()
                z_sampled   = torch.tensor(z, requires_grad=True, dtype=torch.float, device=args.device)
                print("z intialized from a solution of lasso-wavelet")
            # intializing z from null(A)
            elif args.init_strategy == "null_space":
                x_test_flat    = x_test.view([-1,n])
                x_test_flat_np = x_test_flat.data.cpu().numpy()
                A_np        = A.data.cpu().numpy()
                nullA       = null_space(A_np.T)
                coeff       = np.random.normal(0,1,(args.batchsize, nullA.shape[1]))            
                x_null      = np.array([(nullA * c).sum(axis=-1) for c in coeff])
                pert_norm   = 5 # <-- 5 gives optimal results --  bad initialization and not too unstable
                x_null      = x_null / np.linalg.norm(x_null, axis=1, keepdims=True) * pert_norm
                x_perturbed = x_test_flat_np + x_null
                # no clipping x_perturbed to make sure forward model is ||y-Ax|| is the same
                err         = np.matmul(x_test_flat_np,A_np) - np.matmul(x_perturbed,A_np)
                assert (err **2).sum() < 1e-6, "null space does not satisfy ||y-A(x+x0)|| <= 1e-6"
                x_perturbed = x_perturbed.reshape(-1,3,args.size,args.size)
                x_perturbed = torch.tensor(x_perturbed, dtype=torch.float, device=args.device)
                z, _, _     = glow(x_perturbed - 0.5)
                z           = glow.flatten_z(z).clone().detach()
                z_sampled   = torch.tensor(z, requires_grad=True, dtype=torch.float, device=args.device)
                print("z initialized from a point in null space of A")
            else:
                raise "Initialization strategy not defined"
                        
            # selecting optimizer
            if args.optim == "adam":
                optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
            elif args.optim == "lbfgs":
                optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)
            else:
                raise "optimizer not defined"
            
            # to be recorded over iteration
            psnr_t    = torch.nn.MSELoss().to(device=args.device)
            residual  = []; recorded_z = []
            # running optimizer steps
            for t in range(args.steps):
                def closure():
                    optimizer.zero_grad()
                    z_unflat    = glow.unflatten_z(z_sampled, clone=False)
                    x_gen       = glow(z_unflat, reverse=True, reverse_clone=False)
                    x_gen       = glow.postprocess(x_gen,floor_clamp=False)
                    x_test_flat = x_test.view([-1,n])
                    x_gen_flat  = x_gen.view([-1,n])
                    y_true      = torch.matmul(x_test_flat, A) + noise
                    y_gen       = torch.matmul(x_gen_flat, A) 
                    global residual_t
                    residual_t = ((y_gen - y_true)**2).sum(dim=1).mean()
                    z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
                    loss_t      = residual_t + z_reg_loss_t
                    psnr        = psnr_t(x_test, x_gen)
                    psnr        = 10 * np.log10(1 / psnr.item())
                    print("\rAt step=%0.3d|loss=%0.4f|residual=%0.4f|z_reg=%0.5f|psnr=%0.3f"%(t,loss_t.item(),residual_t.item(),z_reg_loss_t.item(), psnr),end="\r")
                    loss_t.backward()
                    return loss_t
                try:
                    optimizer.step(closure)
                    recorded_z.append(z_sampled.data.cpu().numpy())
                    residual.append(residual_t.item())
                except:
                    # try may not work due to instability in the reverse direction.
                    skip_to_next = True
                    break
            
            if skip_to_next:
                break
            
            # getting recovered and true images
            with torch.no_grad():
                x_test_np = x_test.data.cpu().numpy().transpose(0,2,3,1)
                z_unflat  = glow.unflatten_z(z_sampled, clone=False)
                x_gen     = glow(z_unflat, reverse=True, reverse_clone=False)
                x_gen     = glow.postprocess(x_gen,floor_clamp=False)
                x_gen_np  = x_gen.data.cpu().numpy().transpose(0,2,3,1)
                x_gen_np  = np.clip(x_gen_np,0,1)
                z_recov   = z_sampled.data.cpu().numpy()
            Original.append(x_test_np)
            Recovered.append(x_gen_np)
            Z_Recovered.append(z_recov)
            Residual_Curve.append(residual)
            Recorded_Z.append(recorded_z)
                    
            # freeing up memory for second loop
            glow.zero_grad()
            optimizer.zero_grad()
            del x_test, x_gen, optimizer, psnr_t, z_sampled, glow
            torch.cuda.empty_cache()
            print("\nbatch completed")
        
        if skip_to_next:
            print("\nskipping current loop due to instability or user triggered quit")
            continue
    
        # collecting everything together 
        Original     = np.vstack(Original)
        Recovered    = np.vstack(Recovered)
        Z_Recovered  = np.vstack(Z_Recovered)
        Recorded_Z   = np.vstack(Recorded_Z) 
        psnr         = [compare_psnr(x, y) for x,y in zip(Original, Recovered)]
        z_recov_norm = np.linalg.norm(Z_Recovered, axis=-1)
        
        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test        = %d\n"%len(Recovered)
        printout = printout + "\t n             = %d\n"%n
        printout = printout + "\t m             = %d\n"%m
        printout = printout + "\t gamma         = %0.6f\n"%gamma
        printout = printout + "\t optimizer     = %s\n"%args.optim
        printout = printout + "\t lr            = %0.3f\n"%args.lr
        printout = printout + "\t steps         = %0.3f\n"%args.steps
        printout = printout + "\t init_strategy = %s\n"%args.init_strategy
        printout = printout + "\t init_std      = %0.3f\n"%args.init_std
        if init_norm is not None:
            printout = printout + "\t init_norm     = %0.3f\n"%init_norm
        printout = printout + "\t z_recov_norm  = %0.3f\n"%np.mean(z_recov_norm)
        printout = printout + "\t PSNR          = %0.3f\n"%(np.mean(psnr))
        print(printout)
        
        # saving printout
        if args.save_metrics_text:
            with open("%s_cs_glow_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)
    
        
        # setting folder to save results in 
        if args.save_results:
            gamma = gamma.item()
            file_names = [name[0].split("/")[-1] for name in test_dataset.samples]
            if args.init_strategy == "random":
                save_path_template = save_path + "/cs_m_%d_gamma_%0.6f_steps_%d_lr_%0.3f_init_std_%0.2f_optim_%s"
                save_path = save_path_template%(m,gamma,args.steps,args.lr,args.init_std,args.optim)
            elif args.init_strategy == "random_fixed_norm":
                save_path_template = save_path+"/cs_m_%d_gamma_%0.6f_steps_%d_lr_%0.3f_init_%s_%0.3f_optim_%s"
                save_path          = save_path_template%(m,gamma,args.steps,args.lr,args.init_strategy,init_norm, args.optim)
            else:
                save_path_template = save_path + "/cs_m_%d_gamma_%0.6f_steps_%d_lr_%0.3f_init_%s_optim_%s"
                save_path          = save_path_template%(m,gamma,args.steps,args.lr,args.init_strategy,args.optim)
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
            # saving results now
            _ = [sio.imsave(save_path+"/"+name, x) for x,name in zip(Recovered,file_names)]
            Residual_Curve = np.array(Residual_Curve).mean(axis=0)
            np.save(save_path+"/original.npy", Original)
            np.save(save_path+"/recovered.npy", Recovered)
            np.save(save_path+"/z_recovered.npy", Z_Recovered)
            np.save(save_path+"/residual_curve.npy", Residual_Curve)                
            if init_norm is not None:
                np.save(save_path+"/Recorded_Z_init_norm_%d.npy"%init_norm, Recorded_Z) 
        torch.cuda.empty_cache()
        
        
def GANCS(args):
    assert len(args.m) == len(args.gamma), "length of gamma and m are different"
    assert args.init_strategy == 'random', "for dcgan, init_strategy can be only be random"
    loopOver = zip(args.m, args.gamma)    
    for m,gamma in loopOver:
        n  = 100 # size of the latent dim of DCGAN
        modeldir           = "./trained_models/%s/dcgan"%args.model
        test_folder        = "./test_images/%s"%args.dataset
        save_path          = "./results/%s/%s"%(args.dataset,args.experiment)
        
        # loading dataset
        trans           = transforms.Compose([transforms.Resize((args.size,args.size)),transforms.ToTensor()])
        test_dataset    = datasets.ImageFolder(test_folder, transform=trans)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,drop_last=False,shuffle=False)
        
        # sensing matrix
        A = np.random.normal(0,1/np.sqrt(m), size=(args.size*args.size*3,m))
        A = torch.tensor(A, dtype=torch.float, requires_grad=False, device=args.device)
        
        # regularizor
        gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)
            
        # adding noise
        if  args.noise == "random_bora":
            noise = np.random.normal(0,1,size=(args.batchsize,m))
            noise = noise * 0.1/np.sqrt(m)
            noise = torch.tensor(noise, dtype=torch.float, requires_grad=False, device=args.device)
        else:
            noise = np.random.normal(0,1,size=(args.batchsize,m))
            noise = noise / (np.linalg.norm(noise,2,axis=-1, keepdims=True)) * float(args.noise)
            noise = torch.tensor(noise, dtype=torch.float, requires_grad=False, device=args.device)
        
        # getting test images
        Original = []
        Recovered = []
        Residual_Curve = []
        for i, data in enumerate(test_dataloader):
            # getting back of data
            x_test = data[0]
            x_test = x_test.clone().cuda()
            n_test = x_test.size()[0]
            assert n_test == args.batchsize, "please make sure that no. of images are evenly divided by batchsize"
            # loading dcgan model
            generator = Generator(ngpu=1).to(device=args.device)
            generator.load_state_dict(torch.load(modeldir+'/dcgan_G.pt'))
            generator.eval()
            # measurement model
            z_sampled = np.random.normal(0,args.init_std,[n_test,n,1,1])
            z_sampled = torch.tensor(z_sampled, requires_grad=True, dtype=torch.float, device=args.device)
            psnr_t    = torch.nn.MSELoss().to(device=args.device)            
            # selecting optimizer
            if args.optim == "adam":
                optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
            elif args.optim == "lbfgs":
                optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)
            residual  = []
            for t in range(args.steps):
                def closure():
                    optimizer.zero_grad()
                    x_gen       = generator(z_sampled)
                    x_gen       = (x_gen + 1)/2
                    x_test_flat = x_test.view([-1,args.size*args.size*3])
                    x_gen_flat  = x_gen.view([-1,args.size*args.size*3])
                    y_true      = torch.matmul(x_test_flat, A) + noise
                    y_gen       = torch.matmul(x_gen_flat, A) 
                    global residual_t
                    residual_t  = ((y_gen - y_true)**2).sum(dim=1).mean()
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
            x_test_np = x_test.data.cpu().numpy().transpose(0,2,3,1)
            x_gen     = generator(z_sampled)
            x_gen     = (x_gen + 1)/2
            x_gen_np  = x_gen.data.cpu().numpy().transpose(0,2,3,1)
            x_gen_np  = np.clip(x_gen_np,0,1)
            Original.append(x_test_np)
            Recovered.append(x_gen_np)
            Residual_Curve.append(residual)
            # freeing up memory for second loop
            generator.zero_grad()
            optimizer.zero_grad()
            del x_test, x_gen, optimizer, psnr_t, z_sampled, generator
            torch.cuda.empty_cache()
            print("\nbatch completed")
            
        # collecting data togethers
        Original  = np.vstack(Original)
        Recovered = np.vstack(Recovered)
        psnr      = [compare_psnr(x, y) for x,y in zip(Original, Recovered)]
        # print performance metrics
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test    = %d\n"%len(Recovered)
        printout = printout + "\t n         = %d\n"%(args.size*args.size*3)
        printout = printout + "\t m         = %d\n"%m
        printout = printout + "\t PSNR      = %0.3f\n"%np.mean(psnr)
        print(printout)
        if args.save_metrics_text:
            with open("%s_cs_dcgan_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)
        
        # saving images
        if args.save_results:
            gamma = gamma.item()
            file_names = [name[0].split("/")[-1] for name in test_dataset.samples]
            save_path_template = save_path + "/cs_m_%d_gamma_%0.6f_steps_%d_lr_%0.3f_init_std_%0.2f_optim_%s"
            save_path = save_path_template%(m,gamma,args.steps,args.lr,args.init_std,args.optim)
            
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
                    
                    
            _ = [sio.imsave(save_path+"/"+name, x) for x,name in zip(Recovered,file_names)]
            Residual_Curve = np.array(Residual_Curve).mean(axis=0)
            np.save(save_path+"/residual_curve.npy", Residual_Curve)                
            np.save(save_path+"/original.npy", Original)
            np.save(save_path+"/recovered.npy", Recovered)
    

def WVTCS(args):
    loopOver = zip(args.m, args.gamma)    
    for m,gamma in loopOver:
        n                  = args.size*args.size*3
        test_folder        = "./test_images/%s/imgs"%args.dataset
        save_path          = "./results/%s/%s"%(args.dataset,args.experiment)
        divide_by          = 255                    # "max" or 255 or None
        # loading test images
        x_test    = [PIL.Image.open(p) for p in glob(test_folder+"/*")]
        file_names = [name.split("/")[-1] for name in glob(test_folder+"/*")]
        x_test    = [ img.resize((args.size,args.size),PIL.Image.BILINEAR) for img in x_test]
        x_test    = [np.array(img) for img in x_test]
        x_test    = np.array(x_test)
        # normalizing images
        if divide_by == "max":
            x_test = x_test / x_test.max(axis=(1,2,3),keepdims=True)
        elif divide_by == 255:
            x_test = x_test / 255
        elif divide_by == None:
            pass
        n_test    = x_test.shape[0]
        # arg parser to pass to solver methods
        new_args  = {"batch_size":n_test, "lmbd":gamma,"lasso_solver":"sklearn"}
        new_args  = easydict.EasyDict(new_args)    
        estimator = celebA_estimators.lasso_wavelet_estimator(new_args)
        A = np.random.normal(0,1/np.sqrt(m), size=(n,m))
        # adding noise
        if  args.noise == "random_bora":
            noise = np.random.normal(0,1,size=(n_test,m))
            noise = noise * 0.1/np.sqrt(m)
        else:
            noise = np.random.normal(0,1,size=(n_test,m))
            noise = noise / (np.linalg.norm(noise,2,axis=-1, keepdims=True)) * float(args.noise) 
        y_true    = np.matmul(x_test.reshape(n_test,-1), A) + noise
        x_hat     = estimator(np.sqrt(2*m)*A, np.sqrt(2*m)*y_true, new_args)
        x_hat     = np.array(x_hat)
        x_hat     = x_hat.reshape(-1,64,64,3)
        
        x_hat = np.clip(x_hat,0,1)
        
        psnr = [compare_psnr(x,xhat) for x,xhat in zip(x_test,x_hat)]
    
        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test    = %d\n"%len(x_hat)
        printout = printout + "\t n         = %d\n"%n
        printout = printout + "\t m         = %d\n"%m
        printout = printout + "\t solver    = lasso_wavelet\n"
        printout = printout + "\t gamma     = %0.8f\n"%gamma
        printout = printout + "\t PSNR      = %0.3f\n"%np.mean(psnr)
        print(printout)
        
        if args.save_metrics_text:
            with open("%s_cs_wvt_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)
        
        # saving images
        if args.save_results:
            save_path_template = save_path + "/cs_m_%d_lasso_wavelet_gamma_%0.8f"
            save_path = save_path_template%(m,gamma)           
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
                    
            _ = [sio.imsave(save_path+"/"+name, x) for x,name in zip(x_hat,file_names)]
#            _ = [sio.imsave(save_path+"/"+name.split(".")[0]+".jpg", x, quality=100) for x,name in zip(x_hat,file_names)]
            np.save(save_path+"/original.npy", x_test)
            np.save(save_path+"/recovered.npy", x_hat)
            


def DCTCS(args):
    loopOver = zip(args.m, args.gamma)    
    for m,gamma in loopOver:
        n                  = args.size*args.size*3
        test_folder        = "./test_images/%s/imgs"%args.dataset
        save_path          = "./results/%s/%s"%(args.dataset,args.experiment)
        divide_by          = 255                    # "max" or 255 or None
        # loading test images
        x_test    = [PIL.Image.open(p) for p in glob(test_folder+"/*")]
        file_names = [name.split("/")[-1] for name in glob(test_folder+"/*")]
        x_test    = [ img.resize((args.size,args.size),PIL.Image.BILINEAR) if (img.size[0]!=64) else img for img in x_test]
        x_test    = [np.array(img) for img in x_test]
        x_test    = np.array(x_test)
        # normalizing images
        if divide_by == "max":
            x_test = x_test / x_test.max(axis=(1,2,3),keepdims=True)
        elif divide_by == 255:
            x_test = x_test / 255
        elif divide_by == None:
            pass
        n_test    = x_test.shape[0]
        # arg parser to pass to solver methods
        new_args  = {"batch_size":n_test, "lmbd":gamma,"lasso_solver":"sklearn"}
        new_args  = easydict.EasyDict(new_args)    
        estimator = celebA_estimators.lasso_dct_estimator(new_args)
        A = np.random.normal(0,1/np.sqrt(m), size=(n,m))
        # adding noise
        if  args.noise == "random_bora":
            noise = np.random.normal(0,1,size=(n_test,m))
            noise = noise * 0.1/np.sqrt(m)
        else:
            noise = np.random.normal(0,1,size=(n_test,m))
            noise = noise / (np.linalg.norm(noise,2,axis=-1, keepdims=True)) * float(args.noise) 
        y_true    = np.matmul(x_test.reshape(n_test,-1), A) + noise
        x_hat     = estimator(np.sqrt(2*m)*A, np.sqrt(2*m)*y_true, new_args)
        x_hat     = np.array(x_hat)
        x_hat     = x_hat.reshape(-1,64,64,3)
        
        x_hat     = np.clip(x_hat,0,1)
        psnr = [compare_psnr(x,xhat) for x,xhat in zip(x_test,x_hat)]
        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test    = %d\n"%len(x_hat)
        printout = printout + "\t n         = %d\n"%n
        printout = printout + "\t m         = %d\n"%m
        printout = printout + "\t solver    = lasso_dct\n"
        printout = printout + "\t gamma     = %0.8f\n"%gamma
        printout = printout + "\t PSNR      = %0.3f \n"%np.mean(psnr)
        print(printout)
        
        if args.save_metrics_text:
            with open("%s_cs_dct_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)
        
        # saving images
        if args.save_results:
            save_path_template = save_path + "/cs_m_%d_lasso_dct_gamma_%0.8f"
            save_path = save_path_template%(m,gamma)           
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

            _ = [sio.imsave(save_path+"/"+name, x) for x,name in zip(x_hat,file_names)]                    
#            _ = [sio.imsave(save_path+"/"+name.split(".")[0]+".jpg", x, quality=100) for x,name in zip(x_hat,file_names)]
            np.save(save_path+"/original.npy", x_test)
            np.save(save_path+"/recovered.npy", x_hat)
