import argparse
from solvers.inpainter import solveInpainting



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='solve inpainting')
    parser.add_argument('-prior',type=str,help='choose with prior to use glow, dcgan', default='glow')
    parser.add_argument('-experiment', type=str, help='the name of experiment',default='celeba_inpaint_glow_init_std_0.0')
    parser.add_argument('-dataset', type=str, help='the dataset/images to use',default='celeba')
    parser.add_argument('-model', type=str, help='which model to use',default='celeba')
    parser.add_argument('-inpaint_method',type=str,help='type of mask to apply',default='center')
    parser.add_argument('-mask_size',type=float,help='size of the mask, b/w 0 and 1',default=0.3)
    parser.add_argument('-gamma',  type=float, nargs='+',help='regularizor',default=[0,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1,2.5,5,7.5,10,20,30,40,50])
    parser.add_argument('-optim', type=str, help='optimizer', default="lbfgs")
    parser.add_argument('-lr', type=float, help='learning rate', default=1)
    parser.add_argument('-steps',type=int,help='no. of steps to run', default=20)
    parser.add_argument('-batchsize',type=int, help='no. of images to solve in parallel as batches',default=6)
    parser.add_argument('-size',type=int, help='size of images to resize all images to', default=64)
    parser.add_argument('-device',type=str,help='device to use', default='cuda')
    parser.add_argument('-init_strategy',type=str,help="init strategy to use",default='random')
    parser.add_argument('-init_std', type=float,help='std of init_strategy is random', default=0)
    parser.add_argument('-save_metrics_text',type=bool, help='whether to save results to a text file',default=True)
    parser.add_argument('-save_results',type=bool,help='whether to save results after experiments conclude',default=True)
    parser.add_argument('-z_penalty_unsquared', action="store_true",help="use ||z|| if True else ||z||^2")
    args = parser.parse_args()
    solveInpainting(args)
    
