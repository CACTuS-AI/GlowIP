python solve_denoising.py \
-prior glow \
-experiment celeba_denoising_glow_noisestd_0.20 \
-dataset celeba \
-model celeba \
-gamma 0 0.01 0.025 0.05 0.075 0.1 0.25 0.5 0.75 1 2.5 5 7.5 10 20 30 40 50 \
-optim lbfgs \
-lr 1 \
-steps 20 \
-batchsize 6 \
-noise gaussian \
-noise_std 0.2 \
-init_strategy random  \
-init_std 0 

