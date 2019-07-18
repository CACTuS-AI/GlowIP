python solve_cs.py \
-prior glow \
-experiment celeba_cs_glow_ablation_gamma_init_std_0.1 \
-dataset celeba \
-model celeba \
-m     5000 5000  5000  5000   5000  5000 5000 5000 5000 5000 5000 \
-gamma 0    1e-06 1e-05 0.0001 0.001 0.01 0.1  0.25 0.5  0.75 1 \
-optim lbfgs \
-lr 0.1 \
-steps 30 \
-batchsize 6 \
-size 64 \
-device cuda \
-init_strategy random \
-init_std 0.1   \
-save_metrics_text True \
-save_results True







