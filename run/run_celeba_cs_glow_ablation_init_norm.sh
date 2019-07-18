python solve_cs.py \
-prior glow \
-experiment celeba_cs_glow_ablation_gamma_random_fixed_norm \
-dataset celeba \
-model celeba \
-m          5000 5000  5000  5000   5000  5000 5000 5000 5000 5000 5000 5000\
-gamma      0    0     0     0      0     0    0    0    0    0    0    0 \
-init_norms 0    10    20    30     40    50   60   70   80   90   100  120 \
-optim lbfgs \
-lr 0.1 \
-steps 30 \
-batchsize 6 \
-size 64 \
-device cuda \
-init_strategy random_fixed_norm \
-save_metrics_text True \
-save_results True







