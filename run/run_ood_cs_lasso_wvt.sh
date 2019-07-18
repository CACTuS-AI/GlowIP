python solve_cs.py \
-prior wavelet \
-experiment ood_cs_lasso_wavelet \
-dataset ood \
-m       10000  7500  5000 2500 1000 \
-gamma    0.01  0.01  0.01 0.01 0.01 \
-size 64 \
-save_metrics_text True \
-save_results True
