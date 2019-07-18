python solve_cs.py \
-prior dct \
-experiment birds_cs_lasso_dct \
-dataset birds \
-m        12288 10000 7500 5000 2500 1000  750  500  400  300  200  100   50   30   20 \
-gamma    0.01  0.01  0.01 0.01 0.01 0.01  0.01 0.01 0.01 0.01 0.01 0.01  0.01 0.01 0.01 \
-size 64 \
-save_metrics_text True \
-save_results True
