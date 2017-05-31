import numpy as np

N = 7
K_values = [3]

from heom_mean_F2_bias_drude_mode import do_the_calculation

for K in K_values:
    print "calculating for K="+str(K)
    do_the_calculation(N, K)
