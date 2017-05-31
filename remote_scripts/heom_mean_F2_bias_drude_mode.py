import numpy as np
import scipy.constants as constants
from counting_statistics.sparse.fcs_solver import FCSSolver
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
import quant_mech.time_utils as tu
from quant_mech.UBOscillator import UBOscillator
from quant_mech.OBOscillator import OBOscillator

from multiprocessing import Pool

Gamma_L = 1.
Gamma_R = 0.025
bias = 0
T_c = 1.

beta = [0.8, 0.4, 0.1]

mode_freq = 10.
hr_factor = 0.1
damping = 5.

drude_reorg_energy = 0.015
drude_cutoff = 50.

# N = 5
# K = 3

def do_the_calculation(N, K):

    drude = False
    
    def environment(beta, K):
        if drude:
            env = [(), \
                (OBOscillator(drude_reorg_energy, drude_cutoff, beta, K=K), UBOscillator(mode_freq, hr_factor, damping, beta, K=K),), \
                (OBOscillator(drude_reorg_energy, drude_cutoff, beta, K=K), UBOscillator(mode_freq, hr_factor, damping, beta, K=K),)]
        else:
            env = [(), (UBOscillator(mode_freq, hr_factor, damping, beta, K=K),), (UBOscillator(mode_freq, hr_factor, damping, beta, K=K),)]
        return env
    
    model = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], environment=environment(beta[0], K), \
                                            K=K, tc=True, trunc_level=N)
    bias_values = np.linspace(-10, 10, 20)
    mean = np.zeros((len(beta), bias_values.size))
    F2 = np.zeros((len(beta), bias_values.size))
    
    print "Starting calculation at " + str(tu.getTime())
    
    for j,B in enumerate(beta):
        print 'for beta = ' + str(B) + ' at ' + str(tu.getTime())
        model.beta = B
        model.environment = environment(B, K)
        for i,E in enumerate(bias_values):
            print E
            model.bias = E
            solver = FCSSolver(model.heom_matrix(), model.jump_matrix(), model.dv_pops)
            try:
                mean[j,i] = solver.mean()
                F2[j,i] = solver.second_order_fano_factor()
            except RuntimeError:
                print "SINGULAR ERROR!!!!!!!"
                
    print "finished calculation at " + str(tu.getTime())
    
    np.savez('../data/DQD_HEOM_mean_F2_bias_UBO'+('_OBO' if drude else '')+'_N'+str(N)+'_K'+str(K)+'_data.npz', \
             bias_values=bias_values, beta=beta, mean=mean, F2=F2)

