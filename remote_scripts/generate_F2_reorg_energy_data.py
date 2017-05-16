import numpy as np
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.OBOscillator import OBOscillator
import quant_mech.time_utils as tu
from multiprocessing import Pool

Gamma_L = 1.
Gamma_R = 0.025
bias = 2.
T_c = 1. 
beta = 0.4
cutoff = 50.
K = 9
N = 6

def environment(reorg_energy, beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]

re_min = 1
re_max = 4
reorg_energy_values = np.logspace(re_min, re_max, 120)
indices = []
elements = []
indptrs = []

def generate_heom(reorg_energy):
    print str(reorg_energy) + ' at ' + str(tu.getTime())
    print ''
    model_heom = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta, environment=environment(reorg_energy, beta, K), \
                                    K=K, tc=True, trunc_level=N)
    heom_matrix = model_heom.heom_matrix()
    np.savez('data/F2_reorg_energy_heom_data_K'+str(K)+'_vals_'+str(re_min)+'-'+str(re_max)+'_'+str(reorg_energy)+'.npz', reorg_energy=reorg_energy, \
                        indices=heom_matrix.indices, elements=heom_matrix.data, indptrs=heom_matrix.indptr)
     
if __name__ == '__main__':
    pool = Pool(4)
    pool.map(generate_heom, reorg_energy_values)

'''
Sparse data saved in CSR format
'''
model_heom = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta, environment=environment(1., beta, K), \
                                K=K, tc=True, trunc_level=N)
jump_matrix = model_heom.jump_matrix().tocsr()
jump_elements = jump_matrix.data
jump_indices = jump_matrix.indices
jump_indptrs = jump_matrix.indptr
dv_pops = model_heom.dv_pops
shape = jump_matrix.shape

np.savez('data/F2_reorg_energy_heom_data_K'+str(K)+'_vals_'+str(re_min)+'-'+str(re_max)+'.npz', reorg_energy_values=reorg_energy_values, \
         shape=shape, jump_elements=jump_elements, jump_indices=jump_indices, jump_indptrs=jump_indptrs, dv_pops=dv_pops)
    
