import numpy as np
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.OBOscillator import OBOscillator
import quant_mech.time_utils as tu

Gamma_L = 1.
Gamma_R = 0.025
bias = 0.2
T_c = 1. 
beta = 0.4
cutoff = 50.
K = 8
N = 6

def environment(reorg_energy, beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]

re_min = 1
re_max = 2
reorg_energy_values = np.logspace(re_min, re_max, 20)
indices = []
elements = []
indptrs = []

for i,E in enumerate(reorg_energy_values):
    print str(E) + ' at ' + str(tu.getTime())
    model_heom = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta, environment=environment(E, beta, K), \
                                    K=K, tc=True, trunc_level=N)
    heom_matrix = model_heom.heom_matrix()
    indices.append(heom_matrix.indices)
    elements.append(heom_matrix.data)
    indptrs.append(heom_matrix.indptr)
    
shape = heom_matrix.shape
jump_matrix = model_heom.jump_matrix().tocsr()
J_elements = jump_matrix.data
J_indices = jump_matrix.indices
J_indptrs = jump_matrix.indptr
dv_pops = model_heom.dv_pops

'''
Sparse data saved in CSR format
'''
np.savez('../data/F2_reorg_energy_heom_data_'+str(re_min)+'-'+str(re_max)+'.npz', reorg_energy_values=reorg_energy_values, indices=indices, \
                    elements=elements, indptrs=indptrs, shape=shape, jump_elements=J_elements, \
                    jump_indices=J_indices, jump_indptrs=J_indptrs, dv_pops=dv_pops)
    