import numpy as np
import scipy.sparse as sp
import quant_mech.time_utils as tu
from counting_statistics.sparse.fcs_solver import FCSSolver

data = np.load('../data/F2_reorg_energy_heom_data.npz')
reorg_energy_values = data['reorg_energy_values']
elements = data['elements']
indices = data['indices']
indptrs = data['indptrs']
shape = data['shape']
jump_elements = data['jump_elements']
jump_indices = data['jump_indices']
jump_indptrs = data['jump_indptrs']
jump_matrix = sp.csr_matrix((jump_elements, jump_indices, jump_indptrs), shape=shape)
dv_pops = data['dv_pops']

mean_heom = np.zeros(reorg_energy_values.size)
F2_heom = np.zeros(reorg_energy_values.size)
coh_heom = np.zeros(reorg_energy_values.size)

for i,E in enumerate(reorg_energy_values):
    print str(E) + ' at ' + str(tu.getTime())
    heom_matrix = sp.csr_matrix((elements[i], indices[i], indptrs[i]))#, shape=())
    solver = FCSSolver(heom_matrix, jump_matrix, dv_pops)
    mean_heom[i] = solver.mean()
    F2_heom[i] = solver.second_order_fano_factor()
    #coh_heom[i] = np.abs(model_heom.heom_solver.extract_system_density_matrix(solver.ss)[1,2])
    
np.savez('../data/F2_reorg_energy_data.npz', reorg_energy_values=reorg_energy_values, mean_heom=mean_heom, \
                F2_heom=F2_heom, coh_heom=coh_heom)