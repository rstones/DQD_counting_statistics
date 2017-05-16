import numpy as np
import scipy.sparse as sp
import quant_mech.time_utils as tu
from counting_statistics.sparse.fcs_solver import FCSSolver
from multiprocessing import Pool

re_min = 1
re_max = 4
K = 9

data = np.load('data/F2_reorg_energy_heom_data_K'+str(K)+'_vals_'+str(re_min)+'-'+str(re_max)+'.npz')
reorg_energy_values = data['reorg_energy_values']
# elements = data['elements']
# indices = data['indices']
# indptrs = data['indptrs']
shape = data['shape']
jump_elements = data['jump_elements']
jump_indices = data['jump_indices']
jump_indptrs = data['jump_indptrs']
jump_matrix = sp.csr_matrix((jump_elements, jump_indices, jump_indptrs), shape=shape)
dv_pops = data['dv_pops']

mean_heom = np.zeros(reorg_energy_values.size)
F2_heom = np.zeros(reorg_energy_values.size)
coh_heom = np.zeros(reorg_energy_values.size)
results = np.zeros((reorg_energy_values.size, 3))

def calculate_FCS(reorg_energy):
    print str(reorg_energy) + ' at ' + str(tu.getTime())
    data_pt = np.load('data/F2_reorg_energy_heom_data_K'+str(K)+'_vals_'+str(re_min)+'-'+str(re_max)+'_'+str(reorg_energy)+'.npz')
    elements = data_pt['elements']
    indices = data_pt['indices']
    indptrs = data_pt['indptrs']
    heom_matrix = sp.csr_matrix((elements, indices, indptrs), shape=shape)
    solver = FCSSolver(heom_matrix, jump_matrix, dv_pops)
    mean_heom = solver.mean()
    F2_heom = solver.second_order_fano_factor()
    #coh_heom[i] = np.abs(model_heom.heom_solver.extract_system_density_matrix(solver.ss)[1,2])
    return [mean_heom, F2_heom, 0]
    
if __name__ == '__main__':
    pool = Pool(2)
    results = np.array(pool.map(calculate_FCS, reorg_energy_values))
        
np.savez('data/F2_reorg_energy_data_K'+str(K)+'_vals_'+str(re_min)+'-'+str(re_max)+'.npz', reorg_energy_values=reorg_energy_values, mean_heom=results.T[0], \
                F2_heom=results.T[1], coh_heom=results.T[2])
