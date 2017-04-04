import numpy as np
import scipy.constants as constants
from counting_statistics.sparse.fcs_solver import FCSSolver
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.OBOscillator import OBOscillator
import quant_mech.time_utils as tu
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
import sys

# get task_id
task_id = sys.argv[1]
try:
    task_id = int(task_id)
except TypeError:
    print 'ERROR: Task id was None'
    sys.exit(0)
    
stride = 1

print 'JOB ' + str(task_id) + ': starting calculation at ' + str(tu.getTime())

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0.2
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature][1:2]
beta = beta[0]
#reorg_energy = 0.000147
cutoff = 5. # meV

N = 6
K = 6
data = np.load('../data/F2_reorg_energy_heom_data_N'+str(N)+'_K'+str(K)+'.npz')
reorg_energy_values = data['reorg_energy_values']
n_vectors = data['n_vectors']
higher_coupling_elements = data['higher_coupling_elements'][task_id-1]
higher_coupling_row_indices = data['higher_coupling_row_indices']
higher_coupling_column_indices = data['higher_coupling_column_indices']
lower_coupling_elements = data['lower_coupling_elements'][task_id-1]
lower_coupling_row_indices = data['lower_coupling_row_indices']
lower_coupling_column_indices = data['lower_coupling_column_indices']

def environment(reorg_energy, beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]

# F2 = np.zeros(stride)
# coh = np.zeros(stride, dtype='complex128')
# mean = np.zeros(stride)

reorg_energy = reorg_energy_values[task_id-1]
model = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta, \
                            environment=environment(reorg_energy, beta, K), \
                            K=K, tc=True, trunc_level=N)
heom_matrix = model.heom_solver.construct_hierarchy_matrix_no_numba(n_vectors, \
                                    higher_coupling_elements, higher_coupling_row_indices, higher_coupling_column_indices, \
                                    lower_coupling_elements, lower_coupling_row_indices, lower_coupling_column_indices)
try:
    solver = FCSSolver(heom_matrix, model.jump_matrix(), model.dv_pops)
    mean = solver.mean()
    F2 = solver.second_order_fano_factor()
    coh = model.heom_solver.extract_system_density_matrix(solver.ss)[1,2]
except ArpackNoConvergence:
    print "Convergence error!"
        
print 'JOB ' + str(task_id) + ': finished calculation at ' + str(tu.getTime())
    
np.savez('../data/HEOM_F2_reorg_energy_drude_N'+str(N)+'_K'+str(K)+'_task_'+str(task_id)+'.npz', \
                    reorg_energy_values=reorg_energy_values, mean=mean, F2=F2, coh=coh)
# np.savez('/home/zcqsc45/Scratch/HEOM_F2_reorg_energy_drude_N'+str(N)+'_K'+str(K)+'_task_'+str(task_id)+'.npz', \
#                     reorg_energy_values=reorg_energy_values, F2_heom=F2_heom, F2_pert=F2_pert, coh_heom=coh_heom)

