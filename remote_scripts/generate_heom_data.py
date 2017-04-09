import numpy as np
import scipy.constants as constants
from counting_statistics.sparse.fcs_solver import FCSSolver
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.OBOscillator import OBOscillator
import quant_mech.time_utils as tu
from HEOM_counting_statistics.dissipative_DQD_model import DissipativeDQDModel
from counting_statistics.fcs_solver import FCSSolver as DenseFCSSolver
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
import quant_mech.hierarchy_solver_numba_functions as numba_funcs

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
K = 11
N = 6

def environment(reorg_energy, beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]

reorg_energy_values = np.logspace(0, 4, 5)

higher_coupling_elements_list = []
lower_coupling_elements_list = []

for i,E in enumerate(reorg_energy_values):
    print E
    model_heom = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta, environment=environment(E, beta, K), \
                                    K=K, tc=True, trunc_level=N)
    num_dms = model_heom.heom_solver.number_density_matrices()
    num_indices = model_heom.heom_solver.num_aux_dm_indices
    dm_per_tier = model_heom.heom_solver.dm_per_tier()
    n_vectors, higher_coupling_elements, higher_coupling_row_indices, higher_coupling_column_indices, \
        lower_coupling_elements, lower_coupling_row_indices, lower_coupling_column_indices = numba_funcs.generate_hierarchy_and_tier_couplings(num_dms, num_indices, N, dm_per_tier)
    higher_coupling_elements_list.append(higher_coupling_elements)
    lower_coupling_elements_list.append(lower_coupling_elements)   
        
# n_vectors, row and col indices should be the same for each heom matrix? (where only reorg energy is being changed)
# so only need to save coupling elements for each reorg energy

np.savez('../data/F2_reorg_energy_heom_data_N'+str(N)+'_K'+str(K)+'.npz', reorg_energy_values=reorg_energy_values, N=N, K=K, n_vectors=n_vectors, \
                    higher_coupling_elements=np.array(higher_coupling_elements_list), higher_coupling_row_indices=higher_coupling_row_indices, \
                    higher_coupling_column_indices=higher_coupling_column_indices, lower_coupling_elements=np.array(lower_coupling_elements_list), \
                    lower_coupling_row_indices=lower_coupling_row_indices, lower_coupling_column_indices=lower_coupling_column_indices)
