'''
Created on 19 Jan 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
import scipy.sparse as sp
from HEOM_counting_statistics.DQD_HEOM_model import DQDHEOMModel
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from counting_statistics.fcs_solver import FCSSolver as FCD
from counting_statistics.sparse.fcs_solver import FCSSolver as FCS
from quant_mech.OBOscillator import OBOscillator

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature]
reorg_energy = 0.00147
cutoff = 5. # meV
K = 2

# dense_model = DQDHEOMModel(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], drude_reorg_energy=reorg_energy, drude_cutoff=cutoff, \
#                      num_matsubara_freqs=K, temperature_correction=True, sites_to_couple=np.array([0,1,1]))
# dense_solver = FCD(dense_model.heom_matrix(), dense_model.jump_matrix(), dense_model.dv_pops)
# ss = dense_solver.stationary_state(dense_solver.L, dense_solver.pops)
# Q = dense_solver.Q(dense_solver.L, ss, dense_solver.pops)
# R = dense_solver.pseudoinverse(dense_solver.L, 0, Q)
# R_on_Jss = np.dot(R, np.dot(dense_solver.jump_op, ss))
# np.savez('../../data/dense_R_Q.npz', R=R, Q=Q, R_on_Jss=R_on_Jss, ss=ss, \
#                         L=dense_solver.L, J=dense_solver.jump_op)
# #print dense_model.heom_solver.diag_coeffs
# print dense_model.heom_solver.phix_coeffs
# print dense_model.heom_solver.thetax_coeffs
# print dense_model.heom_solver.thetao_coeffs
# print dense_model.heom_solver.matsubara_freqs

def environment(beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]
sparse_model = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], environment=environment(beta[0], K), K=K, tc=True)
sparse_solver = FCS(sparse_model.heom_matrix(), sparse_model.jump_matrix(), sparse_model.dv_pops)
ss = sparse_solver.stationary_state(sparse_solver.L, sparse_solver.pops)
Q = sparse_solver.Q(sparse_solver.L, ss, sparse_solver.pops)
R_on_Jss = sparse_solver.pseudoinverse(sparse_solver.L, Q, sparse_solver.jump_op.dot(ss))
np.savez('../../data/sparse_R_Q.npz', Q=Q.todense(), R_on_Jss=R_on_Jss, ss=ss, \
                    L=sparse_solver.L.todense(), J=sparse_solver.jump_op.todense())
print sparse_model.heom_solver.diag_coeffs
print sparse_model.heom_solver.phix_coeffs
print sparse_model.heom_solver.thetax_coeffs
print sparse_model.heom_solver.thetao_coeffs
print sparse_model.heom_solver.matsubara_freqs
print sparse_model.heom_solver.tc_terms



