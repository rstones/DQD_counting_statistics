'''
Created on 24 Mar 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
from counting_statistics.sparse.fcs_solver import FCSSolver
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.OBOscillator import OBOscillator
import quant_mech.time_utils as tu
from HEOM_counting_statistics.dissipative_DQD_model import DissipativeDQDModel
from counting_statistics.fcs_solver import FCSSolver as DenseFCSSolver
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0.2
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature][1:2]
reorg_energy = 0.00147 # meV
cutoff = 5. # meV
K = 4

def environment(beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]
    
model_heom = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], environment=environment(beta[0], K), \
                                K=K, tc=True, trunc_level=5)

bias_values = np.linspace(-1, 1, 20)
F2_values = np.zeros(bias_values.size)

for i,E in enumerate(bias_values):
    print E
    solver = FCSSolver(model_heom.heom_matrix(), model_heom.jump_matrix(), model_heom.dv_pops)
    F2_values[i] = solver.second_order_fano_factor()
    
import matplotlib.pyplot as plt
plt.plot(bias_values, F2_values)
plt.show()

