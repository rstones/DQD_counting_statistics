'''
Created on 1 Feb 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
import quant_mech.utils as utils
from counting_statistics.sparse.fcs_solver import FCSSolver
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.UBOscillator import UBOscillator
from quant_mech.OBOscillator import OBOscillator

# Gamma_L = 0.001 # eV
# Gamma_R = 0.001 # eV
# bias = 0
# T_c = 0.05 # eV
# 
# drude_reorg_energy = 0.1
# drude_cutoff = 0.1
# mode_freq = 0.2 # eV
# mode_coupling = 0.18 # eV
# mode_S = (mode_coupling / mode_freq)**2
# mode_damping = 0.001 # eV
# temperature = 10. # Kelvin
# k_B = constants.physical_constants["Boltzmann constant in eV/K"][0]
# beta = 1. / (k_B * temperature) # eV^-1

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature]
drude_reorg_energy = 0.00147
drude_cutoff = 5. # meV
mode_freq = 0.5
mode_S = 0.0147 / mode_freq
mode_damping = 0.001

beta = beta[2]

print 1. / beta

K = 0
tc = True

# environment = [(),
#                (OBOscillator(drude_reorg_energy, drude_cutoff, beta, K=K),),
#                (OBOscillator(drude_reorg_energy, drude_cutoff, beta, K=K),)]
environment = [(),
               (UBOscillator(mode_freq, mode_S, mode_damping, beta, K=K),),
               (UBOscillator(mode_freq, mode_S, mode_damping, beta, K=K),)]

model = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, environment, beta, K, tc, trunc_level=7)

bias_values = np.linspace(-1, 1, 50)
current_values = np.zeros(bias_values.size)
F2_values = np.zeros(bias_values.size)
steady_states = np.zeros((bias_values.size, model.system_dimension, model.system_dimension), dtype='complex128')
exciton_splitting = np.zeros(bias_values.size)

for i,E in enumerate(bias_values):
    print E
    model.bias = E
    try:
        solver = FCSSolver(model.heom_matrix(), model.jump_matrix(), model.dv_pops)
        current_values[i] = solver.mean()
        F2_values[i] = solver.second_order_fano_factor()
        steady_states[i] = model.heom_solver.extract_system_density_matrix(solver.ss)
    except:
        print "Error at bias " + str(E)
    exciton_splitting[i] = np.sqrt(model.bias**2 + 4.*model.T_c**2)
    
import matplotlib.pyplot as plt

plt.subplot(411)
plt.plot(bias_values, current_values)

plt.subplot(412)
plt.plot(bias_values, F2_values)
plt.axvline(bias_values[np.argmax(F2_values[:F2_values.size/2])], color='r')
plt.axvline(bias_values[F2_values.size/2 + np.argmax(F2_values[F2_values.size/2:])], color='r')

plt.subplot(413)
plt.plot(bias_values, np.abs(steady_states[:,1,2]))
plt.axvline(bias_values[np.argmax(np.abs(steady_states[:,1,2])[:steady_states.shape[0]/2])], color='r')
plt.axvline(bias_values[steady_states.shape[0]/2 + np.argmax(np.abs(steady_states[:,1,2])[steady_states.shape[0]/2:])], color='r')

plt.subplot(414)
plt.plot(bias_values, exciton_splitting)

plt.show()

    