'''
Created on 28 Nov 2016

@author: richard
'''
import numpy as np
import scipy.constants as constants
from HEOM_counting_statistics.DQD_HEOM_model import DQDHEOMModel
from counting_statistics.fcs_solver import FCSSolver

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature]
reorg_energy = 0.00147
cutoff = 5. # meV

model = DQDHEOMModel(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], drude_reorg_energy=reorg_energy, drude_cutoff=cutoff)
bias_values = np.linspace(-1, 1, 100)
F2 = np.zeros((len(beta)+1, bias_values.size))

for j,B in enumerate(beta):
    print 'for beta = ' + str(B)
    model.beta = B
    for i,E in enumerate(bias_values):
        print E
        model.bias = E
        solver = FCSSolver(model.heom_matrix(), model.jump_matrix(), model.dv_pops)
        F2[j+1,i] = solver.second_order_fano_factor(0)
        
model.drude_reorg_energy = 1.e-9
print 'for no phonons'
for i,E in enumerate(bias_values):
    print E
    model.bias = E
    solver = FCSSolver(model.heom_matrix(), model.jump_matrix(), model.dv_pops)
    F2[0,i] = solver.second_order_fano_factor(0)
    
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

plt.plot(bias_values, F2[0], linewidth=3, ls='--', color='k', label='no phonons')
for i,B in enumerate(beta):
    plt.plot(bias_values, F2[i+1], linewidth=3, label='T = ' + str(temperature[i]) + 'K')
plt.axhline(1, ls='--', color='grey')
plt.xlim(-1.05, 1.05)
plt.ylim(0.72, 1.25)
plt.xlabel(r'bias $\epsilon$ (meV)')
plt.ylabel(r'Fano factor')
plt.legend(fontsize=14).draggable()
plt.show()
