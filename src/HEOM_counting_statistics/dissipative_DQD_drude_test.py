'''
Created on 5 Jan 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
from counting_statistics.fcs_solver import FCSSolver
from HEOM_counting_statistics.dissipative_DQD_model import DissipativeDQDModel

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature]
reorg_energy = 0.00147
cutoff = 5. # meV

def drude_spectral_density(reorg_energy, cutoff):
    def J(delta):
        return (2. * reorg_energy * cutoff * delta) / (delta**2 + cutoff**2)
    return J

bias_values = np.linspace(-1., 1., 500)
F2_values = np.zeros((len(beta)+1, bias_values.size))

model = DissipativeDQDModel(Gamma_L, Gamma_R, 0, T_c, drude_spectral_density(reorg_energy, cutoff), 1.)
solver = FCSSolver(model.liouvillian(), model.jump_matrix(), np.array([1,1,1,0,0]))

for j,B in enumerate(beta):
    model.beta = B
    for i,E in enumerate(bias_values):
        model.bias = E
        solver.L = model.liouvillian()
        F2_values[j+1,i] = solver.second_order_fano_factor(0)
    
model.spectral_density = drude_spectral_density(0, cutoff)
for i,E in enumerate(bias_values):
    model.bias = E
    solver.L = model.liouvillian()
    F2_values[0,i] = solver.second_order_fano_factor(0)
    
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

plt.plot(bias_values, F2_values[0], linewidth=3, ls='--', color='k', label='no phonons')
for i,B in enumerate(beta):
    plt.plot(bias_values, F2_values[i+1], linewidth=3, label='T = ' + str(temperature[i]) + 'K')
plt.axhline(1., ls='--', color='grey', linewidth=2)
plt.xlim(-1.05, 1.05)
plt.ylim(0.72, 1.25)
plt.xlabel(r'bias $\epsilon$ (meV)')
plt.ylabel(r'Fano factor')
plt.legend(fontsize=14).draggable()
plt.show()
    