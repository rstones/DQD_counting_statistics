'''
Created on 5 Jan 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
from counting_statistics.fcs_solver import FCSSolver
from HEOM_counting_statistics.dissipative_DQD_model import DissipativeDQDModel

Gamma_L = 1. #0.1 # meV
Gamma_R = 0.025 #2.5e-3 # meV
bias = 0
T_c = 1. #0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = 10 * constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature]
beta = [0.8, 0.4, 0.1]
reorg_energy = 0.015 #0.00147
cutoff = 50. #5. # meV

def drude_spectral_density(reorg_energy, cutoff):
    def J(delta):
        return (2. * reorg_energy * cutoff * delta) / (delta**2 + cutoff**2)
    return J

bias_values = np.linspace(-1., 1., 500) * 10
F2_values = np.zeros((len(beta)+1, bias_values.size))
coherence = np.zeros((len(beta)+1, bias_values.size))

site_steady_states = np.zeros((len(beta)+1, bias_values.size, 5))
# site_exciton_transform = np.zeros((len(beta)+1, bias_values.size, 3, 3))
# exciton_steady_states = np.zeros((len(beta)+1, bias_values.size, 3, 3))

model = DissipativeDQDModel(Gamma_L, Gamma_R, 0, T_c, drude_spectral_density(reorg_energy, cutoff), 1.)
solver = FCSSolver(model.liouvillian(), model.jump_matrix(), np.array([1,1,1,0,0]))

for j,B in enumerate(beta):
    model.beta = B
    for i,E in enumerate(bias_values):
        model.bias = E
        solver.L = model.liouvillian()
        F2_values[j+1,i] = solver.second_order_fano_factor(0)
        coherence[j+1,i] = solver.ss[3] + solver.ss[4]
        site_steady_states[j+1,i] = solver.ss
    
model.spectral_density = drude_spectral_density(0, cutoff)
for i,E in enumerate(bias_values):
    model.bias = E
    solver.L = model.liouvillian()
    F2_values[0,i] = solver.second_order_fano_factor(0)
    coherence[0,i] = solver.ss[3] + solver.ss[4]
    site_steady_states[0,i] = solver.ss
    
np.savez('../../data/DQD_dissipative_F2_bias_brandes_liouvillian.npz', \
         bias_values=bias_values, beta_values=beta, F2=F2_values, coherence=coherence, \
         site_steady_states=site_steady_states)

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import matplotlib as mpl
from prettyplotlib import brewer2mpl

font = {'size':12}
mpl.rc('font', **font)

fig,ax = ppl.subplots(1, 2, figsize=(8,3.5))

plt.sca(ax[0])
ppl.plot(bias_values, F2_values[0], linewidth=3, ls='--', color='k', label='no phonons')
for i,B in enumerate(beta):
    ppl.plot(bias_values, F2_values[i+1], linewidth=3, label=r'$\beta = ' + str(beta[i]) + '$', show_ticks=True)
ax[0].axhline(1., ls='--', color='grey', linewidth=1)
#plt.xlim(-10.2, 10.2)
#plt.ylim(0.82, 1.25)
ax[0].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[0].set_ylabel(r'Fano factor')
ax[0].text(-10, 1.25, 'a.', fontsize=12)
ppl.legend(fontsize=12).draggable()

plt.sca(ax[1])
ppl.plot(bias_values, np.abs(coherence[0]), linewidth=3, ls='--', color='k', label='no phonons')
for i,B in enumerate(beta):
    ppl.plot(bias_values, np.abs(coherence[i+1]), linewidth=3, label=r'$\beta = ' + str(beta[i]) + '$', show_ticks=True)
ax[1].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[1].set_ylabel(r'coherence $|\rho_{LR}|$')

plt.tight_layout()
plt.show()
    
