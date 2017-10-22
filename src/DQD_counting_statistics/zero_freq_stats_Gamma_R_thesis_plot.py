'''
Created on 9 Sep 2017

@author: richard
'''
import prettyplotlib as ppl

import numpy as np
from counting_statistics.fcs_solver import FCSSolver
import quant_mech.utils as utils


import matplotlib.pyplot as plt
import matplotlib as mpl
from prettyplotlib import brewer2mpl

font = {'size':12}
mpl.rc('font', **font)

Gamma_R_values = np.logspace(-4, 4, 1000)
T_c_values = np.array([0.5, 1., 2.])
lindblad_ops = [np.array([[0,0,0],[1,0,0],[0,0,0]]), np.array([[0,0,1],[0,0,0],[0,0,0]])]
Gamma_L = 1.
Gamma_R = 0.025
lindblad_rates = [Gamma_L, Gamma_R]
jump_idx = np.array([0,1])

bias = 4.
T_c = 1.
def dqd_hamiltonian(bias, T_c):
    return np.array([[0,0,0],[0,bias/2.,T_c],[0,T_c,-bias/2.]])

#solver = FCSSolver.from_hilbert_space(dqd_hamiltonian(bias, T_c), lindblad_ops, lindblad_rates, jump_idx, reduce_dim=True)

current = np.zeros((T_c_values.size, Gamma_R_values.size))
F2 = np.zeros((T_c_values.size, Gamma_R_values.size))
coherence = np.zeros((T_c_values.size, Gamma_R_values.size), dtype='complex128')

site_steady_states = np.zeros((T_c_values.size, Gamma_R_values.size, 3, 3))
site_exciton_transform = np.zeros((T_c_values.size, Gamma_R_values.size, 3, 3))
exciton_steady_states = np.zeros((T_c_values.size, Gamma_R_values.size, 3, 3))

for j,T_c in enumerate(T_c_values):
    for i,Gamma_R in enumerate(Gamma_R_values):
        solver = FCSSolver.from_hilbert_space(dqd_hamiltonian(bias, T_c), lindblad_ops, [Gamma_L, Gamma_R], jump_idx, reduce_dim=False)
        #solver.H = dqd_hamiltonian(bias, T_c)
        current[j,i] = solver.mean()
        F2[j,i] = solver.second_order_fano_factor(0)
        ss = solver.ss
        ss.shape = 3,3
        coherence[j,i] = solver.ss[1,2]
        
        site_steady_states[j,i] = solver.ss
        transform = np.linalg.eig(solver.H)[1] # utils.sorted_eig(solver.H)[1] #
        site_exciton_transform[j,i] = transform
        exciton_steady_states[j,i] = np.dot(transform.T, np.dot(ss, transform))
        
fig,ax = ppl.subplots(ncols=2, nrows=2, figsize=(8,7))
lw = 2

# Fano factor
plt.sca(ax[0,0])
ax[0,0].axhline(1, ls='--', color='grey')
ax[0,0].text(0.00008, 1.27, 'a.')
for i in range(T_c_values.size):
    ppl.semilogx(Gamma_R_values, F2[i], linewidth=lw, label=r'$T_c = $' + str(T_c_values[i]), show_ticks=True)
ax[0,0].set_xlabel(r'$\Gamma_R / \Gamma_L$')
ax[0,0].set_ylabel('Fano factor')
#ax.set_ylim(0.3, 1.35)
ppl.legend(fontsize=10).draggable()

# coherence
plt.sca(ax[0,1])
ax[0,1].text(0.00008, 0.355, 'b.')
for i in range(T_c_values.size):
    ppl.semilogx(Gamma_R_values, np.abs(coherence[i]), linewidth=lw, label=r'$T_c = $' + str(T_c_values[i]), show_ticks=True)
ax[0,1].set_xlabel(r'$\Gamma_R / \Gamma_L$')
ax[0,1].set_ylabel(r'coherence $|\rho_{LR}|$')
#ax.set_ylim(0, 0.4)
ppl.legend(fontsize=10).draggable()

# site populations
plt.sca(ax[1,0])
ax[1,0].text(0.00008, 1.03, 'c.')
ppl.semilogx(Gamma_R_values, np.abs(site_steady_states[2,:,1,1]), linewidth=lw, label=r'$\rho_{LL}$', show_ticks=True)
ppl.semilogx(Gamma_R_values, np.abs(site_steady_states[2,:,2,2]), linewidth=lw, label=r'$\rho_{RR}$', show_ticks=True)
ppl.semilogx(Gamma_R_values, np.abs(site_steady_states[2,:,0,0]), linewidth=lw, label=r'$\rho_{00}$', show_ticks=True)
ax[1,0].set_xlabel(r'$\Gamma_R / \Gamma_L$')
ax[1,0].set_ylabel(r'population')
ax[1,0].set_ylim(-0.02, 1.01)
ppl.legend(fontsize=10).draggable()

# exciton populations
plt.sca(ax[1,1])
ax[1,1].text(0.00008, 1.03, 'd.')
ppl.semilogx(Gamma_R_values, np.abs(exciton_steady_states[2,:,1,1]), linewidth=lw, label=r'$\rho_{++}$', show_ticks=True)
ppl.semilogx(Gamma_R_values, np.abs(exciton_steady_states[2,:,0,0]), linewidth=lw, label=r'$\rho_{--}$', show_ticks=True)
ppl.semilogx(Gamma_R_values, np.abs(exciton_steady_states[2,:,2,2]), linewidth=lw, label=r'$\rho_{00}$', show_ticks=True)
ax[1,1].set_xlabel(r'$\Gamma_R / \Gamma_L$')
ax[1,1].set_ylabel(r'population')
ax[1,1].set_ylim(-0.02, 1.01)
ppl.legend(fontsize=10).draggable()

plt.tight_layout()
plt.show()

