'''
Created on 8 Sep 2017

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

Gamma_R_values = np.logspace(-4, 4, 100)
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
 
 
transform = site_exciton_transform[2,0]
print transform

print exciton_steady_states.shape
 
# fig,ax = plt.subplots(1, figsize=(4,4))
# ax.text(0.00008, 1.02, 'd.', fontsize=12)
# ppl.semilogx(Gamma_R_values, np.abs(exciton_steady_states[2,:,1,1]), linewidth=3, label=r'$\rho_{++}$')
# ppl.semilogx(Gamma_R_values, np.abs(exciton_steady_states[2,:,0,0]), linewidth=3, label=r'$\rho_{--}$')
# ppl.semilogx(Gamma_R_values, np.abs(exciton_steady_states[2,:,2,2]), linewidth=3, label=r'$\rho_{00}$')
# ax.set_xlabel(r'$\Gamma_R$')
# ax.set_ylabel(r'population')
# #ax.set_ylim(0, 1)
# ppl.legend(fontsize=12).draggable()
# plt.tight_layout()
# plt.show()

fig,ax = plt.subplots(1, figsize=(4,4))
      
ax.text(0.00008, 1.06, 'c.', fontsize=12)
ppl.semilogx(ax, Gamma_R_values, np.abs(site_steady_states[2,:,1,1]), linewidth=3, label=r'$\rho_{LL}$')
ppl.semilogx(ax, Gamma_R_values, np.abs(site_steady_states[2,:,2,2]), linewidth=3, label=r'$\rho_{RR}$')
ppl.semilogx(ax, Gamma_R_values, np.abs(site_steady_states[2,:,0,0]), linewidth=3, label=r'$\rho_{00}$')
ax.set_xlabel(r'$\Gamma_R$')
ax.set_ylabel(r'population')
#ax.set_ylim(0, 1)
ppl.legend(fontsize=12).draggable()
plt.tight_layout()
plt.show()
#fig.savefig('../../plots/prettyplotlibtest.png')





