'''
Created on 15 Aug 2017

@author: richard
'''
import numpy as np
from counting_statistics.fcs_solver import FCSSolver
import quant_mech.utils as utils

import prettyplotlib as ppl
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
        

# plt.subplot(121)
# plt.plot(bias_values, current, linewidth=3)
# plt.xlabel(r'energy bias $\epsilon$')
# plt.ylabel(r'current')
# plt.subplot(122)
# plt.plot(bias_values, F2, linewidth=3)
# plt.xlabel(r'energy bias $\epsilon$')
# plt.ylabel(r'F$^{(2)}$(0)')
# plt.subplot(133)
# plt.plot(bias_values, np.abs(coherence), linewidth=3)
# plt.xlabel(r'energy bias $\epsilon$')
# plt.ylabel(r'|coherence|')
# plt.show()

fig,ax = plt.subplots(1, figsize=(4,4))
ax.axhline(1, ls='--', color='grey')
ax.text(0.00008, 1.27, 'a.', fontsize=12)
for i in range(T_c_values.size):
    ppl.semilogx(Gamma_R_values, F2[i], linewidth=3, label=r'$T_c = $' + str(T_c_values[i]))
ax.set_xlabel(r'$\Gamma_R$')
ax.set_ylabel('Fano factor')
#ax.set_ylim(0.3, 1.35)
ppl.legend(fontsize=12).draggable()
plt.tight_layout()
plt.show()
 
fig,ax = plt.subplots(1, figsize=(4,4))
ax.text(0.00008, 0.355, 'b.', fontsize=12)
for i in range(T_c_values.size):
    ppl.semilogx(Gamma_R_values, np.abs(coherence[i]), linewidth=3, label=r'$T_c = $' + str(T_c_values[i]))
ax.set_xlabel(r'$\Gamma_R$')
ax.set_ylabel(r'coherence $|\rho_{LR}|$')
#ax.set_ylim(0, 0.4)
ppl.legend(fontsize=12).draggable()
plt.tight_layout()
plt.show()

# print site_exciton_transform[2,50]
# 
# fig = plt.figure(figsize=(8,7))
# plt.text(0.00015, 0.372, '(b)', fontsize=22)
# for i in range(T_c_values.size):
#     plt.semilogx(Gamma_R_values, np.abs(exciton_steady_states[i,:,1,1]), linewidth=3, label=r'$T_c = $' + str(T_c_values[i]))
# plt.xlabel(r'$\Gamma_R$')
# plt.ylabel(r'coherence $|\rho_{LR}|$')
# plt.ylim(0, 0.4)
# plt.legend(fontsize=14).draggable()
# plt.show()



