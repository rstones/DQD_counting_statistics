'''
Created on 21 Mar 2016

@author: rstones
'''
import numpy as np
from counting_statistics.fcs_solver import FCSSolver
import matplotlib.pyplot as plt
import matplotlib

font = {'size':20}
matplotlib.rc('font', **font)

bias_values = np.linspace(-10, 10, 200)
lindblad_ops = [np.array([[0,0,0],[1,0,0],[0,0,0]]), np.array([[0,0,1],[0,0,0],[0,0,0]])]
Gamma_L = 1.
Gamma_R = 1.e-4
lindblad_rates = [Gamma_L, Gamma_R]
jump_idx = np.array([0,1])

bias_values = np.linspace(-10, 10, 200)
T_c = 3.
def dqd_hamiltonian(bias, T_c):
    return np.array([[0,0,0],[0,bias/2.,T_c],[0,T_c,-bias/2.]])

solver = FCSSolver.from_hilbert_space(dqd_hamiltonian(0, T_c), lindblad_ops, lindblad_rates, jump_idx, reduce_dim=True)

current = np.zeros(bias_values.size)
F2 = np.zeros(bias_values.size)
coherence = np.zeros(bias_values.size, dtype='complex')

for i,v in enumerate(bias_values):
    solver.H = dqd_hamiltonian(2.*v, T_c)
    current[i] = solver.mean()
    F2[i] = solver.second_order_fano_factor(0)
    coherence[i] = solver.ss[2]

plt.subplot(121)
plt.plot(bias_values, current, linewidth=3)
plt.xlabel(r'energy bias $\epsilon$')
plt.ylabel(r'current')
plt.subplot(122)
plt.plot(bias_values, F2, linewidth=3)
plt.xlabel(r'energy bias $\epsilon$')
plt.ylabel(r'F$^{(2)}$(0)')
# plt.subplot(133)
# plt.plot(bias_values, np.abs(coherence), linewidth=3)
# plt.xlabel(r'energy bias $\epsilon$')
# plt.ylabel(r'|coherence|')
plt.show()
