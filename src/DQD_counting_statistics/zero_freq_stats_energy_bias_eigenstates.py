'''
Created on 9 Sep 2017

@author: richard
'''
import numpy as np
from counting_statistics.fcs_solver import FCSSolver
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

bias_values = np.linspace(-10, 10, 100)
T_c_values = np.array([0.5, 1., 2.])
lindblad_ops = [np.array([[0,0,0],[1,0,0],[0,0,0]]), np.array([[0,0,1],[0,0,0],[0,0,0]])]
Gamma_L = 1.
Gamma_R = 0.025
lindblad_rates = [Gamma_L, Gamma_R]
jump_idx = np.array([0,1])

#T_c = 1.
def dqd_hamiltonian(bias, T_c):
    return np.array([[0,0,0],[0,bias/2.,T_c],[0,T_c,-bias/2.]])

solver = FCSSolver.from_hilbert_space(dqd_hamiltonian(0, 1.), lindblad_ops, lindblad_rates, jump_idx, reduce_dim=True)

current = np.zeros((T_c_values.size, bias_values.size))
F2 = np.zeros((T_c_values.size, bias_values.size))
coherence = np.zeros((T_c_values.size, bias_values.size), dtype='complex128')

site_steady_states = np.zeros((T_c_values.size, bias_values.size, 3, 3))
site_exciton_transform = np.zeros((T_c_values.size, bias_values.size, 3, 3))
exciton_steady_states = np.zeros((T_c_values.size, bias_values.size, 3, 3))

for j,T_c in enumerate(T_c_values):
    for i,v in enumerate(bias_values):
        solver.H = dqd_hamiltonian(v, T_c)
        current[j,i] = solver.mean()
        F2[j,i] = solver.second_order_fano_factor(0)
        coherence[j,i] = solver.ss[2]

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

fig = plt.figure(figsize=(8,7))
plt.axhline(1, ls='--', color='grey')
plt.text(-9.7, 1.222, '(a)', fontsize=22)
for i in range(T_c_values.size):
    plt.plot(bias_values, F2[i], linewidth=3, label=r'$T_c = $' + str(T_c_values[i]))
plt.xlabel(r'bias $\epsilon$')
plt.ylabel('Fano factor')
plt.ylim(0.85, 1.25)
plt.legend(fontsize=14).draggable()
plt.show()

fig = plt.figure(figsize=(8,7))
plt.text(-9.7, 0.372, '(b)', fontsize=22)
for i in range(T_c_values.size):
    plt.plot(bias_values, np.abs(coherence[i]), linewidth=3, label=r'$T_c = $' + str(T_c_values[i]))
plt.xlabel(r'bias $\epsilon$')
plt.ylabel(r'coherence $|\rho_{LR}|$')
#plt.ylim(0.85, 1.25)
plt.legend(fontsize=14).draggable()
plt.show()



