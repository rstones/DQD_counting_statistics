'''
Created on 21 May 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
from counting_statistics.sparse.fcs_solver import FCSSolver
import quant_mech.utils as utils
import quant_mech.open_systems as os
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

Gamma_L = 1.
Gamma_R = 0.025
bias = 0
T_c = 1.

beta_values = [0.8, 0.4, 0.1]

mode_freq = 10.
hr_factor = 0.5
damping = 0.5
mode_coupling = np.sqrt(mode_freq) * hr_factor # DOUBLE CHECK THIS!!!

bias_values = np.linspace(-15, 15, 100)
mean = np.zeros((len(beta_values), bias_values.size))
F2 = np.zeros((len(beta_values), bias_values.size))

jump_rates = np.array([Gamma_L, Gamma_R])

def electronic_hamiltonian(bias):
    return np.array([[0, 0, 0],
                 [0, bias/2., T_c],
                 [0, T_c, -bias/2.]])
I_el = np.eye(3)

mode_basis_size = 4
H_mode = utils.vibrational_hamiltonian(mode_freq, mode_basis_size)
I_mode = np.eye(mode_basis_size)
up_mode = utils.raising_operator(mode_basis_size)
down_mode = utils.lowering_operator(mode_basis_size)

def N(omega, beta):
    return 1. / (np.exp(beta*omega) - 1.)

''' B is beta, the inverse temperature'''
def mode_damping_rates(B):
    return np.array([damping*N(mode_freq, B), 
                                   damping*(N(mode_freq, B)+1.)])
    
mode_damping_operators = np.array([np.kron(I_el, up_mode), 
                                   np.kron(I_el, down_mode)])

#H = np.kron(H_el, I_mode) + np.kron(I_el, H_mode) + mode_coupling*np.kron(np.array([[0,0,0],[0,1.,0],[0,0,1.]]), up_mode + down_mode)
lead_operators = np.array([np.kron(np.array([[0, 0, 0],
                                              [1., 0, 0],
                                              [0, 0, 0]]), I_mode), np.kron(np.array([[0, 0, 1.],
                                                                                     [0, 0, 0],
                                                                                     [0, 0, 0]]), I_mode)])

jump_operator = Gamma_R * np.kron(lead_operators[1], lead_operators[1])
# np.set_printoptions(threshold=100000, linewidth=1000)
# print np.nonzero(jump_operator)
# print jump_operator
pops = np.eye(3*mode_basis_size).flatten()

for i,B in enumerate(beta_values):
    damping_rates = mode_damping_rates(B)
    for j,E in enumerate(bias_values):
        print E
        H_el = electronic_hamiltonian(E)
        H = np.kron(H_el, I_mode) + np.kron(I_el, H_mode) + \
                    mode_coupling*np.kron(np.array([[0,0,0],[0,1.,0],[0,0,-1.]]), up_mode + down_mode)
        L = os.super_operator(H, [(lead_operators[0], Gamma_L), (lead_operators[1], Gamma_R), \
                                  (mode_damping_operators[0], damping_rates[0]), (mode_damping_operators[1], damping_rates[1])])
        try:
            solver = FCSSolver(sp.csr_matrix(L), sp.csr_matrix(jump_operator), pops)
            mean[i,j] = solver.mean()
            F2[i,j] = solver.second_order_fano_factor()
        except ArpackNoConvergence:
            print ("No convergence!")
        
import matplotlib.pyplot as plt

plt.subplot(121)
for i in range(len(beta_values)):
    plt.plot(bias_values, mean[i], label=beta_values[i])
plt.legend().draggable()

plt.subplot(122)
for i in range(len(beta_values)):
    plt.plot(bias_values, F2[i], label=beta_values[i])
plt.legend().draggable()

plt.show()

# construct system and mode operators (hamiltonians, system jump operators, mode damping, identity operators)
# tensor product them to get operators in composite Hilbert space
# get jump operator in composite Hilbert space
# construct sparse.FCSSolver object
# getting mode F2 isnt that straight forward using my current counting_statistics package. See previous work iv done
# on it to see how to get jump operator
# get mean and F2 vs bias for several temperatures
# (need to include electronic pure dephasing through Lindblad operators?)




