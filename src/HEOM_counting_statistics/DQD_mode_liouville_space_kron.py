'''
Created on 6 Jun 2017

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

beta_values = [0.8, 0.4, 0.1][2:]

mode_damping_values = [0.5, 2., 5.][1:2]

modeL_freq = 10.
modeL_hr_factor = 0.5
modeL_damping = 2. #0.5
modeL_coupling = np.sqrt(modeL_freq) * modeL_hr_factor # DOUBLE CHECK THIS!!!

modeR_freq = 10.
modeR_hr_factor = 0.5
modeR_damping = 2. #0.5
modeR_coupling = np.sqrt(modeR_freq) * modeR_hr_factor # DOUBLE CHECK THIS!!!

bias_values = np.linspace(-20, 20, 50)
mean = np.zeros(bias_values.size)
F2 = np.zeros(bias_values.size)

jump_rates = np.array([Gamma_L, Gamma_R])

def electronic_hamiltonian(E, J):
    return np.array([[0, 0, 0],
                 [0, E/2., J],
                 [0, J, -E/2.]])
I_el = np.eye(3)
I_el_super = np.eye(9)

mode_basis_size = 4
H_modeL = utils.vibrational_hamiltonian(modeL_freq, mode_basis_size)
H_modeR = utils.vibrational_hamiltonian(modeR_freq, mode_basis_size)
I_mode = np.eye(mode_basis_size)
I_mode_super = np.eye(mode_basis_size**2)
up_mode = utils.raising_operator(mode_basis_size)
down_mode = utils.lowering_operator(mode_basis_size)

def N(omega, beta):
    return 1. / (np.exp(beta*omega) - 1.)

''' B is beta, the inverse temperature'''
def modeL_damping_rates(rate, B):
    return np.array([rate*N(modeL_freq, B), 
                                   rate*(N(modeL_freq, B)+1.)])
    
def modeR_damping_rates(rate, B):
    return np.array([rate*N(modeR_freq, B), 
                                   rate*(N(modeR_freq, B)+1.)])
    
lead_operators = np.array([np.array([[0, 0, 0],
                                      [1., 0, 0],
                                      [0, 0, 0]]), \
                           np.array([[0, 0, 1.],
                                     [0, 0, 0],
                                     [0, 0, 0]])])
    
# function for electronic Liouvillian
# basis: { 00, 0L, 0R, L0, LL, LR, R0, RL, RR }
def electronic_liouvillian(E, J):
    # need to convert Brandes Liouvillian into rho_LR, rho_RL basis
    # construct no phonon Liouvillian with os.super_operator then add in dephasing 'by hand'
    L = os.super_operator(electronic_hamiltonian(E, J), [(lead_operators[0], Gamma_L), (lead_operators[1], Gamma_R)])
    
    '''Dephasing rates on coherences'''
    L[5,4] += 0 # gamma_plus
    L[5,8] -= 0 # gamma_minus
    L[5,5] -= 0 # gamma
    
    L[7,4] += 0 # gamma_plus
    L[7,8] -= 0 # gamma_minus
    L[7,7] -= 0 # gamma
    
    return L

def electronic_interaction_liouvillian_L():
    return modeL_coupling * os.super_operator(np.array([[0,0,0],
                                                        [0,1.,0],
                                                        [0,0,0]]), [])
    
def electronic_interaction_liouvillian_R():
    return modeR_coupling * os.super_operator(np.array([[0,0,0],
                                                        [0,0,0],
                                                        [0,0,1.]]), [])

# function for vibrational Liouvillian
def vibrational_liouvillian_L(damping_rate, B):
    rates = modeL_damping_rates(damping_rate, B)
    return os.super_operator(H_modeL, [(up_mode, rates[0]), (down_mode, rates[1])])

def vibrational_liouvillian_R(damping_rate, B):
    rates = modeR_damping_rates(damping_rate, B)
    return os.super_operator(H_modeR, [(up_mode, rates[0]), (down_mode, rates[1])])

def vibrational_interaction_liouvillian():
    return os.super_operator(up_mode+down_mode, [])

# function for interaction Liouvillian
def interaction_liouvillian_L():
    return np.kron(electronic_interaction_liouvillian_L(), np.kron(vibrational_interaction_liouvillian(), I_mode_super))

def interaction_liouvillian_R():
    return np.kron(electronic_interaction_liouvillian_R(), np.kron(I_mode_super, vibrational_interaction_liouvillian()))

def jump_operator():
    J_op = np.zeros((9,9))
    J_op[0,8] = 1.
    return Gamma_R * np.kron(J_op, np.kron(I_mode_super, I_mode_super))

pops = np.eye(3*mode_basis_size**2).flatten()

# combine and do counting statistics
B = 0.1
damping_rate = 2.

for i,E in enumerate(bias_values):
    print E
    L = np.kron(electronic_liouvillian(E, T_c), np.kron(I_mode_super, I_mode_super)) \
                + np.kron(I_el_super, np.kron(vibrational_liouvillian_L(damping_rate, B), I_mode_super)) \
                + np.kron(I_el_super, np.kron(I_mode_super, vibrational_liouvillian_R(damping_rate, B))) \
                + interaction_liouvillian_L() \
                + interaction_liouvillian_R()
    try:
        solver = FCSSolver(sp.csr_matrix(L), sp.csr_matrix(jump_operator()), pops)
        mean[i] = solver.mean()
        F2[i] = solver.second_order_fano_factor()
    except ArpackNoConvergence:
        print ("No convergence!")
    
import matplotlib.pyplot as plt
plt.plot(bias_values, F2)
plt.show()
    