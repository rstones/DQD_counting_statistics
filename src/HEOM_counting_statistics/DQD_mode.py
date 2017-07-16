'''
Created on 3 Jun 2017

@author: richard
'''
import numpy as np
import scipy.linalg as la
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

mode_damping_values = [0.5, 2., 5.][:1]

modeL_freq = 10.
modeL_hr_factor = 0.5
modeL_damping = 2. #0.5
modeL_coupling = modeL_freq * np.sqrt(modeL_hr_factor) # DOUBLE CHECK THIS!!!

modeR_freq = 10.
modeR_hr_factor = 0.5
modeR_damping = 2. #0.5
modeR_coupling = modeR_freq * np.sqrt(modeR_hr_factor) # DOUBLE CHECK THIS!!!

drude_reorg_energy = 0.015
drude_cutoff = 50.

bias_values = np.linspace(-20, 20, 100)
mean = np.zeros((len(mode_damping_values), len(beta_values), bias_values.size))
F2 = np.zeros((len(mode_damping_values), len(beta_values), bias_values.size))

jump_rates = np.array([Gamma_L, Gamma_R])

def electronic_hamiltonian(bias):
    return np.array([[0, 0, 0],
                 [0, bias/2., T_c],
                 [0, T_c, -bias/2.]])
I_el = np.eye(3)

mode_basis_size = 3
H_modeL = utils.vibrational_hamiltonian(modeL_freq, mode_basis_size)
H_modeR = utils.vibrational_hamiltonian(modeR_freq, mode_basis_size)
I_mode = np.eye(mode_basis_size)
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
    
def spectral_density(delta, freq, hr_factor, damping):
    reorg_energy = freq * hr_factor
    return 2. * reorg_energy * freq**2 \
                * ((delta * damping) / ((delta**2 - freq**2)**2 + delta**2 * damping**2))

def gamma(E, J, B):
        delta = np.sqrt(E**2 + 4.*J**2)
        return (2.*np.pi * J**2 / delta**2) * spectral_density(delta, modeL_freq, modeL_hr_factor, modeL_damping) \
            * (1. / np.tanh(B*delta/2.))
    
def gamma_plus_minus(E, J, B, pm):
        delta = np.sqrt(E**2 + 4.*J**2)
        return - spectral_density(delta, modeL_freq, modeL_hr_factor, modeL_damping) \
                                                * ((E * J * np.pi / (2.*delta**2)) \
                                                * (1. / np.tanh(B*delta/2.)) \
                                                + pm * (J * np.pi / (2.*delta)))
                                                
def planck_distribution(freq, beta):
    return (np.exp(freq*beta) - 1) ** -1

'''Calculates relaxation rate between excitons for overdamped Brownian oscillator spectral density'''                                        
def exciton_relaxation_rate(freq, ex1, ex2, beta, mode_freq, mode_hr_factor, mode_damping):
    return 2. * spectral_density(np.abs(freq), mode_freq, mode_hr_factor, mode_damping) \
                    * np.abs(planck_distribution(freq, beta)) \
                    * np.sum([np.abs(ex1)**2 * np.abs(ex2)**2])
    
modeL_damping_operators = np.array([np.kron(I_el, np.kron(up_mode, I_mode)), 
                                   np.kron(I_el, np.kron(down_mode, I_mode))])

modeR_damping_operators = np.array([np.kron(I_el, np.kron(I_mode, up_mode)), 
                                   np.kron(I_el, np.kron(I_mode, down_mode))])

lead_operators = np.array([np.kron(np.array([[0, 0, 0],
                                              [1., 0, 0],
                                              [0, 0, 0]]), np.kron(I_mode, I_mode)), \
                           np.kron(np.array([[0, 0, 1.],
                                             [0, 0, 0],
                                             [0, 0, 0]]), np.kron(I_mode, I_mode))])

electronic_dissipation_operators = np.array([np.kron(np.array([[0,0,0], # pure dephasing on L
                                                               [0,1.,0],
                                                               [0,0,0]]), np.kron(I_mode, I_mode)), \
                                             np.kron(np.array([[0,0,0], # pure dephasing on R
                                                               [0,0,0],
                                                               [0,0,1.]]), np.kron(I_mode, I_mode)), \
                                             np.kron(np.array([[0,0,0], # relaxation L -> R
                                                               [0,0,0],
                                                               [0,1.,0]]), np.kron(I_mode, I_mode)), \
                                             np.kron(np.array([[0,0,0], # relaxation R -> L
                                                               [0,0,1.],
                                                               [0,0,0]]), np.kron(I_mode, I_mode))]) 

jump_operator = Gamma_R * np.kron(lead_operators[1], lead_operators[1])
# np.set_printoptions(threshold=100000, linewidth=1000)
# print np.nonzero(jump_operator)
# print jump_operator
pops = np.eye(3*mode_basis_size**2).flatten()

for k,g in enumerate(mode_damping_values):
    modeL_damping = g
    modeR_damping = g
    #k = 0
    for i,B in enumerate(beta_values):
        L_damping_rates = modeL_damping_rates(modeL_damping, B)
        R_damping_rates = modeR_damping_rates(modeR_damping, B)
        for j,E in enumerate(bias_values):
            print E
            H_el = electronic_hamiltonian(E)
#             evals,evecs = la.eig(H_el[1:,1:])
#             exciton_to_site_transform = np.eye(3)
#             exciton_to_site_transform[1:,1:] = evecs
#             exciton_to_site_transform = np.kron(exciton_to_site_transform, np.kron(I_mode, I_mode))
#             
#             delta = np.sqrt(E**2 + 4.*T_c**2)
#             down_rate = exciton_relaxation_rate(-delta, evecs[:,0], evecs[:,1], B, drude_reorg_energy, drude_cutoff)
#             up_rate = exciton_relaxation_rate(delta, evecs[:,0], evecs[:,1], B, drude_reorg_energy, drude_cutoff)
            
            H = np.kron(H_el, np.kron(I_mode, I_mode)) + np.kron(I_el, np.kron(H_modeL, I_mode)) +  np.kron(I_el, np.kron(I_mode, H_modeR)) \
                         + modeL_coupling * np.kron(np.array([[0,0,0],[0,1.,0],[0,0,0]]), np.kron(up_mode + down_mode, I_mode)) \
                         + modeR_coupling * np.kron(np.array([[0,0,0],[0,0,0],[0,0,1.]]), np.kron(I_mode, up_mode + down_mode))
            L = os.super_operator(H, [(lead_operators[0], Gamma_L),
                                      (lead_operators[1], Gamma_R),
#                                     (np.dot(exciton_to_site_transform.T, np.dot(electronic_dissipation_operators[0], exciton_to_site_transform)), gamma(E, T_c, B)), 
#                                     (np.dot(exciton_to_site_transform.T, np.dot(electronic_dissipation_operators[1], exciton_to_site_transform)), gamma(E, T_c, B)),
#                                     (np.dot(exciton_to_site_transform.T, np.dot(electronic_dissipation_operators[2], exciton_to_site_transform)), down_rate if E >= 0 else up_rate),
#                                     (np.dot(exciton_to_site_transform.T, np.dot(electronic_dissipation_operators[3], exciton_to_site_transform)), up_rate if E >= 0 else down_rate),
                                      (modeL_damping_operators[0], L_damping_rates[0]),
                                      (modeL_damping_operators[1], L_damping_rates[1]),
                                      (modeR_damping_operators[0], R_damping_rates[0]),
                                      (modeR_damping_operators[1], R_damping_rates[1])
                                      ])
            
            try:
                solver = FCSSolver(sp.csr_matrix(L), sp.csr_matrix(jump_operator), pops)
                mean[k,i,j] = solver.mean()
                F2[k,i,j] = solver.second_order_fano_factor()
            except ArpackNoConvergence:
                print ("No convergence!")
                
np.savez('../../data/DQD_exact_mode_mean_F2_data.npz', beta_values=beta_values, damping_values=mode_damping_values, \
         bias_values=bias_values, F2=F2, mean=mean)
        
import matplotlib.pyplot as plt
 
plt.subplot(121)
#for i in range(len(beta_values)):
plt.plot(bias_values, mean[0,0], label=beta_values[0])
plt.legend().draggable()
 
plt.subplot(122)
#for i in range(len(beta_values)):
plt.plot(bias_values, F2[0,0], label=beta_values[0], linewidth=3)
plt.ylim(0.5, 1.3)
plt.legend().draggable()
 
plt.show()




