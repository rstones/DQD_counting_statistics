'''
Created on 7 Jun 2017

@author: richard
'''
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

bias_values = np.linspace(-10, 10, 100)
mean = np.zeros((len(beta_values), bias_values.size))
F2 = np.zeros((len(beta_values), bias_values.size))

delta_values = np.zeros((len(beta_values), bias_values.size))

jump_rates = np.array([Gamma_L, Gamma_R])

drude_reorg_energy = 0.015
drude_cutoff = 50.

def electronic_hamiltonian(bias):
    return np.array([[0, 0, 0],
                 [0, bias/2., T_c],
                 [0, T_c, -bias/2.]])
I_el = np.eye(3)

def N(omega, beta):
    return 1. / (np.exp(beta*omega) - 1.)
    
# def spectral_density(delta, freq, hr_factor, damping):
#     reorg_energy = freq * hr_factor
#     return 2. * reorg_energy * freq**2 \
#                 * ((delta * damping) / ((delta**2 - freq**2)**2 + delta**2 * damping**2))
                
def spectral_density(delta, reorg_energy, cutoff):
    return (2. * reorg_energy * cutoff * delta) / (delta**2 + cutoff**2)

def gamma(E, J, B):
        delta = np.sqrt(E**2 + 4.*J**2)
        return (2.*np.pi * J**2 / delta**2) * spectral_density(delta, drude_reorg_energy, drude_cutoff) \
            * (1. / np.tanh(B*delta/2.))
    
def gamma_plus_minus(E, J, B, pm):
        delta = np.sqrt(E**2 + 4.*J**2)
        return - spectral_density(delta, drude_reorg_energy, drude_cutoff) \
                                                * ((E * J * np.pi / (2.*delta**2)) \
                                                * (1. / np.tanh(B*delta/2.)) \
                                                - pm * (J * np.pi / (2.*delta)))
                                                
def planck_distribution(freq, beta):
    return (np.exp(freq*beta) - 1) ** -1

'''Calculates relaxation rate between excitons for overdamped Brownian oscillator spectral density'''                                        
def exciton_relaxation_rate(freq, ex1, ex2, beta, drude_reorg_energy, drude_cutoff_freq):
    return 2. * spectral_density(np.abs(freq), drude_reorg_energy, drude_cutoff_freq) \
                    * np.abs(planck_distribution(freq, beta))# \
                    #* np.sum([np.abs(ex1)**2 * np.abs(ex2)**2])

lead_operators = np.array([np.array([[0, 0, 0],
                                      [1., 0, 0],
                                      [0, 0, 0]]), \
                           np.array([[0, 0, 1.],
                                     [0, 0, 0],
                                     [0, 0, 0]])])

'''These are Lindblad operators in exciton basis { 0, +, - }'''
electronic_dissipation_operators = np.array([np.array([[0,0,0], # pure dephasing on +
                                                       [0,1.,0],
                                                       [0,0,0]]), \
                                             np.array([[0,0,0], # pure dephasing on -
                                                       [0,0,0],
                                                       [0,0,1.]]), \
                                             np.array([[0,0,0], # relaxation + -> -
                                                       [0,0,0],
                                                       [0,1.,0]]), \
                                             np.array([[0,0,0], # relaxation - -> +
                                                       [0,0,1.],
                                                       [0,0,0]])])

jump_operator = Gamma_R * np.kron(lead_operators[1], lead_operators[1])
pops = np.eye(3).flatten()

for i,B in enumerate(beta_values):
    for j,E in enumerate(bias_values):
        #print E
        H_el = electronic_hamiltonian(E)
        evals,evecs = la.eigh(H_el[1:,1:])
        exciton_to_site_transform = np.eye(3)
        exciton_to_site_transform[1:,1:] = evecs
        delta = np.sqrt(E**2 + 4.*T_c**2)
        delta_values[i,j] = evals[0] - evals[1]
        #delta = evals[0] - evals[1]
        down_rate = exciton_relaxation_rate(-delta, evecs[:,0], evecs[:,1], B, drude_reorg_energy, drude_cutoff)
        up_rate = exciton_relaxation_rate(delta, evecs[:,0], evecs[:,1], B, drude_reorg_energy, drude_cutoff)
#         print down_rate
#         print up_rate
        L = os.super_operator(H_el, [(lead_operators[0], Gamma_L),
                                  (lead_operators[1], Gamma_R),
                                (np.dot(exciton_to_site_transform, np.dot(electronic_dissipation_operators[0], exciton_to_site_transform.T)), gamma(E, T_c, B)), 
                                (np.dot(exciton_to_site_transform, np.dot(electronic_dissipation_operators[1], exciton_to_site_transform.T)), gamma(E, T_c, B)),
                                (evecs[0,0]**2 * evecs[0,1]**2 * np.dot(exciton_to_site_transform.T, np.dot(electronic_dissipation_operators[2], exciton_to_site_transform)), down_rate if E >= 0 else up_rate),
                                (evecs[1,0]**2 * evecs[1,1]**2 * np.dot(exciton_to_site_transform.T, np.dot(electronic_dissipation_operators[3], exciton_to_site_transform)), up_rate if E >= 0 else down_rate),
                                (evecs[1,0]**2 * evecs[1,1]**2 * np.dot(exciton_to_site_transform.T, np.dot(electronic_dissipation_operators[2], exciton_to_site_transform)), down_rate if E >= 0 else up_rate),
                                (evecs[0,0]**2 * evecs[0,1]**2 * np.dot(exciton_to_site_transform.T, np.dot(electronic_dissipation_operators[3], exciton_to_site_transform)), up_rate if E >= 0 else down_rate)
                                  ])
        
        try:
            solver = FCSSolver(sp.csr_matrix(L), sp.csr_matrix(jump_operator), pops)
            mean[i,j] = solver.mean()
            F2[i,j] = solver.second_order_fano_factor()
        except ArpackNoConvergence:
            print ("No convergence!")
                
# np.savez('../../data/DQD_mode_mean_F2_exact_data.npz', beta_values=beta_values, damping_values=mode_damping_values, \
#          bias_values=bias_values, F2=F2, mean=mean)
        
import matplotlib.pyplot as plt


data = np.load('../../data/DQD_dissipative_F2_bias_brandes_liouvillian.npz')
target_bias_values = data['bias_values']
target_beta_values = data['beta_values']
target_F2 = data['F2']

plt.subplot(121)
#for i in range(len(beta_values)):
plt.plot(bias_values, mean[0], label=beta_values[0])
plt.legend().draggable()
 
plt.subplot(122)
#for i in range(len(beta_values)):
plt.axvline(0, color='r')
plt.plot(target_bias_values, target_F2[3], ls='--', color='grey')
plt.plot(bias_values, F2[0], label=beta_values[0], linewidth=3)
#plt.plot(bias_values, delta_values[0], color='k')
plt.ylim(0.8, 1.25)
plt.legend().draggable()
 
plt.show()




