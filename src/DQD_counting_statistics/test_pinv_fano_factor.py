'''
Created on 26 Oct 2016

@author: richard
'''
import numpy as np
import scipy.linalg as la
import quant_mech.utils as utils
import counting_statistics.counting_statistics as cs
from DQD_counting_statistics.DQD_model import DQDModel

def analytic_R0(Gamma_L, Gamma_R):
    return np.array([[Gamma_L**2 + Gamma_L*Gamma_R, -Gamma_R**2 - Gamma_L*Gamma_R],
                     [-Gamma_L**2 - Gamma_L*Gamma_R, Gamma_R**2 + Gamma_L*Gamma_R]]) / (Gamma_L + Gamma_R)**3

def zero_freq_noise(liouvillian, jump_liouvillian, sys_dim, stationary_state, dv_pops, Gamma_L, Gamma_R):
    J_1 = cs.differentiate_jump_matrix(jump_liouvillian)
    Q = np.eye(sys_dim**2) - np.outer(stationary_state, dv_pops)
    R0 = np.dot(Q, np.dot(la.pinv2(-liouvillian), Q)) #analytic_R0(Gamma_L, Gamma_R) # 
    
    noise = - cs.trace_density_vector(np.dot(cs.differentiate_jump_matrix(J_1), stationary_state), dv_pops) \
                        - 2. * cs.trace_density_vector(np.dot(np.dot(np.dot(J_1, R0), J_1), stationary_state), dv_pops)
    
    return noise

def analytic_F2(Gamma_L, Gamma_R):
    return (Gamma_L**2 + Gamma_R**2) / (Gamma_L + Gamma_R)**2

Gamma_L = 1.
Gamma_R = 2.
L = np.array([[-Gamma_L,Gamma_R],
              [Gamma_L,-Gamma_R]])
LJ = np.array([[0,Gamma_R],
              [0,0]])
dv_pops = np.array([1,1])
ss = utils.stationary_state(L, dv_pops)
print L
print analytic_R0(Gamma_L, Gamma_R)
print ss
print zero_freq_noise(L, LJ, np.sqrt(2), ss, dv_pops, Gamma_L, Gamma_R) / (Gamma_R*ss[-1])
print analytic_F2(Gamma_L, Gamma_R)