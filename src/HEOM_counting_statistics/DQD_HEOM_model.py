'''
Created on 1 Nov 2016

@author: richard
'''
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numdifftools as ndt
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.utils as utils
from quant_mech.OBOscillator import OBOscillator

class DQDHEOMModel():
    
    def __init__(self, Gamma_L, Gamma_R, bias, T_c, drude_reorg_energy=1.e-12, drude_cutoff=40., beta=1., \
                 K=0, temperature_correction=True):
        
        self.system_dimension = 3
        
        self.bias = bias
        self.T_c = T_c
        
        self.Gamma_L = Gamma_L
        self.Gamma_R = Gamma_R
        
        self.jump_operators = np.array([np.array([[0, 0, 0],
                                                  [1., 0, 0],
                                                  [0, 0, 0]]), np.array([[0, 0, 1.],
                                                                         [0, 0, 0],
                                                                         [0, 0, 0]])])
        self.jump_rates = np.array([self.Gamma_L, self.Gamma_R])
        
        self.beta = beta
        self.drude_reorg_energy = drude_reorg_energy
        self.drude_cutoff = drude_cutoff
        
        self.num_matsubara_freqs = K
        self.temperature_correction = temperature_correction
        self.environment = [(), \
                            (OBOscillator(self.drude_reorg_energy, self.drude_cutoff, self.beta, K=K),), \
                            (OBOscillator(self.drude_reorg_energy, self.drude_cutoff, self.beta, K=K),)]
        self.heom_solver = HierarchySolver(self.system_hamiltonian(), self.environment, \
                                           self.beta, self.jump_operators, self.jump_rates, \
                                           num_matsubara_freqs=self.num_matsubara_freqs, \
                                           temperature_correction=self.temperature_correction)
        self.truncation_level = 4
        self.heom_solver.truncation_level = self.truncation_level
        #self.construct_heom()
        
        self.dv_pops = np.zeros(self.system_dimension**2 * self.heom_solver.number_density_matrices())
        self.dv_pops[:self.system_dimension**2] = np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.])
        
        self.dt = 1.e4
        
    def system_hamiltonian(self):
        return np.array([[0, 0, 0],
                         [0, self.bias/2., self.T_c],
                         [0, self.T_c, -self.bias/2.]])

    def construct_heom(self):
        self.heom_solver = HierarchySolver(self.system_hamiltonian(), self.drude_reorg_energy, self.drude_cutoff, \
                                           self.beta, self.jump_operators, self.jump_rates, \
                                           num_matsubara_freqs=self.num_matsubara_freqs, \
                                           temperature_correction=self.temperature_correction)
        self.heom_solver.truncation_level = self.truncation_level
        self.heom = self.heom_solver.construct_hierarchy_matrix_super_fast().tolil()
#         if self.truncation_level > 1:
#             self.init_ss_vector = spla.eigs(self.heom.tocsc(), k=1, sigma=0, which='LM', maxiter=1000)[1]

    def heom_matrix(self):
        self.heom_solver = HierarchySolver(self.system_hamiltonian(), self.environment, \
                                           self.beta, self.jump_operators, self.jump_rates, \
                                           num_matsubara_freqs=self.num_matsubara_freqs, \
                                           temperature_correction=self.temperature_correction)
        self.heom_solver.truncation_level = self.truncation_level
        return np.asarray(self.heom_solver.construct_hierarchy_matrix_super_fast().todense(), dtype='complex128')

    '''
    Make this sparse eventually
    '''
    def jump_matrix(self):
        heom_dim = self.heom_solver.number_density_matrices() * self.heom_solver.system_dimension**2
        LJ = np.zeros((heom_dim, heom_dim))
        LJ[0,8] = self.Gamma_R
#         for i in range(self.heom_solver.number_density_matrices()):
#             LJ[i*self.system_dimension**2,(i+1)*self.system_dimension**2 - 1] = self.Gamma_R
        return LJ
        
#     def heom_chi(self, chi):
#         heom = self.heom.copy()
#         heom[0,8] *= np.exp(chi)
# #         for i in range(self.heom_solver.number_density_matrices()):
# #             #print  '(' + str(i*sys_dim**2) + ', ' + str((i+1)*sys_dim**2 - 1) + ')'
# #             heom[i*self.system_dimension**2,(i+1)*self.system_dimension**2 - 1] *= np.exp(chi)
#         return heom
    
    
    
#     def cumulant_generating_function(self, chi):
#         W = self.heom_chi(chi).todense()
#         ss = utils.stationary_state(W, self.dv_pops)
#         return np.log(np.dot(self.dv_pops, np.dot(la.expm(W*self.dt), ss)))
#         
#     def zero_frequency_noise(self):
#         CGF_dd = ndt.Derivative(self.cumulant_generating_function, n=2, method='central')
#         return CGF_dd(0) / self.dt
#     
#     def zero_frequency_noise_R0(self):
#         ss = utils.stationary_state(self.heom.todense(), self.dv_pops)
#         ss /= np.trace(self.heom_solver.extract_system_density_matrix(ss))
#             
#         LJ = self.jump_matrix()   
#         Q = np.eye(self.system_dimension**2 * self.heom_solver.number_density_matrices()) - np.outer(ss, self.dv_pops)    
#         R0 = np.dot(Q, np.dot(la.pinv2(-self.heom.todense()), Q))
#         
#         return np.dot(self.dv_pops, np.dot(LJ, ss)) \
#                             + 2. * np.dot(self.dv_pops, np.dot(np.dot(np.dot(LJ, R0), LJ), ss))
#     
#     @staticmethod
#     def pseudoinverse(L, freq, Q):
#         return np.dot(Q, np.dot(la.pinv2(1.j*freq*np.eye(L.shape[0]) - L), Q))
#     
#     def finite_frequency_noise(self, freq):
#         ss = utils.stationary_state(self.heom.todense(), self.dv_pops)
#         ss /= np.trace(self.heom_solver.extract_system_density_matrix(ss))
#         
#         LJ = self.jump_matrix()   
#         Q = np.eye(self.system_dimension**2 * self.heom_solver.number_density_matrices()) - np.outer(ss, self.dv_pops)
#         noise = np.zeros(freq.size, dtype='complex128')
#         for i in range(len(freq)):
#             R_plus = self.pseudoinverse(self.heom.todense(), freq[i], Q)
#             R_minus = self.pseudoinverse(self.heom.todense(), -freq[i], Q)
#             noise[i] = np.dot(self.dv_pops, np.dot(LJ \
#                             + np.dot(np.dot(LJ, R_plus), LJ) \
#                                         + np.dot(np.dot(LJ, R_minus), LJ), ss))
#         return noise
#     
#     def mean(self):
#         ss = utils.stationary_state(self.heom.todense(), self.dv_pops)
#         #ss.shape = self.system_dimension, self.system_dimension
#         ss = self.heom_solver.extract_system_density_matrix(ss)
#         return self.Gamma_R * ss[2,2]
#     
#     def second_order_fano_factor(self):
#         return self.zero_frequency_noise_R0() / self.mean()
#     
#     def finite_frequency_fano_factor(self, freq):
#         return self.finite_frequency_noise(freq) / self.mean()
    