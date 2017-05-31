'''
Created on 19 Jan 2017

@author: richard
'''
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numdifftools as ndt
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.utils as utils

class DQDHEOMModelSparse():
    
    def __init__(self, Gamma_L, Gamma_R, bias, T_c, environment=[], beta=1., K=0, tc=True, trunc_level=5, \
                 dissipator_test=False):
        
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
        self.environment = environment
        
        self.filter = False
        
        self.K = K
        self.tc = tc
        self.heom_solver = HierarchySolver(self.system_hamiltonian(), self.environment, \
                                           self.beta, self.jump_operators, self.jump_rates, N=trunc_level,\
                                           num_matsubara_freqs=self.K, temperature_correction=self.tc, \
                                           dissipator_test=dissipator_test)
        self.truncation_level = trunc_level
        #self.heom_solver.truncation_level = self.truncation_level
        
        self.dv_pops = np.zeros(self.system_dimension**2 * self.heom_solver.number_density_matrices())
        self.dv_pops[:self.system_dimension**2] = np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.])
        
    def system_hamiltonian(self):
        return np.array([[0, 0, 0],
                         [0, self.bias/2., self.T_c],
                         [0, self.T_c, -self.bias/2.]])

    def heom_matrix(self):
        self.heom_solver = HierarchySolver(self.system_hamiltonian(), self.environment, \
                                           self.beta, self.jump_operators, self.jump_rates, N=self.truncation_level, \
                                           num_matsubara_freqs=self.K, temperature_correction=self.tc)
        #self.heom_solver.truncation_level = self.truncation_level
        return sp.csr_matrix(self.heom_solver.construct_hierarchy_matrix_super_fast())

    def jump_matrix(self):
        if not self.filter:
            heom_dim = self.heom_solver.number_density_matrices() * self.heom_solver.system_dimension**2
        else:
            heom_dim = self.heom_solver.num_dms * self.heom_solver.system_dimension**2
        LJ = sp.lil_matrix((heom_dim, heom_dim), dtype='complex128')
        LJ[0,8] = self.Gamma_R
        return LJ
    