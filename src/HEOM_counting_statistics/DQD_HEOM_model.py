'''
Created on 1 Nov 2016

@author: richard
'''
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numdifftools as ndt
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.utils as utils

class DQDHEOMModel():
    
    def __init__(self):
        
        self.system_dimension = 3
        
        self.bias = 0
        self.T_c = 3.
        
        self.Gamma_L = 1.
        self.Gamma_R = 1.e-4
        
        self.jump_operators = np.array([np.array([[0, 0, 0],
                                                  [1., 0, 0],
                                                  [0, 0, 0]]), np.array([[0, 0, 1.],
                                                                         [0, 0, 0],
                                                                         [0, 0, 0]])])
        self.jump_rates = np.array([self.Gamma_L, self.Gamma_R])
        
        self.temperature = 300.
        self.drude_reorg_energy = 0.2
        self.drude_cutoff = 40.
        
        self.heom_solver = HierarchySolver(self.system_hamiltonian(), self.drude_reorg_energy, self.drude_cutoff, self.temperature, self.jump_operators, self.jump_rates)
        self.truncation_level = 1
        self.heom_solver.truncation_level = self.truncation_level
        self.construct_heom()
        
        self.dt = 1.e4
        
    def system_hamiltonian(self):
        return np.array([[0, 0, 0],
                         [0, self.bias/2., self.T_c],
                         [0, self.T_c, -self.bias/2.]])

    def construct_heom(self):
        self.heom_solver = HierarchySolver(self.system_hamiltonian(), self.drude_reorg_energy, self.drude_cutoff, self.temperature, self.jump_operators, self.jump_rates)
        self.heom_solver.truncation_level = self.truncation_level
        self.heom = self.heom_solver.construct_hierarchy_matrix_super_fast().tolil()
        if self.truncation_level > 1:
            self.init_ss_vector = spla.eigs(self.heom.tocsc(), k=1, sigma=0, which='LM', maxiter=1000)[1]
#         else:
#             self.init_ss_vector = utils.stationary_state_svd(self.heom.todense(), np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.]))
        
    def heom_chi(self, chi):
        heom = self.heom.copy()
        #heom[0,8] *= np.exp(chi)
        for i in range(self.heom_solver.number_density_matrices()):
            #print  '(' + str(i*sys_dim**2) + ', ' + str((i+1)*sys_dim**2 - 1) + ')'
            heom[i*self.system_dimension**2,(i+1)*self.system_dimension**2 - 1] *= np.exp(chi)
        return heom
    
    def cumulant_generating_function(self, chi):
        W = self.heom_chi(chi).tocsc()
        if self.truncation_level > 1:
            ss = spla.eigs(W, k=1, sigma=0, which='LM', maxiter=1000, v0=self.init_ss_vector)[1]
            ss /= np.trace(self.heom_solver.extract_system_density_matrix(ss))
        else:
            ss = utils.stationary_state_svd(self.heom.todense(), np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.]))
            #ss.shape = self.system_dimension, self.system_dimension
        dv_pops = np.zeros(9 * self.heom_solver.number_density_matrices())
        dv_pops[:9] = np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.])
        return np.log(dv_pops.dot(spla.expm(W.multiply(self.dt)).dot(ss)))
    
    def zero_frequency_noise(self):
        CGF_dd = ndt.Derivative(self.cumulant_generating_function, n=2, method='central')
        return CGF_dd(0) / self.dt
    
    def mean(self):
        if self.truncation_level > 1:
            ss = spla.eigs(self.heom_chi(0).tocsc(), k=1, sigma=0, which='LM', v0=self.init_ss_vector)[1]
            ss = self.heom_solver.extract_system_density_matrix(ss)
            ss /= np.trace(ss)
            print ss.shape
        else:
            ss = utils.stationary_state_svd(self.heom.todense(), np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.]))
            ss.shape = self.system_dimension, self.system_dimension
            print ss.shape
        return self.Gamma_R * ss[2,2]
    
    def second_order_fano_factor(self):
        return self.zero_frequency_noise() / self.mean()
    
np.set_printoptions(precision=5, linewidth=150, suppress=True)
model = DQDHEOMModel()

bias_values = np.linspace(-10, 10, 100)
current = np.zeros(bias_values.size)
F2 = np.zeros(bias_values.size)

for i,E in enumerate(bias_values):
    print E
    model.bias = E
    model.construct_heom()
    current[i] = model.mean()
    F2[i] = model.second_order_fano_factor()
    
import matplotlib.pyplot as plt
plt.subplot(121)
plt.plot(bias_values, F2)
plt.subplot(122)
plt.plot(bias_values, current)
plt.show()
    