'''
Created on 28 Oct 2016

@author: richard
'''
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numdifftools as ndt
import quant_mech.utils as utils
from quant_mech.hierarchy_solver import HierarchySolver

np.set_printoptions(precision=4, linewidth=150, suppress=True)

class SRLHEOMModel():
    
    def __init__(self):
        self.hamiltonian = np.array([[100.,8.],[8.,0]])
        self.jump_operators = np.array([np.array([[0, 0],[1., 0]]), np.array([[0, 1.],[0, 0]])])
        self.Gamma_L = 1.
        self.Gamma_R = 1.#e-4
        self.jump_rates = np.array([self.Gamma_L, self.Gamma_R])
        
        self.dt = 1.e4
        
        self.hs = HierarchySolver(self.hamiltonian, 35., 40., temperature=300., jump_operators=self.jump_operators, jump_rates=self.jump_rates)
        self.hs.truncation_level = 4
        self.hm = self.hs.construct_hierarchy_matrix_super_fast().tolil()
        #print self.hm.todense()
        
        self.init_ss_vector = spla.eigs(self.hm.tocsc(), k=1, sigma=0, which='LM', maxiter=1000)[1]

    def hm_chi(self, chi):
        new_hm = self.hm.copy()
#         sys_dim = self.hs.system_dimension
#         for i in range(self.hs.number_density_matrices()):
#             print  'i = ' + str(i*sys_dim**2)
#             print 'j = ' + str((i+1)*sys_dim**2 - 1)
#             new_hm[i*sys_dim**2,(i+1)*sys_dim**2 - 1] *= np.exp(chi)
        new_hm[0,3] *= np.exp(chi)
        return new_hm
    
    # define cumulant generating function
    def cumulant_generating_function(self, chi):
        W = self.hm_chi(chi).tocsc()
        ss = spla.eigs(W, k=1, sigma=0, which='LM', maxiter=1000, v0=self.init_ss_vector)[1]
        #ss = spla.svds(W, k=1, which='SM', maxiter=10000, v0=self.init_ss_vector)[0]
        ss /= np.trace(self.hs.extract_system_density_matrix(ss))
        dv_pops = np.zeros(4 * self.hs.number_density_matrices())
        dv_pops[:4] = np.array([1., 0, 0, 1.])
        #ss = utils.stationary_state_svd(W.todense(), dv_pops)
        return np.log(dv_pops.dot(spla.expm(W.multiply(self.dt)).dot(ss)))
    
    def zero_frequency_noise(self):
        CGF_dd = ndt.Derivative(self.cumulant_generating_function, n=2, method='central')
        return CGF_dd(0)[0] / self.dt

    def mean(self):
        ss = spla.eigs(self.hm_chi(0).tocsc(), k=1, sigma=0, which='LM', v0=self.init_ss_vector)[1]
        #ss = spla.svds(self.hm_chi(0).tocsc(), k=1, which='SM', maxiter=10000, v0=self.init_ss_vector)[0]
        ss = self.hs.extract_system_density_matrix(ss)
        ss /= np.trace(ss)
        return self.Gamma_R * ss[1,1]
    
    def fano_factor(self):
        return self.zero_frequency_noise() / self.mean()

model = SRLHEOMModel()

# find second derivative of CGF
print 'calculating noise...'
noise = model.zero_frequency_noise()
print 'S = ' + str(noise)

# find mean
print 'calculating mean...'
mean = model.mean()
print 'I = ' + str(mean)

# calculate zero-freq Fano factor
F2 = noise / mean
print 'F2 = ' + str(F2)
