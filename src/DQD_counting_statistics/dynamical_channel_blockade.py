'''
Created on 31 May 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
from counting_statistics.counting_statistics import CountingStatistics

class DQDDynamicalChannelBlockade(CountingStatistics):
    
    def __init__(self, remove_elements=False):
        
        self.system_dimension = 3
        
        self.bias = 0
        self.tunnelling_coupling = 3.
        
        self.Gamma_L = 1.
        self.Gamma_R = 1.
        
        # call __init__ of super class to instantiate attributes required for counting statistics calculations
        CountingStatistics.__init__(self, remove_elements)
    
    # basis { empty, L, R }
    def system_hamiltonian(self):
        return np.array([[0, 0, 0],
                         [0, self.bias/2., self.tunnelling_coupling],
                         [0, self.tunnelling_coupling, -self.bias/2.]])
        
    def lead_operators(self):
        return [(np.array([[0, 0, 0],
                           [1., 0, 0],
                           [1., 0, 0]]), self.Gamma_L), (np.array([[0, 1., 1.],
                                                                  [0, 0, 0],
                                                                  [0, 0, 0]]), self.Gamma_R)]
    
    def jump_operators(self):
        return self.lead_operators()

#     def liouvillian(self, Gamma_L, Gamma_R, epsilon, T_c):
#         theta = 0.5 * np.arctan(T_c / epsilon)
#         return np.array([[-Gamma_L, np.sin(theta)**2 * Gamma_R, np.cos(theta)**2 * Gamma_R],
#                          [np.sin(theta)**2 * Gamma_L, -np.sin(theta)**2 * Gamma_R, 0],
#                          [np.cos(theta)**2 * Gamma_L, 0, -np.cos(theta)**2 * Gamma_R]])
    

    
