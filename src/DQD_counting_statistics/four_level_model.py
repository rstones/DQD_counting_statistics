'''
Created on 17 Mar 2016

@author: rstones
'''
import numpy as np
from counting_statistics.counting_statistics import CountingStatistics

class FourLevelModel(CountingStatistics):
    
    def __init__(self, remove_elements=False):
        
        self.system_dimension = 4
        
        self.bias = 0
        self.tunnelling_coupling = 3.
        
        # lead coupling rates
        self.Gamma_L = 1.
        self.Gamma_R = 1.
        
        # excitation rates
        self.gamma_ex = 0.4
        
        # call __init__ of super class to instantiate attributes required for counting statistics calculations
        CountingStatistics.__init__(self, remove_elements)
    
    # basis { empty, L, R }
    def system_hamiltonian(self):
        return np.array([[0, 0, 0, 0],
                         [0, self.bias/2., self.tunnelling_coupling, 0],
                         [0, self.tunnelling_coupling, -self.bias/2., 0],
                         [0, 0, 0, 0]])
        
    def lead_operators(self):
        return [(np.array([[0, 0, 0, 1.],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]]), self.Gamma_L), (np.array([[0, 0, 0, 0],
                                                                  [0, 0, 0, 0],
                                                                  [0, 0, 0, 0],
                                                                  [0, 0, 1., 0]]), self.Gamma_R)]
        
    def excitation_operators(self):
        return [(np.array([[0, 0, 0, 0],
                           [1., 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]]), self.gamma_ex), (np.array([[0, 1., 0, 0],
                                                                              [0, 0, 0, 0],
                                                                              [0, 0, 0, 0],
                                                                              [0, 0, 0, 0]]), self.gamma_ex+0.01)]
    
    def jump_operators(self):
        return self.lead_operators() + self.excitation_operators()