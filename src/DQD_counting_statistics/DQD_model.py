'''
Created on 8 Mar 2016

@author: rstones
'''
import numpy as np
from counting_statistics.counting_statistics import CountingStatistics

class DQDModel(CountingStatistics):
    
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
                           [0, 0, 0]]), self.Gamma_L), (np.array([[0, 0, 1.],
                                                                  [0, 0, 0],
                                                                  [0, 0, 0]]), self.Gamma_R)]
    
    def jump_operators(self):
        return self.lead_operators()