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
        self.Gamma_R = 1.e-4
        
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
    
import quant_mech.utils as utils
model = DQDModel()
bias_values = np.linspace(-10,10,100)

F2 = np.zeros(bias_values.size)
current = np.zeros(bias_values.size)

for i,E in enumerate(bias_values):
    model.bias = 2.*E
    ss = utils.stationary_state_svd(model.liouvillian(0), np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.]))
    F2[i] = model.second_order_fano_factor(ss)
    current[i] = model.Gamma_R * ss[-1]
    
import matplotlib.pyplot as plt

plt.subplot(121)
plt.plot(bias_values, F2)
plt.subplot(122)
plt.plot(bias_values, current)
plt.show()
