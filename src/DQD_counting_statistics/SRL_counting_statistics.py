'''
Created on 27 Oct 2016

@author: richard
'''
import numpy as np
import quant_mech.utils as utils
from counting_statistics.counting_statistics import CountingStatistics

class SRLModel(CountingStatistics):
    
    def __init__(self, remove_elements=False):
        self.system_dimension = 2
        self.Gamma_R = 1.
        self.Gamma_L = 1.
        CountingStatistics.__init__(self, remove_elements)
    
    def system_hamiltonian(self):
        return np.array([[0,0],[0,0]])
    
    def lead_operators(self):
        return [(np.array([[0,0],[1.,0]]), self.Gamma_L), (np.array([[0,1.],[0,0]]), self.Gamma_R)]
    
    def jump_operators(self):
        return self.lead_operators()
    
    def analytic_F2(self):
        return (self.Gamma_L**2 + self.Gamma_R**2) / (self.Gamma_L + self.Gamma_R)**2
    
Gamma_R_values = np.linspace(1.e-4,10,100)

model = SRLModel()

anal_F2 = np.zeros(Gamma_R_values.size)
num_F2 = np.zeros(Gamma_R_values.size)
mean = np.zeros(Gamma_R_values.size)

for i,GR in enumerate(Gamma_R_values):
    model.Gamma_R = GR
    ss = utils.stationary_state_svd(model.liouvillian(), model.density_vector_populations())
    anal_F2[i] = model.analytic_F2()
    num_F2[i] = model.second_order_fano_factor(ss)
    mean[i] = model.mean(ss)
    
import matplotlib.pyplot as plt

plt.subplot(121)
plt.plot(Gamma_R_values, mean)

plt.subplot(122)
plt.plot(Gamma_R_values, anal_F2, label='anal')
plt.plot(Gamma_R_values, num_F2, label='num')
plt.legend().draggable()
plt.show()
