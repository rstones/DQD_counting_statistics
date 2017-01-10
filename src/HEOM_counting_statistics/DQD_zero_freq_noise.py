'''
Created on 28 Nov 2016

@author: richard
'''
import numpy as np
from HEOM_counting_statistics.DQD_HEOM_model import DQDHEOMModel
from counting_statistics.fcs_solver import FCSSolver

model = DQDHEOMModel(0, 3., 1., 1.e-4)
 
bias_values = np.linspace(-10, 10, 100)
current = np.zeros(bias_values.size)
F2 = np.zeros(bias_values.size)
 
for i,E in enumerate(bias_values):
    model.bias = E
    solver = FCSSolver(model.heom_matrix(), model.jump_matrix(), model.dv_pops)
    current[i] = solver.mean()
    F2[i] = solver.second_order_fano_factor(0)
     
import matplotlib.pyplot as plt
plt.subplot(121)
plt.plot(bias_values, current, linewidth=3)
plt.xlim(-10,10)
plt.ylim(0,0.04)
plt.xlabel(r'energy bias $\epsilon$')
plt.ylabel(r'current')
 
plt.subplot(122)
plt.plot(bias_values, F2, linewidth=3)
plt.xlim(-10,10)
plt.xlabel(r'energy bias $\epsilon$')
plt.ylabel(r'F$^{(2)}$(0)')
 
plt.show()
    