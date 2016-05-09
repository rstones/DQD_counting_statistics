'''
Created on 14 Mar 2016

@author: rstones
'''
import numpy as np
import quant_mech.utils as utils
from DQD_counting_statistics.DQD_model import DQDModel
import matplotlib.pyplot as plt

bias_values = np.array([0, 1.5, 3., 4.5, 6.])
model = DQDModel(remove_elements=True)
model.Gamma_R = 1.e-8

current = np.zeros(bias_values.size)
F2 = np.zeros(bias_values.size)
coherence = np.zeros(bias_values.size, dtype='complex')

for i,v in enumerate(bias_values):
    model.bias = v
    ss = utils.stationary_state_svd(model.liouvillian(), model.density_vector_populations())
    print ss
    current[i] = model.mean(ss)
    F2[i] = model.second_order_fano_factor(ss)
    coherence[i] = ss[2]

plt.subplot(121)
plt.plot(bias_values, current)
plt.subplot(122)
plt.plot(bias_values, F2)
plt.show()
