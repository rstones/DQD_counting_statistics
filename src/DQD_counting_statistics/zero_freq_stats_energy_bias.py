'''
Created on 21 Mar 2016

@author: rstones
'''
import numpy as np
import quant_mech.utils as utils
from DQD_counting_statistics.DQD_model import DQDModel
import matplotlib.pyplot as plt
import matplotlib

font = {'size':20}
matplotlib.rc('font', **font)

bias_values = np.linspace(-10, 10, 200)
model = DQDModel(remove_elements=True)
model.Gamma_R = 1.e-4

current = np.zeros(bias_values.size)
F2 = np.zeros(bias_values.size)
coherence = np.zeros(bias_values.size, dtype='complex')

for i,v in enumerate(bias_values):
    model.bias = 2.*v
    ss = utils.stationary_state_svd(model.liouvillian(), model.density_vector_populations())
    current[i] = model.mean(ss)
    F2[i] = model.second_order_fano_factor(ss)
    coherence[i] = ss[2]

plt.subplot(131)
plt.plot(bias_values, current, linewidth=3)
plt.xlabel(r'energy bias $\epsilon$')
plt.ylabel(r'current')
plt.subplot(132)
plt.plot(bias_values, F2, linewidth=3)
plt.xlabel(r'energy bias $\epsilon$')
plt.ylabel(r'F$^{(2)}$(0)')
plt.subplot(133)
plt.plot(bias_values, np.abs(coherence), linewidth=3)
plt.xlabel(r'energy bias $\epsilon$')
plt.ylabel(r'|coherence|')
plt.show()
