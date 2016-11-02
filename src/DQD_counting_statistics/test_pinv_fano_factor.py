'''
Created on 26 Oct 2016

@author: richard
'''
import numpy as np
import scipy.linalg as la
import quant_mech.utils as utils
from DQD_counting_statistics.DQD_model import DQDModel

model = DQDModel(remove_elements=True)
model.Gamma_R = 1.e-4

energy_gap_values = np.linspace(-10, 10, 100)
fano_factor = np.zeros(energy_gap_values.size)

for i,E in enumerate(energy_gap_values):
    model.bias = E
    ss = utils.stationary_state_svd(model.liouvillian(0), model.density_vector_populations())
    fano_factor[i] = model.second_order_fano_factor(ss)
import matplotlib.pyplot as plt
plt.plot(energy_gap_values, fano_factor)
plt.show()

# chi_values = np.linspace(-1, 1, 100)
# cgf = []#np.zeros(chi_values.size)
# for chi in chi_values:
#     cgf.append(model.cumulant_generating_function_temp(chi))
#     
# print chi_values
# print
# print cgf
#     
# import matplotlib.pyplot as plt
# plt.plot(chi_values, cgf)
# plt.show()
