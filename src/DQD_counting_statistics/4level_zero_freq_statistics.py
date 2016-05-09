'''
Created on 17 Mar 2016

@author: rstones
'''
import numpy as np
import quant_mech.utils as utils
from DQD_counting_statistics.DQD_model import DQDModel
import matplotlib.pyplot as plt
from DQD_counting_statistics.four_level_model import FourLevelModel

bias_values = np.array([0, 1.5, 3., 4.5, 6.])
Gamma_R_range = np.logspace(-4, 3, 1000)
model = FourLevelModel(remove_elements=True)

current = np.zeros((bias_values.size, Gamma_R_range.size))
F2 = np.zeros((bias_values.size, Gamma_R_range.size))
coherence = np.zeros((bias_values.size, Gamma_R_range.size), dtype='complex')

for i,v in enumerate(bias_values):
    model.bias = v
    for j,Gamma_R in enumerate(Gamma_R_range):
        model.Gamma_R = Gamma_R
        ss = utils.stationary_state_svd(model.liouvillian(), model.density_vector_populations())
        current[i,j] = model.mean(ss)
        F2[i,j] = model.second_order_fano_factor(ss)
        coherence[i,j] = ss[2]

#np.savez('../../data/four_level_zero_freq_counting_statistics_data.npz', Gamma_R_range=Gamma_R_range, bias_values=bias_values, current=current, F2=F2, coherence=coherence)

fig,(ax1,ax2,ax3) = plt.subplots(1,3)
for i,v in enumerate(bias_values):
    ax1.semilogx(Gamma_R_range, current[i], label=v)
    ax2.semilogx(Gamma_R_range, F2[i], label=v)
    ax3.semilogx(Gamma_R_range, np.real(coherence[i]), label=v)
ax1.legend().draggable()
ax2.legend().draggable()
ax3.legend().draggable()
plt.show()