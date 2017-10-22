import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt
import matplotlib as mpl
from prettyplotlib import brewer2mpl

font = {'size':12}
mpl.rc('font', **font)

#data = np.load('../../data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N6_K3_beta2_data_fixed.npz')
data = np.load('../../data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N7_K3_more_data.npz')
beta = data['beta']
#beta_index = data['beta_index']
bias_values = data['bias_values']
F2 = data['F2']
mean = data['mean']

from scipy.interpolate import interp1d
int_bias_values = np.linspace(-20,20,2000)
int_F2 = interp1d(bias_values, F2, kind='cubic')

data_mode = np.load('../../data/DQD_exact_mode_mean_F2_data.npz')
bias_values_mode = data_mode['bias_values']
F2_mode = data_mode['F2']
mean_mode = data_mode['mean']

# temp_data = np.load('../../data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N7_K3_data_temp.npz')
# temp_F2 = temp_data['F2'][2]
# temp_bias_values = temp_data['bias_values']

fig,ax = ppl.subplots(1, 2, figsize=(8,3.5))
lw = 2

plt.sca(ax[0])
plt.text(-20, 0.0184, 'a.')
ppl.plot(bias_values, mean, linewidth=lw, label='non-perturbative', show_ticks=True)
ppl.plot(bias_values_mode, mean_mode[0,0], linewidth=lw, label='coherent mode', show_ticks=True)
ax[0].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[0].set_ylabel(r'current / e')
ppl.legend(fontsize=10).draggable()


plt.sca(ax[1])
plt.text(-20, 0.98, 'b.')
ppl.plot(bias_values, F2, linewidth=lw, label='non-perturbative', show_ticks=True)
ppl.plot(bias_values_mode, F2_mode[0,0], linewidth=lw, label='coherent mode', show_ticks=True)
ax[1].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[1].set_ylabel('Fano factor')
ppl.legend(fontsize=10).draggable()

plt.tight_layout()
plt.show()


