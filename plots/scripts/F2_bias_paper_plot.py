import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

data = np.load('../../data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N7_K3_data.npz')
bias_values = data['bias_values']
beta_values = data['beta']
mean = data['mean']
F2 = data['F2']

data_np = np.load('../../data/HEOM_F2_bias_drude_data.npz')
bias_values_np = data_np['bias_values']
F2_np = data_np['F2'][0]

data_diss = np.load('../../data/DQD_dissipative_F2_strong_coupling_UBO.npz')
bias_diss = data_diss['bias']
beta_diss = data_diss['beta']
F2_diss = data_diss['F2']

plt.figure(figsize=(8,7))

#plt.axhline(1, ls='--', color='grey')
plt.plot(bias_diss, F2_diss[0], linewidth=3, ls='--', color='k', label='no phonons')
plt.plot(bias_diss, F2_diss[3], linewidth=3, ls='--', color='r', label='WCA')
plt.plot(bias_values, F2[2], linewidth=3, ls='-', color='r', label='NP')

plt.xlabel(r'bias $\epsilon$')
plt.ylabel('Fano factor')

plt.legend(fontsize=14).draggable()
plt.show()