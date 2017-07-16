import numpy as np
import matplotlib.pyplot as plt
import matplotlib

data_heom = np.load('../../data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N7_K3_data.npz')
bias_values_heom = data_heom['bias_values']
beta_values_heom = data_heom['beta']
mean_heom = data_heom['mean']
F2_heom = data_heom['F2']

data = np.load('../../data/DQD_mode_mean_F2_exact_data.npz')
bias_values = data['bias_values']
beta_values = data['beta_values']
damping_values = data['damping_values']
F2 = data['F2']
mean = data['mean']

data_diss = np.load('../../data/DQD_dissipative_F2_strong_coupling_UBO.npz')
bias_diss = data_diss['bias']
beta_diss = data_diss['beta']
F2_diss = data_diss['F2']

plt.plot(bias_values_heom, F2_heom, linewidth=3, ls='-', color='g', label='non-perturbative')
plt.plot(bias_values, F2[1,2], linewidth=3, ls='--', color='g', label='coherent mode')
plt.plot(bias_diss, F2_diss[3], linewidth=3, color='grey', ls='--', label='weak coupling approx')

plt.xlabel('bias')
plt.ylabel('Fano factor')
plt.legend().draggable()
plt.show()