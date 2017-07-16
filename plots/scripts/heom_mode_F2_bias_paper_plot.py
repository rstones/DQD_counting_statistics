import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

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

plt.figure(figsize=(8,7))

#plt.subplot(121)
# plt.plot(bias_values, mean * 100, linewidth=3, color='g', label='non-perturbative')
# plt.plot(bias_values_mode, mean_mode[0,0] * 100, linewidth=3, color='g', ls='--', label='coherent mode')
# plt.xlabel(r'bias $(1 / T_c)$')
# plt.ylabel(r'mean $(10^2 / e T_c)$')
# plt.text(-19, 1.69, '(a)', fontsize=22)

# plt.subplot(122)
#plt.plot(int_bias_values, int_F2(int_bias_values), linewidth=3, color='g', label='non-perturbative')
plt.plot(bias_values, F2, linewidth=3, color='g', label='non-perturbative')
plt.plot(bias_values_mode, F2_mode[0,0], linewidth=3, color='g', ls='--', label='coherent mode')
#plt.plot(temp_bias_values, temp_F2)
plt.xlabel(r'bias $(1 / T_c)$')
plt.ylabel('Fano factor')
plt.text(-19, 0.97, '(b)', fontsize=22)

plt.legend(fontsize=14).draggable()
plt.show()