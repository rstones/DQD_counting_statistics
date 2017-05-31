import numpy as np
import matplotlib.pyplot as plt
import matplotlib

data = np.load('../../data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N7_K3_data.npz')
bias_values = data['bias_values']
beta_values = data['beta']
mean = data['mean']
F2 = data['F2']

data_np = np.load('../../data/HEOM_F2_bias_drude_data.npz')
bias_values_np = data_np['bias_values']
F2_np = data_np['F2'][0]

for i,beta in enumerate(beta_values):
    plt.subplot(121)
    plt.plot(bias_values, mean[i], linewidth=3)
    plt.subplot(122)
    plt.plot(bias_values, F2[i], linewidth=3)
    
plt.plot(bias_values_np, F2_np, ls='--', color='k', linewidth=3)

plt.show()