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

for i,beta in enumerate(beta_values):
    plt.subplot(121)
    plt.plot(bias_diss, F2_diss[i+1], linewidth=3, label=r'$\beta = $ '+str(beta))
    plt.subplot(122)
    plt.plot(bias_values, F2[i], linewidth=3)
    
    
plt.subplot(121)
plt.plot(bias_diss, F2_diss[0], linewidth=3, ls='--', color='k')
plt.xlabel(r'bias $\epsilon$')
plt.ylabel('Fano factor')
plt.legend().draggable()

plt.subplot(122)
plt.plot(bias_values_np, F2_np, linewidth=3, ls='--', color='k')
plt.xlabel(r'bias $\epsilon$')
plt.ylabel('Fano factor')
    
plt.show()