import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

data = np.load('../../data/HEOM_F2_bias_drude_data.npz')
bias_values = data['bias_values']
F2 = data['F2']
beta = data['beta']

# plt.subplot(121)
# plt.plot(bias_values, mean[0], linewidth=3, ls='--', color='k', label='no phonons')
# for i,B in enumerate(beta):
#     plt.plot(bias_values, mean[i+1], linewidth=3, label='T = ' + str(temperature[i]) + 'K')
# #plt.axhline(1, ls='--', color='grey')
# plt.xlim(-1.05, 1.05)
# plt.ylim(0, 0.0015)
# plt.xlabel(r'bias $\epsilon$ (meV)')
# plt.ylabel(r'current')
# 
# plt.subplot(122)

plt.figure(figsize=(8,7))
plt.plot(bias_values, F2[0], linewidth=3, ls='--', color='k', label='no phonons')
for i,B in enumerate(beta):
    plt.plot(bias_values, F2[i+1], linewidth=3, label=r'$\beta = ' + str(beta[i]) + '$')
plt.axhline(1, ls='--', color='grey')
plt.axvline(2)
plt.xlim(-10.2, 10.2)
plt.ylim(0.82, 1.25)
plt.xlabel(r'bias $(1 / T_c)$')
plt.ylabel(r'Fano factor')
plt.text(-9.7, 1.222, '(b)', fontsize=22)
plt.legend(fontsize=14).draggable()
plt.show()