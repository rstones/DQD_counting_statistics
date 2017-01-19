import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

data = np.load('../../data/HEOM_F2_bias_drude_K_4.npz')
data_np = np.load('../../data/HEOM_F2_bias_drude_no_phonon.npz')

F2_np = data_np['F2']
F2 = data['F2']
bias_values = data['bias_values']
temperature = data['temperature']

plt.plot(bias_values, F2_np, linewidth=3, ls='--', color='k', label='no phonons')
for i in range(1,4):
    plt.plot(bias_values, F2[i], linewidth=3, label='T = ' + str(temperature[i-1]) + 'K')
plt.axhline(1, ls='--', color='grey')
plt.xlim(-1.05, 1.05)
plt.ylim(0.72, 1.25)
plt.xlabel(r'bias $\epsilon$ (meV)')
plt.ylabel(r'Fano factor')
plt.legend(fontsize=14).draggable()
plt.show()