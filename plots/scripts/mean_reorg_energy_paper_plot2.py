import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

data = np.load('../../data/F2_reorg_energy_data_vals_compiled_sorted.npz')
reorg_energy_values = data['reorg_energy_values']
mean = data['mean']

pert_data = np.load('../../data/F2_reorg_energy_perturbative_data.npz')
pert_reorg_energy_values = pert_data['reorg_energy_values']
pert_mean = pert_data['mean']

plt.figure(figsize=(8,7))
plt.text(0.0015, 1.87, '(a)', fontsize=22)
plt.semilogx(pert_reorg_energy_values, pert_mean*100, linewidth=3, color='g', ls='--', label='WCA')
plt.semilogx(reorg_energy_values, mean*100, linewidth=3, color='g', label='NP')
plt.xlim(1e-3, 1e4)
plt.ylim(0,2)
plt.xlabel(r'reorg energy $\lambda / T_c$')
plt.ylabel(r'mean ($\times10^{-2}$)')
plt.legend(fontsize=14).draggable()

plt.show()