import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

data = np.load('../../data/F2_reorg_energy_data_vals_compiled_sorted.npz')
reorg_energy_values = data['reorg_energy_values']
F2 = data['F2']

data_K8 = np.load('../../remote_scripts/data/F2_reorg_energy_data_K8_vals_1-4.npz')
F2_K8  = data_K8['F2_heom']
reorg_energy_K8 = data_K8['reorg_energy_values']

data_K9 = np.load('../../remote_scripts/data/F2_reorg_energy_data_K9_vals_1-4.npz')
F2_K9  = data_K9['F2_heom']
reorg_energy_K9 = data_K9['reorg_energy_values']

pert_data = np.load('../../data/F2_reorg_energy_perturbative_data_large_reorg_energy.npz')
pert_reorg_energy_values = pert_data['reorg_energy_values']
pert_F2 = pert_data['F2']


plt.figure(figsize=(8,7))

plt.text(0.0015, 1.34, '(b)', fontsize=22)
plt.semilogx(pert_reorg_energy_values, pert_F2, linewidth=3, color='g', ls='--', label='WCA')
plt.semilogx(reorg_energy_values, F2, linewidth=3, color='g', label='NP')
plt.xlim(1.e-3, 1.e10)
plt.ylim(0.6, 1.4)
plt.ylabel('Fano factor')
plt.xlabel(r'reorg energy $(1 / T_c)$')
plt.legend(fontsize=14).draggable()
plt.axvline(0.015)

a = plt.axes([0.54, 0.58, 0.32, 0.28])
plt.semilogx(reorg_energy_K8, F2_K8, linewidth=3, color='r', label='K=8')
plt.semilogx(reorg_energy_K9, F2_K9, linewidth=3, color='b', label='K=9')
plt.semilogx(reorg_energy_values, F2, linewidth=3, color='g', label='K=10')

plt.xlabel(r'reorg energy $(1 / T_c)$', fontsize=12)
plt.ylabel('Fano factor', fontsize=12)

plt.xticks([1e1, 1e2, 1e3, 1e4], fontsize=12)
plt.yticks([0.6, 0.8, 1.0, 1.2], fontsize=12)
plt.xlim(1.e1, 1.e4)
plt.ylim(0.6,1)

plt.legend(fontsize=12).draggable()

plt.show()