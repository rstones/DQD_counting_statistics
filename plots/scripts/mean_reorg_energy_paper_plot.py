import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

reorg_energy_values = []
mean_heom = []
mean_pert = []

data0 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K8_-6-0.npz')
reorg_energy_values = np.append(reorg_energy_values, data0['reorg_energy_values'])
mean_heom = np.append(mean_heom, data0['mean_heom'])
mean_pert = np.append(mean_pert, data0['mean_pert'])

data1 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K10_0-2.npz')
reorg_energy_values = np.append(reorg_energy_values, data1['reorg_energy_values'])
mean_heom = np.append(mean_heom, data1['mean_heom'])
mean_pert = np.append(mean_pert, data1['mean_pert'])

data2 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K10_2-4.npz')
reorg_energy_values = np.append(reorg_energy_values, data2['reorg_energy_values'])
mean_heom = np.append(mean_heom, data2['mean_heom'])
mean_pert = np.append(mean_pert, data2['mean_pert'])

data3 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K10_4-5.npz')
reorg_energy_values = np.append(reorg_energy_values, data3['reorg_energy_values'])
mean_heom = np.append(mean_heom, data3['mean_heom'])
mean_pert = np.append(mean_pert, data3['mean_pert'])

plt.figure(figsize=(8,7))
plt.semilogx(reorg_energy_values, mean_pert*1000, linewidth=3, ls='--', color='g', label='WCA')
plt.semilogx(reorg_energy_values, mean_heom*1000, linewidth=3, color='g', label='NP')
plt.xlim(1e-5, 1e4)
plt.ylim(0,2)
plt.xlabel(r'reorg energy $\lambda / T_c$')
plt.ylabel(r'mean ($\times10^{-3}$)')
plt.text(0.000015, 1.87, '(a)', fontsize=22)
plt.legend(fontsize=14).draggable()
plt.show()