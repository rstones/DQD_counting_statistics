import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

reorg_energy_values = []
F2_heom = []
F2_pert = []

data0 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K8_-6-0.npz')
reorg_energy_values = np.append(reorg_energy_values, data0['reorg_energy_values'])
F2_heom = np.append(F2_heom, data0['F2_heom'])
F2_pert = np.append(F2_pert, data0['F2_pert'])

data1 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K10_0-2.npz')
reorg_energy_values = np.append(reorg_energy_values, data1['reorg_energy_values'])
F2_heom = np.append(F2_heom, data1['F2_heom'])
F2_pert = np.append(F2_pert, data1['F2_pert'])

data2 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K10_2-4.npz')
reorg_energy_values = np.append(reorg_energy_values, data2['reorg_energy_values'])
F2_heom = np.append(F2_heom, data2['F2_heom'])
F2_pert = np.append(F2_pert, data2['F2_pert'])

data3 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K10_4-5.npz')
reorg_energy_values = np.append(reorg_energy_values, data3['reorg_energy_values'])
F2_heom = np.append(F2_heom, data3['F2_heom'])
F2_pert = np.append(F2_pert, data3['F2_pert'])

plt.figure(figsize=(8,7))
plt.semilogx(reorg_energy_values, F2_pert, linewidth=3, ls='--', color='g', label='WCA')
plt.semilogx(reorg_energy_values, F2_heom, linewidth=3, color='g', label='NP')
plt.xlim(1e-5, 1e4)
plt.ylim(0.5, 1.4)
plt.xlabel(r'reorg energy $\lambda / T_c$')
plt.ylabel('Fano factor')
plt.text(0.000015, 1.34, '(b)', fontsize=22)
plt.legend(fontsize=14).draggable()
plt.show()