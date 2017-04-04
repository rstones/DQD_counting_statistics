import numpy as np

data1 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7K_-6_0.npz')
data2 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7K_0_4_N8_K3.npz')

ss_data = np.load('../../data/HEOM_time_propagate_to_steady_state_test.npz')
ss_reorg_energies = ss_data['reorg_energy_values']
ss_steady_states = ss_data['steady_states']
mean = np.zeros(ss_reorg_energies.size)
for i in range(3):
    ss = ss_steady_states[i][:81]
    ss.shape = 9,9
    mean[i] = 2.5e-3*ss[8,8]

current_heom = np.append(data1['mean_heom'], data2['mean_heom'])
current_pert = np.append(data1['mean_pert'], data2['mean_pert'])
F2_heom = np.append(data1['F2_heom'], data2['F2_heom'])
F2_pert = np.append(data1['F2_pert'], data2['F2_pert'])
coh_heom = np.append(data1['coh_heom'], data2['coh_heom'])
coh_pert = np.append(data1['coh_pert'], data2['coh_pert'])
reorg_energy_values = np.append(data1['reorg_energy_values'], data2['reorg_energy_values'])

import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

plt.subplot(131)
plt.semilogx(reorg_energy_values, current_heom, color='b', linewidth=2, label='heom')
plt.semilogx(reorg_energy_values, current_pert, color='b',ls='--', linewidth=2, label='weak coupling')
#plt.plot(ss_reorg_energies, mean, linestyle='None', color='r', marker='o', markersize=6, label=r'heom $\rho(t\rightarrow\infty)$')
plt.legend(fontsize=14).draggable()
plt.xlabel(r'reorg energy $\lambda$')
plt.ylabel('current')
plt.ylim(0,0.002)

plt.subplot(132)
plt.semilogx(reorg_energy_values, F2_heom, color='b',linewidth=2)
plt.semilogx(reorg_energy_values, F2_pert, ls='--', color='b',linewidth=2)
plt.xlabel(r'reorg energy $\lambda$')
plt.ylabel('F2')
plt.ylim(-1,3)

plt.subplot(133)
plt.semilogx(reorg_energy_values, coh_heom, color='b',linewidth=2)
plt.semilogx(reorg_energy_values, coh_pert, ls='--', color='b',linewidth=2)
plt.xlabel(r'reorg energy $\lambda$')
plt.ylabel('|coherence|')

plt.show()