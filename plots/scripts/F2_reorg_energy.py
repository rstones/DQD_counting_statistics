import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

dataK6 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K6.npz')#_small_omegac.npz')
reorg_energy_valuesK6 = dataK6['reorg_energy_values']
F2_heomK6 = dataK6['F2_heom']
mean_heomK6 = dataK6['mean_heom']
coh_heomK6 = dataK6['coh_heom']
F2_pert = dataK6['F2_pert']
mean_pert = dataK6['mean_pert']
coh_pert = dataK6['coh_pert']

dataK7 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K7.npz')#_small_omegac.npz')
reorg_energy_valuesK7 = dataK7['reorg_energy_values']
F2_heomK7 = dataK7['F2_heom']
mean_heomK7 = dataK7['mean_heom']
coh_heomK7 = dataK7['coh_heom']
# F2_pertK7 = dataK7['F2_pert']
# mean_pertK7 = dataK7['mean_pert']
# coh_pertK7 = dataK7['coh_pert']

dataK8 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K8.npz')#_small_omegac.npz')
reorg_energy_valuesK8 = dataK8['reorg_energy_values']
F2_heomK8 = dataK8['F2_heom']
mean_heomK8 = dataK8['mean_heom']
coh_heomK8 = dataK8['coh_heom']

dataK9 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K9.npz')#_small_omegac.npz')
reorg_energy_valuesK9 = dataK9['reorg_energy_values']
F2_heomK9 = dataK9['F2_heom']
mean_heomK9 = dataK9['mean_heom']
coh_heomK9 = dataK9['coh_heom']

dataK10 = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K10.npz')#_small_omegac.npz')
reorg_energy_valuesK10 = dataK10['reorg_energy_values']
F2_heomK10 = dataK10['F2_heom']
mean_heomK10 = dataK10['mean_heom']
coh_heomK10 = dataK10['coh_heom']

plt.subplot(131)
plt.semilogx(reorg_energy_valuesK6, mean_pert, ls='--', color='k', linewidth=3)
plt.semilogx(reorg_energy_valuesK6, mean_heomK6, color='r', linewidth=3, label='K=6')
plt.semilogx(reorg_energy_valuesK7, mean_heomK7, color='b', linewidth=3, label='K=7')
plt.semilogx(reorg_energy_valuesK8, mean_heomK8, color='c', linewidth=3, label='K=8')
plt.semilogx(reorg_energy_valuesK9, mean_heomK9, color='m', marker='o', label='K=9')
plt.semilogx(reorg_energy_valuesK10, mean_heomK10, color='g', marker='o', label='K=10')
plt.ylim(0,0.003)
plt.xlabel(r'reorg energy $\lambda$')
plt.ylabel(r'mean')
plt.title(r'$\Omega_c = 5, N=6$', fontsize=18)
plt.legend().draggable()

plt.subplot(132)
plt.semilogx(reorg_energy_valuesK6, F2_pert, ls='--', color='k', linewidth=3)
plt.semilogx(reorg_energy_valuesK6, F2_heomK6, color='r', linewidth=3)
plt.semilogx(reorg_energy_valuesK7, F2_heomK7, color='b', linewidth=3)
plt.semilogx(reorg_energy_valuesK8, F2_heomK8, color='c', linewidth=3)
plt.semilogx(reorg_energy_valuesK9, F2_heomK9, color='m', marker='o')
plt.semilogx(reorg_energy_valuesK10, F2_heomK10, color='g', marker='o')
plt.ylim(0.4,1.5)
plt.xlabel(r'reorg energy $\lambda$')
plt.ylabel(r'F2')

plt.subplot(133)
plt.semilogx(reorg_energy_valuesK6, coh_pert, ls='--', color='k', linewidth=3)
plt.semilogx(reorg_energy_valuesK6, coh_heomK6, color='r', linewidth=3)
plt.semilogx(reorg_energy_valuesK7, coh_heomK7, color='b', linewidth=3)
plt.semilogx(reorg_energy_valuesK8, coh_heomK8, color='c', linewidth=3)
plt.semilogx(reorg_energy_valuesK9, coh_heomK9, color='m', marker='o')
plt.semilogx(reorg_energy_valuesK10, coh_heomK10, color='g', marker='o')
plt.ylim(0,0.3)
plt.xlabel(r'reorg energy $\lambda$')
plt.ylabel(r'|coherence|')

plt.show()