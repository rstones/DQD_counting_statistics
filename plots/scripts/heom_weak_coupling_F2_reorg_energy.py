import numpy as np
import matplotlib.pyplot as plt

data = np.load('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_N7_K4.npz')
reorg_energy_values = data['reorg_energy_values']
F2_heom = data['F2_heom']
F2_pert = data['F2_pert']
mean_heom = data['mean_heom']
mean_pert = data['mean_pert']
coh_heom = data['coh_heom']
coh_pert = data['coh_pert']

for i in range(4):
    plt.subplot(131)
    plt.semilogx(reorg_energy_values, mean_heom[i])
    plt.semilogx(reorg_energy_values, mean_pert[i], ls='--')
    plt.subplot(132)
    plt.semilogx(reorg_energy_values, F2_heom[i])
    plt.semilogx(reorg_energy_values, F2_pert[i], ls='--')
    plt.subplot(133)
    plt.semilogx(reorg_energy_values, coh_heom[i])
    plt.semilogx(reorg_energy_values, coh_pert[i], ls='--')
    
plt.show()
    
    
    