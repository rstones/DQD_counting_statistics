import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt
import matplotlib as mpl
from prettyplotlib import brewer2mpl

font = {'size':12}
mpl.rc('font', **font)


bmap = brewer2mpl.get_map('Set2', 'qualitative', 6)
colors = bmap.mpl_colors[::-1]
mpl.rcParams['axes.color_cycle'] = colors

data = np.load('../../data/F2_reorg_energy_data_vals_compiled_sorted.npz')
reorg_energy_values = data['reorg_energy_values']
F2 = data['F2']
mean = data['mean']

data_K8 = np.load('../../remote_scripts/data/F2_reorg_energy_data_K8_vals_1-4.npz')
F2_K8  = data_K8['F2_heom']
reorg_energy_K8 = data_K8['reorg_energy_values']

data_K9 = np.load('../../remote_scripts/data/F2_reorg_energy_data_K9_vals_1-4.npz')
F2_K9  = data_K9['F2_heom']
reorg_energy_K9 = data_K9['reorg_energy_values']

pert_data = np.load('../../data/F2_reorg_energy_perturbative_data_large_reorg_energy.npz')
pert_reorg_energy_values = pert_data['reorg_energy_values']
pert_F2 = pert_data['F2']
pert_mean = pert_data['mean']

fig,ax = ppl.subplots(1, 2, figsize=(8,3.5))
lw = 2

plt.sca(ax[0])
ax[0].text(1e-3 + 0.0008, 0.0208, 'a.')
ppl.semilogx(pert_reorg_energy_values, pert_mean, linewidth=lw, label='WCA', show_ticks=True)
ppl.semilogx(reorg_energy_values, mean, linewidth=lw, label='NP', show_ticks=True)
ax[0].set_xlim(1e-3, 1e4)
ax[0].set_ylim(-0.0005, 0.0205)
ax[0].set_xlabel(r'$\lambda / \Gamma_L$')
ax[0].set_ylabel('current / e')
ppl.legend(fontsize=10).draggable()

plt.sca(ax[1])
ax[1].text(1e-3 + 0.0008, 1.42, 'b.')
ax[1].axhline(1, ls='--', color='grey')
ppl.semilogx(pert_reorg_energy_values, pert_F2, linewidth=lw, label='WCA', show_ticks=True)
ppl.semilogx(reorg_energy_values, F2, linewidth=lw, label='NP', show_ticks=True)
ax[1].set_xlabel(r'$\lambda / \Gamma_L$')
ax[1].set_ylabel('Fano factor')
ax[1].set_xlim(1.e-3, 1.e4)
ax[1].set_ylim(0.58, 1.42)
ppl.legend(loc='lower left', fontsize=10).draggable()

a = plt.axes([0.76, 0.7, 0.2, 0.26])

plt.sca(a)
ppl.semilogx(reorg_energy_K8, F2_K8, linewidth=2, label='K=8', show_ticks=True)
ppl.semilogx(reorg_energy_K9, F2_K9, linewidth=2, label='K=9', show_ticks=True)
ppl.semilogx(reorg_energy_values, F2, linewidth=2, label='K=10', show_ticks=True)

a.set_xlabel(r'$\lambda / \Gamma_L$', fontsize=9, labelpad=-3)
a.set_ylabel('Fano factor', fontsize=8, labelpad=-1)
plt.xticks([1e1, 1e2, 1e3, 1e4], fontsize=8)
plt.yticks([0.6, 0.8, 1.0, 1.2], fontsize=8)
a.set_xlim(1.e1, 1.e4)
a.set_ylim(0.6,1)
ppl.legend(fontsize=8).draggable()

plt.tight_layout()
plt.show()




