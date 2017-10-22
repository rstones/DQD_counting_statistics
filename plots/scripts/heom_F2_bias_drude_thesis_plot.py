import numpy as np

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import matplotlib as mpl
from prettyplotlib import brewer2mpl

font = {'size':12}
mpl.rc('font', **font)

data = np.load('../../data/HEOM_F2_bias_drude_data_inc_steady_states.npz')

bias_values = data['bias_values']
beta_values = data['beta']
mean = data['mean']
F2 = data['F2']
site_steady_states = data['site_steady_states']

new_site_steady_states = np.zeros((len(beta_values)+1, bias_values.size, 3, 3), dtype='complex128')
exciton_steady_states = np.zeros(new_site_steady_states.shape, dtype='complex128')
T_c = 1.
def system_hamiltonian(bias, T_c):
    return np.array([[0, 0, 0],
                     [0, bias/2., T_c],
                     [0, T_c, -bias/2.]])
    
def rearrange_transform(transform):
    rearranged_basis_transform = np.zeros(transform.shape)
    for basis_vec in transform.T:
        if np.count_nonzero(basis_vec) == 1:
            rearranged_basis_transform[:,0] = basis_vec
        elif np.all(basis_vec >= 0) or np.all(basis_vec <= 0):
            rearranged_basis_transform[:,1] = basis_vec
        else:
            rearranged_basis_transform[:,2] = basis_vec
    return rearranged_basis_transform

for i,E in enumerate(bias_values):
    H = system_hamiltonian(E, T_c)
    transform = np.linalg.eig(H)[1]
    
    rearranged_basis_transform = rearrange_transform(transform)
    ss = site_steady_states[0,i]
    ss.shape = 3,3
    new_site_steady_states[0,i] = ss
    exciton_steady_states[0,i] = np.dot(rearranged_basis_transform.T, np.dot(ss, rearranged_basis_transform))
    
    for j,B in enumerate(beta_values):
        rearranged_basis_transform = rearrange_transform(transform)
        ss = site_steady_states[j+1,i]
        ss.shape = 3,3
        new_site_steady_states[j+1,i] = ss
        exciton_steady_states[j+1,i] = np.dot(rearranged_basis_transform.T, np.dot(ss, rearranged_basis_transform))

fig,ax = ppl.subplots(2,2, figsize=(8,7))
lw = 2

plt.sca(ax[0,0])
plt.axhline(1, ls='--', color='grey')
ppl.plot(bias_values, F2[0], linewidth=lw, ls='--', color='k', show_ticks=True)
for i in range(1,4):
    ppl.plot(bias_values, F2[i], linewidth=lw, label=r'$\beta = '+str(beta_values[i-1])+'$', show_ticks=True)
ppl.legend(fontsize=10).draggable()
ax[0,0].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[0,0].set_ylabel(r'Fano factor')

plt.sca(ax[0,1])
ppl.plot(bias_values, np.abs(site_steady_states[0,:,5]), linewidth=lw, ls='--', color='k', show_ticks=True)
for i in range(1,4):
    ppl.plot(bias_values, np.abs(site_steady_states[i,:,5]), linewidth=lw, label=r'$\beta = '+str(beta_values[i-1])+'$', show_ticks=True)
ppl.legend(fontsize=10).draggable()
ax[0,1].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[0,1].set_ylabel(r'coherence $|\rho_{LR}|$')

plt.sca(ax[1,0])
pop_labels = [r'$\rho_{00}$', r'$\rho_{LL}$', r'$\rho_{RR}$']

# for i in range(3):
#     ppl.plot(bias_values, new_site_steady_states[0,:,i,i], linewidth=1, ls='--', label=pop_labels[i], show_ticks=True)
# 
# for i in range(3):
#     ppl.plot(bias_values, new_site_steady_states[1,:,i,i], linewidth=lw, label=pop_labels[i], show_ticks=True)
    
for i in range(3):
    ppl.plot(bias_values, new_site_steady_states[3,:,i,i], linewidth=lw, label=pop_labels[i], show_ticks=True)
ppl.legend(fontsize=10).draggable()
ax[1,0].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[1,0].set_ylabel(r'populations')

plt.sca(ax[1,1])
ex_pop_labels = [r'$\rho_{00}$', r'$\rho_{++}$', r'$\rho_{--}$']
# for i in range(3):
#     ppl.plot(bias_values, exciton_steady_states[3,:,i,i], linewidth=lw, label=pop_labels[i], show_ticks=True)

ppl.plot(bias_values, np.abs(exciton_steady_states[0,:,1,2]), linewidth=lw, ls='--', color='k', show_ticks=True)
for i in range(1,4):
    ppl.plot(bias_values, np.abs(exciton_steady_states[i,:,1,2]), linewidth=lw, label=r'$\beta = '+str(beta_values[i-1])+'$', show_ticks=True)
ppl.legend(fontsize=10).draggable()
ax[1,1].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[1,1].set_ylabel(r'populations')

plt.tight_layout()
plt.show()



