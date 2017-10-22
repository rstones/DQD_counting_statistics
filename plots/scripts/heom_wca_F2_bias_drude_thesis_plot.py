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

wca_data = np.load('../../data/DQD_dissipative_F2_bias_brandes_liouvillian.npz')

wca_bias_values = wca_data['bias_values']
wca_beta_values = wca_data['beta_values']
wca_F2 = wca_data['F2']
wca_site_steady_states = wca_data['site_steady_states']

new_site_steady_states = np.zeros((len(beta_values)+1, bias_values.size, 3, 3), dtype='complex128')
exciton_steady_states = np.zeros(new_site_steady_states.shape, dtype='complex128')

wca_new_site_steady_states = np.zeros((len(beta_values)+1, wca_bias_values.size, 3, 3), dtype='complex128')
wca_exciton_steady_states = np.zeros(wca_new_site_steady_states.shape, dtype='complex128')

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
        #rearranged_basis_transform = rearrange_transform(transform)
        ss = site_steady_states[j+1,i]
        ss.shape = 3,3
        new_site_steady_states[j+1,i] = ss
        exciton_steady_states[j+1,i] = np.dot(rearranged_basis_transform.T, np.dot(ss, rearranged_basis_transform))
        
        
for i,E in enumerate(wca_bias_values):
    H = system_hamiltonian(E, T_c)
    transform = np.linalg.eig(H)[1]
    
    rearranged_basis_transform = rearrange_transform(transform)
    
    wca_ss = wca_site_steady_states[0,i]
    new_ss = np.zeros((3,3), dtype='complex128')
    new_ss[0,0] = wca_ss[0]
    new_ss[1,1] = wca_ss[1]
    new_ss[2,2] = wca_ss[2]
    new_ss[1,2] = wca_ss[3] + wca_ss[4]
    new_ss[2,1] = wca_ss[3] - wca_ss[4]
    wca_new_site_steady_states[0,i] = new_ss
    wca_exciton_steady_states[0,i] = np.dot(rearranged_basis_transform.T, np.dot(new_ss, rearranged_basis_transform))
    
    for j,B in enumerate(beta_values):        
        wca_ss = wca_site_steady_states[j+1,i]
        new_ss = np.zeros((3,3), dtype='complex128')
        new_ss[0,0] = wca_ss[0]
        new_ss[1,1] = wca_ss[1]
        new_ss[2,2] = wca_ss[2]
        new_ss[1,2] = wca_ss[3] + wca_ss[4]
        new_ss[2,1] = wca_ss[3] - wca_ss[4]
        wca_new_site_steady_states[j+1,i] = new_ss
        wca_exciton_steady_states[j+1,i] = np.dot(rearranged_basis_transform.T, np.dot(new_ss, rearranged_basis_transform))

fig,ax = ppl.subplots(2,2, figsize=(8,7))
lw = 2

plt.sca(ax[0,0])
plt.text(-10., 1.24, 'a.')
plt.axhline(1, ls='--', color='grey')
ppl.plot(wca_bias_values, wca_F2[0], linewidth=lw, ls='--', color='k', show_ticks=True)
for i in range(1,4):
    ppl.plot(wca_bias_values, wca_F2[i], linewidth=lw, label=r'$\beta = '+str(beta_values[i-1])+'$', show_ticks=True)
ppl.legend(fontsize=10).draggable()
ax[0,0].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[0,0].set_ylabel(r'Fano factor')
ax[0,0].set_ylim(0.89, 1.24)
ax[0,0].set_yticks([0.9, 1.0, 1.1, 1.2])

plt.sca(ax[0,1])
plt.text(-10., 1.24, 'b.')
plt.axhline(1, ls='--', color='grey')
ppl.plot(bias_values, F2[0], linewidth=lw, ls='--', color='k', show_ticks=True)
for i in range(1,4):
    ppl.plot(bias_values, F2[i], linewidth=lw, label=r'$\beta = '+str(beta_values[i-1])+'$', show_ticks=True)
ppl.legend(fontsize=10).draggable()
ax[0,1].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[0,1].set_ylabel(r'Fano factor')
ax[0,1].set_ylim(0.89, 1.24)
ax[0,1].set_yticks([0.9, 1.0, 1.1, 1.2])


plt.sca(ax[1,0])
plt.text(-10., 0.36, 'c.')
#pop_labels = [r'$\rho_{00}$', r'$\rho_{LL}$', r'$\rho_{RR}$']
ppl.plot(wca_bias_values, np.abs(wca_new_site_steady_states[0,:,1,2]), linewidth=lw, ls='--', color='k', show_ticks=True)
for i in range(1,4):
    ppl.plot(wca_bias_values, np.abs(wca_new_site_steady_states[i,:,1,2]), linewidth=lw, label=r'$\beta = '+str(beta_values[i-1])+'$', show_ticks=True)
ppl.legend(fontsize=10).draggable()
ax[1,0].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[1,0].set_ylabel(r'coherence $|\rho_{LR}|$')
ax[1,0].set_ylim(-0.02, 0.36)

plt.sca(ax[1,1])
plt.text(-10., 0.36, 'd.')
#ex_pop_labels = [r'$\rho_{00}$', r'$\rho_{++}$', r'$\rho_{--}$']
ppl.plot(bias_values, np.abs(new_site_steady_states[0,:,1,2]), linewidth=lw, ls='--', color='k', show_ticks=True)
for i in range(1,4):
    ppl.plot(bias_values, np.abs(new_site_steady_states[i,:,1,2]), linewidth=lw, label=r'$\beta = '+str(beta_values[i-1])+'$', show_ticks=True)
ppl.legend(fontsize=10).draggable()
ax[1,1].set_xlabel(r'$\epsilon / \Gamma_L$')
ax[1,1].set_ylabel(r'coherence $|\rho_{LR}|$')
ax[1,1].set_ylim(-0.02, 0.36)

plt.tight_layout()
plt.show()



