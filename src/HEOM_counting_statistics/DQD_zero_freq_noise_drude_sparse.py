'''
Created on 19 Jan 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
from counting_statistics.sparse.fcs_solver import FCSSolver
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.OBOscillator import OBOscillator

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0
T_c = 0.1 # meV
temperature = [1.4, 2.7, 18.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature]
reorg_energy = 0.00147
cutoff = 5. # meV
K = 0

def environment(beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]

model = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], environment=environment(beta[0], K), K=K, tc=True)
bias_values = np.linspace(-1, 1, 100)
F2 = np.zeros((len(beta)+1, bias_values.size))

for j,B in enumerate(beta):
    print 'for beta = ' + str(B)
    model.beta = B
    model.environment = environment(B, K)
    for i,E in enumerate(bias_values):
        #print E
        model.bias = E
        solver = FCSSolver(model.heom_matrix(), model.jump_matrix(), model.dv_pops)
        try:
            F2[j+1,i] = solver.mean() #solver.second_order_fano_factor()
        except RuntimeError:
            print "SINGULAR ERROR!!!!!!!"
        
model.drude_reorg_energy = 1.e-9
print 'for no phonons'
for i,E in enumerate(bias_values):
    #print E
    model.bias = E
    solver = FCSSolver(model.heom_matrix(), model.jump_matrix(), model.dv_pops)
    try:
        F2[0,i] = solver.mean() #solver.second_order_fano_factor()
    except RuntimeError:
        print "SINGULAR ERROR!!!!!!!"
    
#np.savez('../../data/HEOM_F2_bias_drude_no_phonon.npz', bias_values=bias_values, F2=F2[0])
    
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

plt.plot(bias_values, F2[0], linewidth=3, ls='--', color='k', label='no phonons')
for i,B in enumerate(beta):
    plt.plot(bias_values, F2[i+1], linewidth=3, label='T = ' + str(temperature[i]) + 'K')
plt.axhline(1, ls='--', color='grey')
plt.xlim(-1.05, 1.05)
plt.ylim(0.72, 1.25)
plt.xlabel(r'bias $\epsilon$ (meV)')
plt.ylabel(r'Fano factor')
plt.legend(fontsize=14).draggable()
plt.show()