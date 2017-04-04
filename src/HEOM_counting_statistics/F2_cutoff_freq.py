'''
Created on 22 Mar 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
from counting_statistics.sparse.fcs_solver import FCSSolver
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.OBOscillator import OBOscillator
import quant_mech.time_utils as tu
from HEOM_counting_statistics.dissipative_DQD_model import DissipativeDQDModel
from counting_statistics.fcs_solver import FCSSolver as DenseFCSSolver
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0.2
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature][1:2]
print beta
reorg_energy = 1. #0.00147
#cutoff = 5. # meV
K = 4

def environment(cutoff, beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]
    
def drude_spectral_density(reorg_energy, cutoff):
    def J(delta):
        return (2. * reorg_energy * cutoff * delta) / (delta**2 + cutoff**2)
    return J

model_heom = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], environment=environment(1., beta[0], K), \
                                K=K, tc=True, trunc_level=7)
model_pert = DissipativeDQDModel(Gamma_L, Gamma_R, bias, T_c, drude_spectral_density(1.,1.), beta)

cutoff_values = np.linspace(0.2, 5., 50)
F2_heom = np.zeros((len(beta), cutoff_values.size))
F2_pert = np.zeros((len(beta), cutoff_values.size))
coh_heom = np.zeros((len(beta), cutoff_values.size))
coh_pert = np.zeros((len(beta), cutoff_values.size))
mean_heom = np.zeros((len(beta), cutoff_values.size))
mean_pert = np.zeros((len(beta), cutoff_values.size))

for j,B in enumerate(beta):
    print "calculating for beta = " + str(B)
    model_heom.beta = B
    model_pert.beta = B
    for i,wc in enumerate(cutoff_values):
        print wc
        model_heom.environment = environment(wc, B, K)
        try:
            solver = FCSSolver(model_heom.heom_matrix(), model_heom.jump_matrix(), model_heom.dv_pops)
            mean_heom[j,i] = solver.mean()
            F2_heom[j,i] = solver.second_order_fano_factor()
            coh_heom[j,i] = np.abs(model_heom.heom_solver.extract_system_density_matrix(solver.ss)[1,2])
        except ArpackNoConvergence:
            print "Convergence error!"
        model_pert.spectral_density = drude_spectral_density(reorg_energy, wc)
        solver_pert = DenseFCSSolver(model_pert.liouvillian(), model_pert.jump_matrix(), np.array([1,1,1,0,0]))
        mean_pert[j,i] = solver_pert.mean()
        F2_pert[j,i] = solver_pert.second_order_fano_factor(0)
        coh_pert[j,i] = np.abs(solver_pert.ss[3] + solver_pert.ss[4])
        
np.savez('../../data/HEOM_weak_coupling_F2_cutoff_drude_T2.7K_N7_K4_lambda_1.npz', cutoff_values=cutoff_values, \
                    F2_heom=F2_heom, F2_pert=F2_pert, coh_heom=coh_heom, coh_pert=coh_pert, \
                    mean_heom=mean_heom, mean_pert=mean_pert)
        
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

colours = ['r', 'b', 'g']

#plt.subplot(211)
for i,B in enumerate(beta):
    plt.subplot(131)
    plt.plot(cutoff_values, mean_heom[i], linewidth=3, ls='-', color=colours[i], label='T = ' + str(temperature[i]) + 'K')
    plt.plot(cutoff_values, mean_pert[i], linewidth=3, ls='--', color=colours[i])
    plt.subplot(132)
    plt.plot(cutoff_values, F2_heom[i], linewidth=3, ls='-', color=colours[i], label='T = ' + str(temperature[i]) + 'K')
    plt.plot(cutoff_values, F2_pert[i], linewidth=3, ls='--', color=colours[i])
    plt.subplot(133)
    plt.plot(cutoff_values, np.abs(coh_heom[i]), linewidth=3, ls='-', color=colours[i], label='T = ' + str(temperature[i]) + 'K')
    plt.plot(cutoff_values, np.abs(coh_pert[i]), linewidth=3, ls='--', color=colours[i])

plt.subplot(132)
plt.xlabel(r'$\Omega_c$ (meV)')
plt.ylabel('F2')
plt.legend().draggable()

plt.subplot(133)
plt.xlabel(r'$\Omega_c$ (meV)')
plt.ylabel('|coherence|')

# plt.subplot(212)
# for i,B in enumerate(beta):
#     plt.semilogx(reorg_energy_values, coh_heom[i], linewidth=3, ls='-', color=colours[i], label='T = ' + str(temperature[i]) + 'K')
#     plt.semilogx(reorg_energy_values, coh_pert[i], linewidth=3, ls='--', color=colours[i])

plt.show()
        
        
        
        
