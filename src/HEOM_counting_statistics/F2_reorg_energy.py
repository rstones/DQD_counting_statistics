'''
Created on 9 Feb 2017

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
#reorg_energy = 0.000147
cutoff = 5. # meV
K = 10

def environment(reorg_energy, beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]
    
def drude_spectral_density(reorg_energy, cutoff):
    def J(delta):
        return (2. * reorg_energy * cutoff * delta) / (delta**2 + cutoff**2)
    return J

model_heom = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], environment=environment(0.1, beta[0], K), \
                                K=K, tc=True, trunc_level=6)
model_pert = DissipativeDQDModel(Gamma_L, Gamma_R, bias, T_c, drude_spectral_density(0,cutoff), beta)

# model_heom.environment = environment(0.00147, beta[0], K)
# hm = model_heom.heom_matrix()
# model_heom.dv_pops = np.zeros(model_heom.system_dimension**2 * model_heom.heom_solver.num_dms)
# model_heom.dv_pops[:model_heom.system_dimension**2] = np.array([1., 0, 0, 0, 1., 0, 0, 0, 1.])
# print FCSSolver.stationary_state(hm, model_heom.dv_pops)[:9]

reorg_energy_values = np.logspace(0, 4, 5)
F2_heom = np.zeros((len(beta), reorg_energy_values.size))
F2_pert = np.zeros((len(beta), reorg_energy_values.size))
coh_heom = np.zeros((len(beta), reorg_energy_values.size))
coh_pert = np.zeros((len(beta), reorg_energy_values.size))
mean_heom = np.zeros((len(beta), reorg_energy_values.size))
mean_pert = np.zeros((len(beta), reorg_energy_values.size))

for j,B in enumerate(beta):
    print "calculating for beta = " + str(B)
    model_heom.beta = B
    model_pert.beta = B
    for i,E in enumerate(reorg_energy_values):
        print str(E) + ' at ' + str(tu.getTime())
        model_heom.environment = environment(E, B, K)
        try:
            solver = FCSSolver(model_heom.heom_matrix(), model_heom.jump_matrix(), model_heom.dv_pops)
            mean_heom[j,i] = solver.mean()
            F2_heom[j,i] = solver.second_order_fano_factor()
            coh_heom[j,i] = np.abs(model_heom.heom_solver.extract_system_density_matrix(solver.ss)[1,2])
        except ArpackNoConvergence:
            print "Convergence error!"
        model_pert.spectral_density = drude_spectral_density(E, cutoff)
        solver_pert = DenseFCSSolver(model_pert.liouvillian(), model_pert.jump_matrix(), np.array([1,1,1,0,0]))
        mean_pert[j,i] = solver_pert.mean()
        F2_pert[j,i] = solver_pert.second_order_fano_factor(0)
        coh_pert[j,i] = np.abs(solver_pert.ss[3] + solver_pert.ss[4])

# fname = '../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K7.npz'
# data = np.load(fname)
# reorg_energy_values = np.append(data['reorg_energy_values'], reorg_energy_values)
# F2_heom = np.append(data['F2_heom'], F2_heom)
# F2_pert = np.append(data['F2_pert'], F2_pert)
# coh_heom = np.append(data['coh_heom'], coh_heom)
# coh_pert = np.append(data['coh_pert'], coh_pert)
# mean_heom = np.append(data['mean_heom'], mean_heom)
# mean_pert = np.append(data['mean_pert'], mean_pert)

np.savez('../../data/HEOM_weak_coupling_F2_reorg_energy_drude_T2.7_N6_K10.npz', reorg_energy_values=reorg_energy_values, \
                    F2_heom=F2_heom.squeeze(), F2_pert=F2_pert.squeeze(), coh_heom=coh_heom.squeeze(), \
                    coh_pert=coh_pert.squeeze(), mean_heom=mean_heom.squeeze(), mean_pert=mean_pert.squeeze())
        
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

colours = ['r', 'b', 'g']

#plt.subplot(211)
for i,B in enumerate(beta):
    plt.subplot(131)
    plt.semilogx(reorg_energy_values, mean_heom[i], linewidth=3, ls='-', color=colours[i], label='T = ' + str(temperature[i]) + 'K')
    plt.semilogx(reorg_energy_values, mean_pert[i], linewidth=3, ls='--', color=colours[i])
    plt.subplot(132)
    plt.semilogx(reorg_energy_values, F2_heom[i], linewidth=3, ls='-', color=colours[i], label='T = ' + str(temperature[i]) + 'K')
    plt.semilogx(reorg_energy_values, F2_pert[i], linewidth=3, ls='--', color=colours[i])
    plt.subplot(133)
    plt.semilogx(reorg_energy_values, np.abs(coh_heom[i]), linewidth=3, ls='-', color=colours[i], label='T = ' + str(temperature[i]) + 'K')
    plt.semilogx(reorg_energy_values, np.abs(coh_pert[i]), linewidth=3, ls='--', color=colours[i])

plt.subplot(132)
plt.xlabel(r'$\lambda$ (meV)')
plt.ylabel('F2')
plt.legend().draggable()

plt.subplot(133)
plt.xlabel(r'$\lambda$ (meV)')
plt.ylabel('|coherence|')

# plt.subplot(212)
# for i,B in enumerate(beta):
#     plt.semilogx(reorg_energy_values, coh_heom[i], linewidth=3, ls='-', color=colours[i], label='T = ' + str(temperature[i]) + 'K')
#     plt.semilogx(reorg_energy_values, coh_pert[i], linewidth=3, ls='--', color=colours[i])

plt.show()
        
        
        
        
        