import numpy as np
from HEOM_counting_statistics.dissipative_DQD_model import DissipativeDQDModel
from counting_statistics.fcs_solver import FCSSolver

Gamma_L = 1.
Gamma_R = 0.025
bias = 2.
T_c = 1. 
beta = 0.4
cutoff = 50.

def drude_spectral_density(reorg_energy, cutoff):
    def J(delta):
        return (2. * reorg_energy * cutoff * delta) / (delta**2 + cutoff**2)
    return J

model_pert = DissipativeDQDModel(Gamma_L, Gamma_R, bias, T_c, drude_spectral_density(0,cutoff), beta)

reorg_energy_values = np.logspace(-3, 15, 320)
F2_pert = np.zeros(reorg_energy_values.size)
coh_pert = np.zeros(reorg_energy_values.size)
mean_pert = np.zeros(reorg_energy_values.size)

for i,E in enumerate(reorg_energy_values):
    model_pert.spectral_density = drude_spectral_density(E, cutoff)
    solver_pert = FCSSolver(model_pert.liouvillian(), model_pert.jump_matrix(), np.array([1,1,1,0,0]))
    mean_pert[i] = solver_pert.mean()
    F2_pert[i] = solver_pert.second_order_fano_factor(0)
    coh_pert[i] = np.abs(solver_pert.ss[3] + solver_pert.ss[4])
    
np.savez('../data/F2_reorg_energy_perturbative_data_large_reorg_energy.npz', reorg_energy_values=reorg_energy_values, F2=F2_pert, \
                    mean=mean_pert, coh=coh_pert)
    
