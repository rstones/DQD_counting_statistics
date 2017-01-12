'''
Created on 23 Dec 2016

@author: richard
'''
import numpy as np
from HEOM_counting_statistics.dissipative_DQD_model import DissipativeDQDModel
from HEOM_counting_statistics.DQD_HEOM_model import DQDHEOMModel
from counting_statistics.fcs_solver import FCSSolver

Gamma_L = 1.
Gamma_R = 1.e-4
bias = 0
T_c = 3.

temperature = 300.
beta = 0.610 # get beta from photon population at certain freq
reorg_energy_values = np.linspace(1.e-9,100,100)
cutoff_freq = 1.

def drude_spectral_density(reorg_energy):
    
    def J(omega):
        return 2.*omega*reorg_energy*cutoff_freq / (omega**2 + cutoff_freq**2)
    
    return J

diss_model = DissipativeDQDModel(Gamma_L, Gamma_R, bias, T_c, drude_spectral_density(1.e-9), beta)
heom_model = DQDHEOMModel(bias, T_c, Gamma_L, Gamma_R, 1.e-9, cutoff_freq, beta)

diss_F2 = np.zeros(reorg_energy_values.size)
heom_F2 = np.zeros(reorg_energy_values.size)

for i,Er in enumerate(reorg_energy_values):
    diss_model.spectral_density = drude_spectral_density(reorg_energy_values[i])
    heom_model.drude_reorg_energy = reorg_energy_values[i]
    diss_solver = FCSSolver(diss_model.liouvillian(), diss_model.jump_matrix(), np.array([1.,1.,1.,0,0]))
    heom_solver = FCSSolver(heom_model.heom_matrix(), heom_model.jump_matrix(), heom_model.dv_pops)
    diss_F2[i] = diss_solver.second_order_fano_factor(0)
    heom_F2[i] = heom_solver.second_order_fano_factor(0)
    
np.savez('../../data/DQD_counting_stats_F2_coupling_strength.npz', reorg_energy_values=reorg_energy_values, diss_F2=diss_F2, heom_F2=heom_F2)

import matplotlib.pyplot as plt
plt.plot(reorg_energy_values, diss_F2, label='diss')
plt.plot(reorg_energy_values, heom_F2, label='heom')
plt.legend().draggable()
plt.show()
    