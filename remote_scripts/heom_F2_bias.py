import numpy as np
import scipy.constants as constants
from HEOM_counting_statistics.DQD_HEOM_model import DQDHEOMModel
from counting_statistics.fcs_solver import FCSSolver
import quant_mech.time_utils as tu

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature]
reorg_energy = 0.00147
cutoff = 5. # meV

model = DQDHEOMModel(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], drude_reorg_energy=reorg_energy, drude_cutoff=cutoff, \
                     num_matsubara_freqs=0, temperature_correction=True, sites_to_couple=np.array([0,1,1]))
bias_values = np.linspace(-1, 1, 100)
F2 = np.zeros((len(beta)+1, bias_values.size))

for j,B in enumerate(beta):
    print 'for beta = ' + str(B) + ' at ' + str(tu.getTime())
    model.beta = B
    for i,E in enumerate(bias_values):
        model.bias = E
        solver = FCSSolver(model.heom_matrix(), model.jump_matrix(), model.dv_pops)
        F2[j+1,i] = solver.second_order_fano_factor(0)
    
print 'Saving...'
np.savez('../data/HEOM_F2_bias_drude_K_'+str(model.num_matsubara_freqs)+'.npz', F2=F2, bias_values=bias_values, beta=beta, temperature=temperature)

