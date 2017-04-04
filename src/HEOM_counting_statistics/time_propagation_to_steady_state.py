'''
Created on 14 Mar 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
from counting_statistics.sparse.fcs_solver import FCSSolver
from HEOM_counting_statistics.DQD_HEOM_model_sparse import DQDHEOMModelSparse
from quant_mech.OBOscillator import OBOscillator
import quant_mech.time_utils as tu
from scipy.integrate import complex_ode
import numpy.testing as npt

Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meVfigure_1
bias = 0.2
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature][1:2]
#reorg_energy = 0.000147
cutoff = 5. # meV
K = 4

def environment(reorg_energy, beta, K):
    return [(), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),), \
            (OBOscillator(reorg_energy, cutoff, beta, K=K),)]
    
def drude_spectral_density(reorg_energy, cutoff):
    def J(delta):
        return (2. * reorg_energy * cutoff * delta) / (delta**2 + cutoff**2)
    return J

model_heom = DQDHEOMModelSparse(Gamma_L, Gamma_R, bias, T_c, beta=beta[0], environment=environment(40., beta[0], K), \
                                K=K, tc=True, trunc_level=7)

print 'calculating steady state...'
steady_state = FCSSolver.stationary_state(model_heom.heom_matrix(), model_heom.dv_pops).squeeze()
steady_state = steady_state[:9]
steady_state.shape = 3,3
print steady_state

print 'calculating time evolution...'
init_state = np.array([[0,0,0],[0,1.,0],[0,0,0]])
diff = 1
while np.any(diff > 1.5e-2):
    model_heom.heom_solver.init_system_dm = init_state
    dm_history,time = model_heom.heom_solver.calculate_time_evolution(0.05, 50.)
    init_state = dm_history[-1]
    diff = np.abs(steady_state - init_state)
    print diff

print time.size
print dm_history.shape
if time.size != dm_history.shape[0]:
    dm_history = dm_history[:-1]

import matplotlib.pyplot as plt
plt.axhline(steady_state[0,0], ls='--', color='k')
plt.plot(time, [dm[0,0] for dm in dm_history], linewidth=2, color='k', label='0')
plt.axhline(steady_state[1,1], ls='--', color='b')
plt.plot(time, [dm[1,1] for dm in dm_history], linewidth=2, color='b', label='L')
plt.axhline(steady_state[2,2], ls='--', color='r')
plt.plot(time, [dm[2,2] for dm in dm_history], linewidth=2, color='r', label='R')

plt.ylim(0,1)
plt.legend().draggable()
plt.show()

# reorg_energy_values = np.logspace(0,2,3)
# 
# print 'calculating stationary state at ' + str(tu.getTime())
# init_state = FCSSolver.stationary_state(model_heom.heom_matrix(), model_heom.dv_pops).squeeze()
# 
# steady_states = np.zeros((3, init_state.size))
# 
# for i,E in enumerate(reorg_energy_values):
#     print 'Starting calculation for reorg energy ' + str(E) + ' at ' + str(tu.getTime())
#     model_heom.environment = environment(E, beta[0], K)
#     hierarchy_matrix = model_heom.heom_matrix()
#     t0 = 0
#     time_step = 0.001 # try smaller time steps
#         
#     def f(t, rho):
#         return hierarchy_matrix.dot(rho)
#     
#     state = steady_states[i-1] if i > 0 else init_state 
#     converged = False
#     r = complex_ode(f).set_integrator('dopri5', nsteps=50000)
#     r.set_initial_value(state, t0)
#     while not converged and r.successful():
#         prev_state = state
#         state = r.integrate(r.t+time_step)
#         try:
#             npt.assert_allclose(state, prev_state, rtol=0, atol=1.e-3) # make sure tolerances are suitable
#             converged = True
#             steady_states[i] = state
#         except AssertionError:
#             print np.amax(state - prev_state)
#         
# print 'Saving data...'
# np.savez('../../data/HEOM_time_propagate_to_steady_state_test.npz', reorg_energy_values=reorg_energy_values, steady_states=steady_states)
            

        

