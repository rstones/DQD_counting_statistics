'''
Created on 8 Mar 2016

@author: rstones
'''
import numpy as np
#from DQD_counting_statistics.DQD_model import DQDModel
import quant_mech.utils as utils
import matplotlib.pyplot as plt
import matplotlib
from counting_statistics.fcs_solver import FCSSolver

font = {'size':20}
matplotlib.rc('font', **font)

#model = DQDModel(remove_elements=True)

def dqd_hamiltonian(bias, T):
    return np.array([[0,0,0],[0,bias/2,T],[0,T,-bias/2]])

dqd_lindblad_ops = [np.array([[0,0,0],[1,0,0],[0,0,0]]), np.array([[0,0,1],[0,0,0],[0,0,0]])]
dqd_lindblad_rates = [1., 1.]

solver = FCSSolver(dqd_hamiltonian(0, 3.), dqd_lindblad_ops, dqd_lindblad_rates, np.array([0,1]), reduce_dim=True)
freq_range = np.linspace(0, 10., 100)
bias_values = np.array([0]) #, 1.5, 3., 4.5, 6.])

F2 = np.zeros((bias_values.size, freq_range.size))

plt.axhline(1., ls='--', color='grey')

for i,v in enumerate(bias_values):
    #model.bias = v
    solver.H = dqd_hamiltonian(v, 3.)
    print solver.liouvillian()
    #ss = utils.stationary_state_svd(model.liouvillian(), model.density_vector_populations())
    #F2[i] = model.second_order_fano_factor(ss, freq_range=freq_range)
    F2[i] = solver.second_order_fano_factor(freq_range)
    plt.plot(freq_range, F2[i], label=r'$\epsilon$ = ' + str(v), linewidth=3)
plt.legend().draggable()
plt.xlabel('frequency')
plt.ylabel(r'F$^{(2)}$($\omega$)')
plt.show()