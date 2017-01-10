'''
Created on 1 Dec 2016

@author: richard
'''
import numpy as np
import scipy.constants as constants
from counting_statistics.fcs_solver import FCSSolver

def liouvillian(Gamma_R, Gamma_L, bias, T_c, gamma_plus, gamma_minus, gamma):
    '''DQD Liouvillian with dissipation derived under assumption of weak coupling to the
    heat bath.'''
    return np.array([[-Gamma_L, 0, Gamma_R, 0, 0],
                     [Gamma_L, 0, 0, 0, 2.*T_c],
                     [0, 0, -Gamma_R, 0, -2.*T_c],
                     [0, gamma_plus, -gamma_minus, -0.5*Gamma_R - gamma, -bias],
                     [0, -T_c, T_c, bias, -0.5*Gamma_R - gamma]])
    
def gamma(bias, T_c, g, beta, cutoff_freq):
    '''Pure dephasing rate for an Ohmic spectral density.'''
    delta = np.sqrt(bias**2 + 4.*T_c**2)
    return (g*np.pi/delta**2) * (bias**2 / beta + 2.*T_c**2 * delta * np.exp(-delta/cutoff_freq) * (1./np.tanh(0.5*beta*delta)))

def gamma_plus_minus(bias, T_c, g, beta, cutoff_freq, pm=1.):
    '''Population relaxation dephasing rate for Ohmic spectral density.'''
    delta = np.sqrt(bias**2 + 4.*T_c**2)
    return (g*np.pi*T_c/delta**2) * (bias/beta - 0.5*np.exp(-delta/cutoff_freq) * (bias*delta*(1./np.tanh(0.5*beta*delta)) \
                                     + pm * delta**2))

# units in meV unless stated
Gamma_R = 2.5e-3
Gamma_L = 0.1
T_c = 0.1
g = 8.e-4
cutoff_freq = 5.
temperature_values = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] * 10**3 # convert to meV
beta_values = [1. / (k_B * T) for T in temperature_values] # inverse meV
bias_values = np.linspace(-1, 1, 100)

jump_op = np.zeros((5,5))
jump_op[0,2] = Gamma_R
pops = np.array([1,1,1,0,0])
solver = FCSSolver(liouvillian(0,0,0,0,0,0,0), jump_op, pops)

current = np.zeros((len(beta_values)+1, bias_values.size))
F2 = np.zeros((len(beta_values)+1, bias_values.size))

def current_F2_energy_bias():
    current = np.zeros(bias_values.size)
    F2 = np.zeros(bias_values.size)
    for i,E in enumerate(bias_values):
        solver.L = liouvillian(Gamma_R, Gamma_L, E, T_c, \
                               gamma_plus_minus(E, T_c, g, beta, cutoff_freq, 1.), \
                               gamma_plus_minus(E, T_c, g, beta, cutoff_freq, -1.), \
                               gamma(E, T_c, g, beta, cutoff_freq))
        current[i] = solver.mean()
        F2[i] = solver.second_order_fano_factor(0)
    return current, F2

for j,beta in enumerate(beta_values):
    current[j+1], F2[j+1] = current_F2_energy_bias()

g = 0
current[0], F2[0] = current_F2_energy_bias() 
    
import matplotlib.pyplot as plt

colours = ['k', 'r', 'b', 'g']
for i in range(len(beta_values)+1):
    plt.subplot(121)
    plt.plot(bias_values, current[i], label=str(temperature_values[i-1])+'K' if i>0 else 'no phonons', \
                        ls='-' if i>0 else '--', color=colours[i], linewidth=2)

    plt.subplot(122)
    plt.plot(bias_values, F2[i], ls='-' if i>0 else '--', color=colours[i], linewidth=2)
    
plt.subplot(121)
plt.legend().draggable()

plt.subplot(122)
plt.ylim(0.7, 1.25)
plt.axhline(1, color='grey', ls='--')

plt.show()
    

    

    