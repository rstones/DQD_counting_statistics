'''
Created on 13 Jan 2017

@author: richard
'''
import numpy as np
import scipy.linalg as la
import scipy.constants as constants

def gamma(bias, T_c, alpha, beta, cutoff_freq):
    '''Pure dephasing rate for an Ohmic spectral density.'''
    delta = np.sqrt(bias**2 + 4.*T_c**2)
    return (alpha*np.pi/delta**2) * (bias**2 / beta + 2.*T_c**2 * delta * np.exp(-delta/cutoff_freq) * (1./np.tanh(0.5*beta*delta)))

def gamma_plus_minus(bias, T_c, alpha, beta, cutoff_freq, pm=1.):
    '''Population relaxation dephasing rate for Ohmic spectral density.'''
    delta = np.sqrt(bias**2 + 4.*T_c**2)
    return (alpha*np.pi*T_c/delta**2) * (bias/beta - 0.5*np.exp(-delta/cutoff_freq) * (bias*delta*(1./np.tanh(0.5*beta*delta)) \
                                     + pm * delta**2))

def liouvillian(Gamma_L, Gamma_R, bias, T_c, alpha, beta, cutoff_freq):
    gam = gamma(bias, T_c, alpha, beta, cutoff_freq)
    gam_plus = gamma_plus_minus(bias, T_c, alpha, beta, cutoff_freq, pm=1.)
    gam_minus = gamma_plus_minus(bias, T_c, alpha, beta, cutoff_freq, pm=-1.)
    return np.array([[-Gamma_L, 0, Gamma_R, 0, 0],
                     [Gamma_L, 0, 0, 0, 2.*T_c],
                     [0, 0, -Gamma_R, 0, -2.*T_c],
                     [0, gam_plus, -gam_minus, -0.5*Gamma_R - gam, -bias],
                     [0, -T_c, T_c, bias, -0.5*Gamma_R - gam]]), gam, gam_plus, gam_minus
    
def stationary_state(L, pops):
    # calculate
    u,s,v = la.svd(L)
    # check for number of nullspaces
    # normalize
    ss = v[-1].conj() / np.dot(pops, v[-1])
    return ss

def F2(Gamma_L, Gamma_R, bias, T_c, alpha, beta, cutoff_freq):
    '''Zero-frequency Fano factor for a dissipative DQD system.'''
    L,gam,gam_plus,gam_minus = liouvillian(Gamma_L, Gamma_R, bias, T_c, alpha, beta, cutoff_freq)
    ss = stationary_state(L, np.array([1,1,1,0,0]))
    return 1. + 2.*Gamma_R * ((-(T_c**2*Gamma_R + 2.*T_c*gam_plus*bias + 2.*T_c**2*gam)*ss[1] \
                               - (0.25*Gamma_R**2*Gamma_L + bias**2*Gamma_L + T_c**2*Gamma_R + Gamma_L*gam**2 + Gamma_R*Gamma_L*gam + 2.*T_c**2*gam + 2.*T_c*gam_plus*bias)*ss[2] \
                               + 2.*T_c*bias*Gamma_L*ss[3] - (T_c*Gamma_R*Gamma_L + 2.*T_c*gam*Gamma_L)*ss[4]) \
                              / (0.25*Gamma_R**3*Gamma_L + Gamma_R*Gamma_L*gam**2 + Gamma_R**2*Gamma_L*gam \
                                 - 2.*T_c*gam_minus*bias*Gamma_L + 2.*T_c*gam_plus*bias*Gamma_L + bias**2*Gamma_R*Gamma_L \
                                 + 4.*T_c**2*gam*Gamma_L + 2.*T_c**2*Gamma_R*Gamma_L + 2.*T_c*Gamma_R*gam_plus*bias \
                                 + T_c**2*Gamma_R**2 + 2.*T_c**2*Gamma_R*gam))
    
Gamma_L = 0.1 # meV
Gamma_R = 2.5e-3 # meV
bias = 0
T_c = 0.1 # meV
temperature = [1.4, 2.7, 12.] # Kelvin
k_B = constants.physical_constants["Boltzmann constant in eV/K"][0] * 1.e3 # meV / Kelvin
beta = [1. / (k_B * T) for T in temperature]
alpha = 8.e-4
cutoff_freq = 5. # meV

bias_values = np.linspace(-1, 1, 100)
F2_values = np.zeros((len(beta)+1, bias_values.size))

for i,E in enumerate(bias_values):
    F2_values[0,i] = F2(Gamma_L, Gamma_R, E, T_c, 0, 1., cutoff_freq)

for i,B in enumerate(beta):
    for j,E in enumerate(bias_values):
        F2_values[i+1,j] = F2(Gamma_L, Gamma_R, E, T_c, alpha, B, cutoff_freq)
        
import matplotlib.pyplot as plt
plt.plot(bias_values, F2_values[0], linewidth=3, ls='--', color='k', label='no phonons')
for i,B in enumerate(beta):
    plt.plot(bias_values, F2_values[i+1], linewidth=3, label="T = " + str(temperature[i]) + 'K')
plt.axhline(1., ls='--', color='grey', linewidth=2)
plt.legend().draggable()
plt.xlim(-1.05, 1.05)
plt.ylim(0.72, 1.25)
plt.show()


