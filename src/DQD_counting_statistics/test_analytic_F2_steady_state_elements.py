'''
Created on 23 May 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
#from DQD_counting_statistics.DQD_model import DQDModel

#model = DQDModel(remove_elements=True)

epsilon_values = np.linspace(-10, 10, 200)
F2_values = np.zeros(epsilon_values.size)
steady_states = np.zeros((epsilon_values.size, 5))

T_c = 3.
Gamma_L = 1.
Gamma_R = 1.e-4

def liouvillian(Gamma_R, Gamma_L, epsilon, T_c):
    return np.array([[-Gamma_L, 0, Gamma_R, 0, 0],
                     [Gamma_L, 0, 0, 0, -2.*T_c],
                     [0, 0, -Gamma_R, 0, 2.*T_c],
                     [0, 0, 0, -0.5*Gamma_R, 2.*epsilon],
                     [0, T_c, -T_c, -2.*epsilon, -0.5*Gamma_R]])

def F2(Gamma_R, Gamma_L, epsilon, T_c):
    L = liouvillian(Gamma_R, Gamma_L, epsilon, T_c)
    steady_state = utils.stationary_state_svd(L, np.array([1., 1., 1., 0, 0]))
    rho_00 = steady_state[0]
    rho_LL = steady_state[1]
    rho_RR = steady_state[2]
    real_rho_LR = steady_state[3]
    imag_rho_LR = steady_state[4]
    return 1. + (2. / (0.25*(Gamma_R**2)*Gamma_L + 2.*(T_c**2)*Gamma_L + 4.*(epsilon**2)*Gamma_L + (T_c**2)*Gamma_R)) \
                * (-(T_c**2)*Gamma_R*rho_LL - (0.25*(Gamma_R**2)*Gamma_L + (T_c**2)*Gamma_R + 4.*(epsilon**2)*Gamma_L)*rho_RR \
                 + 4.*T_c*epsilon*Gamma_L*real_rho_LR + T_c*Gamma_R*Gamma_L*imag_rho_LR), steady_state
    '''
    return 1. + (2. / (0.25*(Gamma_R**2)*Gamma_L + 2.*(T_c**2)*Gamma_L + 4.*(epsilon**2)*Gamma_L + (T_c**2)*Gamma_R)) \
        * ((4.*(T_c**2)*Gamma_R + 4.*(T_c**2)*Gamma_L)*(rho_RR**2) + (2.*(T_c**2)*Gamma_R + 4.*(T_c**2)*Gamma_L)*rho_RR*rho_LL + 4.*(T_c**2)*Gamma_R*rho_RR*rho_00 \
           - (0.25*(Gamma_R**2)*Gamma_L + 4.*(epsilon**2)*Gamma_L + 5.*(T_c**2)*Gamma_R + 4.*(T_c**2)*Gamma_L)*rho_RR \
           - (T_c**2)*Gamma_R*rho_LL + 4.*T_c*epsilon*Gamma_L*real_rho_LR + T_c*Gamma_R*Gamma_L*imag_rho_LR)
    '''
        
for i,E in enumerate(epsilon_values):
    F2_values[i], steady_states[i] = F2(Gamma_R, Gamma_L, E, T_c)

plt.subplot(121)
plt.plot(epsilon_values, F2_values)
plt.subplot(122)
plt.plot(epsilon_values, Gamma_R*steady_states.T[2])
plt.show()