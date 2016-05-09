'''
Created on 14 Mar 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt


def F2(a1, a2, a0_prime, a1_prime):
    return 1. + ((2.*a0_prime*a2) / a1**2) - ((2.*a1_prime) / a1)

# def a1(Gamma_L, Gamma_R, T_c, e):
#     return (Gamma_R**2 * T_c**2) + (0.25 * Gamma_L * Gamma_R**3) + (7. * T_c**2 * Gamma_L * Gamma_R) + (4. * e**2 * Gamma_L * Gamma_R) \
#                             - (2. * Gamma_L * Gamma_R * T_c**2)
# 
# def a2(Gamma_R, T_c, e):
#     return (0.25*Gamma_R**3) + (7. * T_c**2 * Gamma_R) + (4. * e**2 * Gamma_R) + (1.25 * Gamma_R**2) + (4. * T_c**2) + (4. * e**2)

small_Gamma_R = True

if not small_Gamma_R:

    def a1(Gamma_L, Gamma_R, T_c, e):
        return (Gamma_R**2 * T_c**2) + (0.25 * Gamma_L * Gamma_R**3) + (e**2 * Gamma_L * Gamma_R) + (2. * Gamma_L * Gamma_R * T_c**2)
    
    def a2(Gamma_L, Gamma_R, T_c, e):
        return (0.25 * Gamma_R**3) + (e**2 * (Gamma_L + Gamma_R)) + (4. * T_c**2 * (Gamma_L + Gamma_R)) + (1.25 * Gamma_R**2 * Gamma_L)
    
    def a0_prime(Gamma_L, Gamma_R, T_c):
        return -Gamma_L * Gamma_R**2 * T_c**2
    
    def a1_prime(Gamma_L, Gamma_R, T_c):
        return -2. * Gamma_L * Gamma_R * T_c**2
    
    bias_values = np.array([0, 1.5, 3., 4.5, 6.])
    Gamma_L = 1.
    T_c = 3.
    Gamma_R_range = np.logspace(-4, 3, 1000)
    
    F2_values = np.zeros((bias_values.size, Gamma_R_range.size))
    
    for j,bias in enumerate(bias_values):
        for i,v in enumerate(Gamma_R_range):
            F2_values[j,i] = F2(a1(Gamma_L, v, T_c, bias), a2(Gamma_L, v, T_c, bias), a0_prime(Gamma_L, v, T_c), a1_prime(Gamma_L, v, T_c))
    
    for i,v in enumerate(bias_values):
        plt.semilogx(Gamma_R_range, F2_values[i], label=v)
    plt.legend().draggable()
    plt.show()

else:
    
    def a1(Gamma_L, T_c, e):
        return 0.25 * Gamma_L**3 + 4. * e**2 * Gamma_L
    
    def a2(Gamma_L, T_c, e):
        return 1.25 * Gamma_L**2 + 4. * e**2
    
    def a0_prime(Gamma_L, T_c):
        return Gamma_L**2 * T_c**2
    
    def a1_prime(Gamma_L, T_c):
        return 2. * Gamma_L * T_c**2
    
    bias_values = np.array([0, 1.5, 3., 4.5, 6.])
    Gamma_L = 1.
    T_c = 3.
    Gamma_R_range = np.logspace(-4, 3, 1000)
    
    F2_values = np.zeros(bias_values.size)
    
    for j,bias in enumerate(bias_values):
        F2_values[j] = F2(a1(Gamma_L, T_c, bias), a2(Gamma_L, T_c, bias), a0_prime(Gamma_L, T_c), a1_prime(Gamma_L, T_c))
    
    for i,v in enumerate(bias_values):
        plt.axhline(F2_values[i], label=v)
    plt.legend().draggable()
    plt.show()

