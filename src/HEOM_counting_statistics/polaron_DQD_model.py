'''
Created on 13 Dec 2016

@author: richard
'''
import numpy as np
import scipy.integrate as int
import scipy.misc.derivative as deriv

class PolaronDQDModel(object):
    
    def _init__(self, Gamma_L, Gamma_R, bias, T_c, spectral_density, beta):
        self.Gamma_L = Gamma_L
        self.Gamma_R = Gamma_R
        self.bias = bias
        self.T_c = T_c
        
        self.beta = beta
        self.spectral_density = spectral_density
    
    def zero_frequency_fano_factor(self):
        return 1. + 2.*self.Gamma_R * self.diff_nr()
    
    def diff_nr(self):
        
        def differentiand(z):
            return self.g_plus(z) / self.N(z)
        
        return self.Gamma_L * deriv(differentiand, 0, dx=1.e-6)
    
    def g_plus(self, z):
        cz = self.Cz(z)
        return self.T_c**2 * (cz / (1. + 0.5*self.Gamma_R*cz) + cz)
    
    def g_minus(self, z):
        cz = self.Cz(z)
        return self.T_c**2 * (cz.conj() / (1. + 0.5*self.Gamma_R*cz) + cz)

    def N(self, z):
        return (z + self.Gamma_R + self.g_minus(z)) * (z + self.Gamma_L) + (z + self.Gamma_R + self.Gamma_L) * self.g_plus(z)
    
    def Ct(self, t):
        freq = np.linspace(0, 100, 999) # simps works best with an odd number of samples
        
        def integrand(omega):
            return (self.spectral_density(omega) / omega**2) \
                            * ((1. - np.cos(omega*t)) * (1./np.tanh(self.beta*omega/2.)) \
                                + 1.j*np.sin(omega*t))
        
        return np.exp(-int.simps(integrand(freq), freq))
    
    def Cz(self, z):
        time = np.linspace(0, 100, 999)
        
        def integrand(t):
            return np.exp(-z*t) * np.exp(1.j*self.bias*t) *self.Ct(t)
    
        return int.simps(integrand, time)
    