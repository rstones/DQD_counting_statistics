'''
Created on 13 Dec 2016

@author: richard
'''
import numpy as np

class DissipativeDQDModel(object):
    
    def __init__(self, Gamma_L, Gamma_R, bias, T_c, spectral_density, beta):
        self.Gamma_L = Gamma_L
        self.Gamma_R = Gamma_R
        self.bias = bias
        self.T_c = T_c
        
        self.beta = beta
        self.spectral_density = spectral_density
        
        self.ohmic = False
        self.ohmic_alpha = 8.e-4
        self.ohmic_cutoff = 5.
    
    def liouvillian(self):
        '''DQD Liouvillian with dissipation derived under assumption of weak coupling to the
        heat bath.'''
        return np.array([[-self.Gamma_L, 0, self.Gamma_R, 0, 0],
                         [self.Gamma_L, 0, 0, 0, 2.*self.T_c],
                         [0, 0, -self.Gamma_R, 0, -2.*self.T_c],
                         [0, self.gamma_plus_minus(1), -self.gamma_plus_minus(-1), -0.5*self.Gamma_R - self.gamma(), -self.bias],
                         [0, -self.T_c, self.T_c, self.bias, -0.5*self.Gamma_R - self.gamma()]])
        
    def jump_matrix(self):
        jm = np.zeros((5,5))
        jm[0,2] = self.Gamma_R
        return jm
    
    def gamma(self):
        delta = np.sqrt(self.bias**2 + 4.*self.T_c**2)
        if self.ohmic:
            return self.ohmic_gamma()
        else:
            return (2.*np.pi * self.T_c**2 / delta**2) * self.spectral_density(delta) * (1. / np.tanh(self.beta*delta/2.))
    
    def gamma_plus_minus(self, pm):
        delta = np.sqrt(self.bias**2 + 4.*self.T_c**2)
        if self.ohmic:
            return self.ohmic_gamma_plus_minus(pm)
        else:
            return - self.spectral_density(delta) * ((self.bias * self.T_c * np.pi / (2.*delta**2)) * (1. / np.tanh(self.beta*delta/2.)) + pm * (self.T_c * np.pi / (2.*delta)))
    
    def ohmic_gamma(self):
        delta = np.sqrt(self.bias**2 + 4.*self.T_c**2)
        return (self.ohmic_alpha*np.pi/delta**2) * (self.bias**2 / self.beta + 2.*self.T_c**2 * delta * np.exp(-delta/self.ohmic_cutoff) * (1./np.tanh(0.5*self.beta*delta)))
    
    def ohmic_gamma_plus_minus(self, pm):
        delta = np.sqrt(self.bias**2 + 4.*self.T_c**2)
        return (self.ohmic_alpha*np.pi*self.T_c/delta**2) * (self.bias/self.beta - 0.5*np.exp(-delta/self.ohmic_cutoff) * (self.bias*delta*(1./np.tanh(0.5*self.beta*delta)) \
                                     + pm * delta**2))