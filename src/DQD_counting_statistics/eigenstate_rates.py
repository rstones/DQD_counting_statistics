'''
Created on 9 Sep 2017

@author: richard
'''
import numpy as np

bias_values = np.linspace(-4, 4, 100)

T_c = 2.

def theta(bias, T_c):
    return 0.5 * np.arctan(2. * T_c / bias)

theta_values = theta(bias_values, T_c)

import matplotlib.pyplot as plt

plt.plot(bias_values, np.sin(theta_values)**2, label=r'$\sin^2\theta$')
plt.plot(bias_values, np.cos(theta_values)**2, label=r'$\cos^2\theta$')
plt.legend().draggable()
plt.xlabel('bias')
plt.show()