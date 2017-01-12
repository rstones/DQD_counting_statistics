'''
Created on 28 Nov 2016

@author: richard
'''
import numpy as np
import matplotlib.pyplot as plt
from HEOM_counting_statistics.DQD_HEOM_model import DQDHEOMModel

model = DQDHEOMModel(0, 3., 1., 1.)
freq = np.linspace(0,10,100)
F2 = model.finite_frequency_fano_factor(freq)
plt.plot(freq, F2)
plt.xlabel('frequency $\omega$')
plt.ylabel(r'F$^{(2)}(\omega)$')
plt.show()