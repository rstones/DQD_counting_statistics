import numpy as np

data = np.load('../../data/HEOM_F2_bias_UBO_N6_K4.npz')
bias_values = data['bias_values']
F2 = data['F2']
mean = data['mean']

import matplotlib.pyplot as plt

plt.subplot(121)
for i in range(4):
    plt.plot(bias_values, mean[i])

plt.subplot(122)
for i in range(4):
    plt.plot(bias_values, F2[i])

plt.show()