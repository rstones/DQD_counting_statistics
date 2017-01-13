import numpy as np
import matplotlib.pyplot as plt

data = np.load('../../data/HEOM_F2_bias_drude_K_1.npz')
data_np = np.load('../../data/HEOM_F2_bias_drude_no_phonon.npz')

F2_np = data_np['F2']
F2 = data['F2']
bias_values = data['bias_values']

plt.plot(bias_values, F2_np, ls='--', color='k')
for i in range(1,4):
    plt.plot(bias_values, F2[i])
plt.show()