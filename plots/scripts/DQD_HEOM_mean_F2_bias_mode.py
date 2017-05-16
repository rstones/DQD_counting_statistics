import numpy as np
import matplotlib.pyplot as plt

data = np.load('../../data/DQD_HEOM_mean_F2_bias_UBO_N5_K3_data.npz')

bias_values = data['bias_values']
beta = data['beta']
mean = data['mean']
F2 = data['F2']

for i in range(beta.size):
    plt.plot(bias_values, F2[i])
plt.show()