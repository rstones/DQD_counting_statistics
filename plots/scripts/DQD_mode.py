import numpy as np
import matplotlib.pyplot as plt
import matplotlib

data = np.load('../../data/DQD_mode_mean_F2_exact_data.npz')
bias_values = data['bias_values']
beta_values = data['beta_values']
damping_values = data['damping_values']
F2 = data['F2']
mean = data['mean']

for i,g in enumerate(damping_values):
    
    for j,B in enumerate(beta_values):
        plt.subplot(2,3,i+1)
        plt.plot(bias_values, mean[i,j])
        plt.subplot(2,3,i+4)
        plt.plot(bias_values, F2[i,j])
        
plt.subplot(2,3,4)
plt.ylim(0.5, 1.3)
plt.subplot(2,3,5)
plt.ylim(0.5, 1.3)
plt.subplot(2,3,6)
plt.ylim(0.5, 1.3)

plt.show()
            
            