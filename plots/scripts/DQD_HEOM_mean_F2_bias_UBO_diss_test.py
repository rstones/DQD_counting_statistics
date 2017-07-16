import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# num_beta_values = 3
# F2 = []
# mean = []
# bias_values = []
# 
# for i in range(num_beta_values):
# 
#     data = np.load('../../remote_scripts/data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N6_K3_beta'+str(i)+'_data.npz')
#     F2.append(data['F2'][i])
#     mean.append(data['mean'][i])
#     bias_values.append(data['bias_values'])
#     beta_values = data['beta']
#     
# np.savez('../../remote_scripts/data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N6_K3_diss_test_data.npz', \
#          bias_values=bias_values[0], mean=np.array(mean), F2=np.array(F2), beta_values=beta_values)

data = np.load('../../remote_scripts/data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N6_K3_data.npz')
bias_values = data['bias_values']
beta_values = data['beta']
F2 = data['F2']
mean = data['mean']

test_data = np.load('../../remote_scripts/data/DQD_HEOM_mean_F2_bias_strong_coupling_UBO_N6_K3_diss_test_data.npz')
test_bias_values = test_data['bias_values']
test_beta_values = test_data['beta_values']
test_F2 = test_data['F2']
test_mean = test_data['mean']

data_np = np.load('../../data/HEOM_F2_bias_drude_data.npz')
bias_values_np = data_np['bias_values']
F2_np = data_np['F2'][0]

for i,B in enumerate(beta_values):
    plt.subplot(121)
    plt.plot(bias_values, F2[i], linewidth=3)
    plt.subplot(122)
    plt.plot(test_bias_values, test_F2[i], linewidth=3)

plt.subplot(121)    
plt.plot(bias_values_np, F2_np, ls='--', color='k', linewidth=3)
plt.subplot(122)    
plt.plot(bias_values_np, F2_np, ls='--', color='k', linewidth=3)

plt.show()