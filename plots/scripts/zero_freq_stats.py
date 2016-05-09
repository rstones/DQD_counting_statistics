import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':20}
matplotlib.rc('font', **font)

lw = 3

data = np.load('../../data/four_level_zero_freq_counting_statistics_data.npz')
Gamma_R_range = data['Gamma_R_range']
bias_values = data['bias_values']
current = data['current']
F2 = data['F2']
coherence = data['coherence']

fig,(ax1,ax2,ax3) = plt.subplots(1,3)
for i,v in enumerate(bias_values):
    ax1.semilogx(Gamma_R_range, current[i], label=r'$\epsilon =$ ' + str(v), linewidth=lw)
    ax2.semilogx(Gamma_R_range, F2[i], label=r'$\epsilon =$ ' + str(v), linewidth=lw)
    ax3.semilogx(Gamma_R_range, np.abs(coherence[i]), label=r'$\epsilon =$ ' + str(v), linewidth=lw)
    
ax1.set_xlabel(r'$\Gamma_R$')
ax2.set_xlabel(r'$\Gamma_R$')
ax3.set_xlabel(r'$\Gamma_R$')

ax1.set_ylabel(r'current')
ax2.set_ylabel(r'F$^{(2)}$(0)')
ax3.set_ylabel(r'|coherence|')

ax1.legend().draggable()
ax2.legend().draggable()
ax3.legend().draggable()
plt.show()