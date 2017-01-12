import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':16}
matplotlib.rc('font', **font)

freq_values = np.linspace(0,100,1000)

# ohmic params
g = 8.e-4
ohmic_cutoff = 5. # meV

# drude params
reorg_energy = 0.00147 # meV
drude_cutoff = 5. # meV

def ohmic_spectral_density(omega, g, cutoff):
    return g * omega * np.exp(-omega/cutoff)

def drude_spectral_density(omega, reorg_energy, cutoff):
    return (2. * omega * reorg_energy * cutoff) / (omega**2 + cutoff**2)

plt.plot(freq_values, ohmic_spectral_density(freq_values, g, ohmic_cutoff), linewidth=3, label='Ohmic')
plt.plot(freq_values, drude_spectral_density(freq_values, reorg_energy, drude_cutoff), linewidth=3, ls='--', color='red', label='Drude')
plt.xlabel('frequency (meV)')
plt.ylabel('arb. units')
plt.legend().draggable()
plt.show()