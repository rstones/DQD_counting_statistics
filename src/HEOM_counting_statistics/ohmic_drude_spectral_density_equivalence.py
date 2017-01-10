import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

freq_values = np.linspace(0,100,1000)

def ohmic_spectral_density(omega, g, cutoff):
    return g * omega * np.exp(-omega/cutoff)

def drude_spectral_density(omega, reorg_energy, cutoff):
    return (2. * omega * reorg_energy * cutoff) / (omega**2 + cutoff**2)

fig,ax = plt.subplots()
a = plt.plot(freq_values, ohmic_spectral_density(freq_values, 8.e-4, 5))
b, = plt.plot(freq_values, drude_spectral_density(freq_values, 8.e-4, 5))

ax_reorg_energy = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='white')
ax_cutoff = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg='white')

s_reorg_energy = Slider(ax_reorg_energy, 'Reorg energy', 0, 3.e-3, valinit=8.e-4)
s_cutoff = Slider(ax_cutoff, 'Cutoff freq', 0, 10, valinit=5)

def update(val):
    b.set_ydata(drude_spectral_density(freq_values, s_reorg_energy.val, s_cutoff.val))
    fig.canvas.draw_idle()
    print s_reorg_energy.val
s_reorg_energy.on_changed(update)
s_cutoff.on_changed(update)

plt.show()