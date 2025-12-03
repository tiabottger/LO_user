"""
Code to explore the the shapes and magnitudes of various functions that
sre used in the NPZD model equations, comparing Fennel and Banas versions.
"""

import matplotlib.pyplot as plt
import numpy as np
from lo_tools import plotting_functions as pfun

import fennel_functions as ff
import banas_functions as bnf
from importlib import reload
reload(ff)
reload(bnf)

plt.close('all')
pfun.start_plot(fs=10, figsize=(18,10))

fig, axes = plt.subplots(nrows=2, ncols=3)

# Example calculations to look at functional dependence.
# F_ for Fennel, B_ for Banas

# mu_max(T)
T = np.linspace(0,25,100)
F_mu_max = ff.get_mu_max(T)
B_mu_max = bnf.mu_0 * np.ones(len(T))
#
ax = axes[0,0]
ax.plot(T, F_mu_max, '-b', lw=3, label='Fennel')
ax.plot(T, B_mu_max, '--b', lw=3, label='Banas')
ax.legend(loc='lower right')
ax.set_xlabel('Temperature [degC]')
ax.set_ylabel('mu_max [d-1]')
ax.text(.05,.9,'Max Phytoplankton Growth Rate vs. T',transform=ax.transAxes)
ax.grid(True)
ax.set_ylim(bottom=0)

# I(z) (called E(z) in Banas)
swrad0 = 500
z_w = np.linspace(-100,0,101)
z_rho = z_w[:-1] + np.diff(z_w)/2
# no phytoplankton or chlorophyll
Phy = np.zeros(len(z_rho))
Chl = np.zeros(len(z_rho))
F_E = ff.get_E(swrad0, z_rho, z_w, Chl)
B_E = bnf.get_E(swrad0, z_rho, z_w, Phy, 32)
#
ax = axes[0,1]
ax.plot(F_E, z_rho, '-', c='g', lw=3, label='Fennel')
ax.plot(B_E, z_rho, '--', c='g', lw=3, label='Banas')
ax.legend(loc='lower right')
ax.set_xlabel('E [W m-2]')
ax.set_ylabel('Z [m]')
ax.text(.05,.9,'Light vs. Z',transform=ax.transAxes)
ax.grid(True)

# f(E)
swrad0 = np.linspace(0,200,100)
T = 10 * np.ones(len(swrad0))
F_f = ff.get_f(swrad0, ff.get_mu_max(T))
B_f = bnf.get_f(swrad0)
#
ax = axes[1,0]
ax.plot(swrad0, F_f, '-', c='purple', lw=3, label='Fennel')
ax.plot(swrad0, B_f, '--', c='purple', lw=3, label='Banas')
ax.legend(loc='lower right')
ax.set_xlabel('E [W m-2]')
ax.set_ylabel('f(I)')
ax.text(.05,.9,'P-I Curve',transform=ax.transAxes)
ax.grid(True)

# L(NO3, NH4)
NO3 = np.linspace(0,30,100)
NH4 = 0 * NO3
L_NO3, L_NH4 = ff.get_L(NO3, NH4)
F_L = L_NO3 + L_NH4
B_L = bnf.get_L(NO3)
#
ax = axes[1,1]
ax.plot(NO3, F_L, '-', c='brown', lw=3, label='Fennel')
ax.plot(NO3, B_L, '--', c='brown', lw=3, label='Banas')
ax.legend(loc='lower right')
ax.set_xlabel('NO3 [mmol N m-3]')
ax.set_ylabel('L')
ax.text(.05,.9,'M-M Curve',transform=ax.transAxes)
ax.grid(True)

plt.show()
pfun.end_plot()
