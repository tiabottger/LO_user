"""
Tia Bottger created 1/19/2026
Plot the stretched grid for the ae0 analytical model estuary.
Look at resolution in meter coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt

from lo_tools import zfun
import gfun_utility as gfu

# ae0 analytical model estuary
# copied from LO/user/gfun_user.py 

# define grid coordinates
lon_list = [-2, 0, 1, 2]
x_res_list = [2500, 500, 500, 2500]

lat_list = [43, 44.9, 45.1, 47]
y_res_list = [2500, 500, 500, 2500]

Lon_vec, Lat_vec = gfu.stretched_grid(
    lon_list, x_res_list,
    lat_list, y_res_list)
lon, lat = np.meshgrid(Lon_vec, Lat_vec)

# make bathymetry by hand
z = np.zeros(lon.shape)
x, y = zfun.ll2xy(lon, lat, 0, 45) # convert to meter resolution
zshelf = x * 1e-3 
zestuary = -20 + 20*x/1e5 + 20/(1e4)*np.abs(y)
z = zshelf
mask = zestuary < z # land mask (where estuary shallower than shelf)
z[mask] = zestuary[mask]

# make masked array for plotting 
# set land points to nan
zm = z.copy()
zm[zm >= 0] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

# subplot 1: bathymetry -----------------
ax = axes[0]
cs = ax.pcolormesh(x, y, zm, vmin=np.nanmin(z), vmax=0, cmap='Spectral_r')
fig.colorbar(cs, ax=ax, label='z [m]')
ax.set_title('ae0 bathymetry')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.axis('equal')
ax.grid(True)

# subplot 2: bathymetry + grid points -----------------
ax = axes[1]
cs = ax.pcolormesh(x, y, zm, vmin=np.nanmin(z), vmax=0, cmap='Spectral_r')
fig.colorbar(cs, ax=ax, label='z [m]')
ax.plot(x, y, '.k', alpha=0.1, markersize=1)
ax.set_title('ae0 bathymetry + grid points')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.axis('equal')
ax.grid(True)

plt.tight_layout()
plt.show()

# ----- plot grid bathymetry-----
fig, ax = plt.subplots()
cs = ax.pcolormesh(x, y, zm, vmin=z.min(), vmax=0, cmap='Spectral_r')
fig.colorbar(cs, ax=ax) 
ax.plot(x, y, '.k', alpha=0.1, markersize=1)
ax.set_title('ae0 stretched grid (meters)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.axis('equal')
ax.grid(True)
plt.show()

# ----- plot grid bathymetry overlay points-----
fig, ax = plt.subplots()
cs = ax.pcolormesh(x, y, zm, vmin=z.min(), vmax=0, cmap='Spectral_r')
fig.colorbar(cs, ax=ax) 
ax.plot(x, y, '.k', alpha=0.1, markersize=1)
ax.set_title('ae0 stretched grid (meters)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.axis('equal')
plt.show()

# ----- plot grid points in meters -----
plt.figure(figsize=(6, 8))
plt.plot(x, y, '.k', markersize=1)
plt.title('ae0 stretched grid (meters)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.grid(True)
plt.show()

# ----- plot grid points -----
plt.figure(figsize=(6, 8))
plt.plot(lon, lat, '.k', markersize=1)
plt.title('ae0 stretched grid resolution')
plt.xlabel('lon')
plt.ylabel('lat')
plt.axis('equal')
plt.grid(True)
plt.show()