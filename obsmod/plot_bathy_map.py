"""
Plot station bathymetry to compare to station error plots to tease out 
whether the spatial pattern of errors has to do with resolving shallow areas
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial import cKDTree

from matplotlib.colors import TwoSlopeNorm
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun

Ldir = Lfun.Lstart()

# ============================================================
# USER SETTINGS
# ============================================================

year = '2014'
otype = 'bottle'

# Map extent
lat_low = 46.95
lat_high = 48.35
lon_low = -123.3
lon_high = -122.1

# Dataset keys
obs_key = 'obs'
lo_key = 'cas7_t1_x11ab'
ssc_key = 'ssc'

# ============================================================
# LOAD DATA
# ============================================================

# load the basin masks
mask_ds = xr.open_dataset('basin_masks_from_pugetsoundDObox.nc')

in_dir = Ldir['parent'] / 'LO_output' / 'obsmod'
out_dir = Ldir['parent'] / 'LO_output' / 'obsmod_plots'
Lfun.make_dir(out_dir)

in_fn = in_dir / (
    f'combined_{otype}_{year}_cas7_t1_x11ab_ssc.pkl'
)

df0_dict = pickle.load(open(in_fn, 'rb'))

# Remove non-DataFrame entries such as metadata
df0_dict = {
    key: df
    for key, df in df0_dict.items()
    if isinstance(df, pd.DataFrame)
}

# ============================================================
# LOAD COMBINED DATASET
# ============================================================

with open(in_fn, 'rb') as file:
    df0_dict = pickle.load(file)

# Remove metadata or other non-DataFrame entries
df0_dict = {
    key: value
    for key, value in df0_dict.items()
    if isinstance(value, pd.DataFrame)
}

obs_df = df0_dict['obs'].copy()

required_columns = ['name', 'lon', 'lat']

# ============================================================
# CREATE ONE LOCATION PER STATION
# ============================================================

station_df = obs_df[
    ['name', 'lon', 'lat']
].copy()

station_df['name'] = station_df['name'].astype(str)

station_df = station_df.replace(
    [np.inf, -np.inf],
    np.nan,
)

station_df = station_df.dropna(
    subset=['name', 'lon', 'lat']
)

station_df = (
    station_df
    .groupby('name', as_index=False)
    .agg(
        lon=('lon', 'mean'),
        lat=('lat', 'mean'),
        n_samples=('name', 'size'),
    )
)

# ------------------------------------------------------------
# Filter to Puget Sound basin mask
# ------------------------------------------------------------
mask_lon = mask_ds['lon_rho'].values
mask_lat = mask_ds['lat_rho'].values
puget_sound_mask = (
    mask_ds['mask_pugetsound'].values)

# Build a nearest-neighbor tree from valid grid coordinates
grid_points = np.column_stack((
    mask_lon.ravel(),
    mask_lat.ravel(),
))

valid_grid = (
    np.isfinite(grid_points[:, 0])
    & np.isfinite(grid_points[:, 1])
)

tree = cKDTree(grid_points[valid_grid])

# Observation/station sample coordinates
station_points = np.column_stack((
    station_df['lon'].to_numpy(),
    station_df['lat'].to_numpy(),
))

# Find nearest valid grid point for each sample
_, nearest_valid_index = tree.query(station_points)

# Convert indices from the valid subset back to flattened grid indices
valid_flat_indices = np.flatnonzero(valid_grid)

nearest_flat_index = valid_flat_indices[
    nearest_valid_index
]

eta_index, xi_index = np.unravel_index(
    nearest_flat_index,
    mask_lon.shape,
)

# Determine whether each sample is inside Puget Sound
inside_puget_sound = (
    puget_sound_mask[eta_index, xi_index] == 1
)

station_df = station_df.loc[
    inside_puget_sound
].copy()

# ============================================================
# LOAD LIVEOCEAN GRID AND BATHYMETRY
# ============================================================

Ldir = Lfun.Lstart(gridname='cas7')
grid_fn = Ldir['grid'] / 'grid.nc'
grid_ds = xr.open_dataset(grid_fn)
with xr.open_dataset(grid_fn) as grid_ds:

    lon_rho = grid_ds['lon_rho'].values
    lat_rho = grid_ds['lat_rho'].values
    h = grid_ds['h'].values
    mask_rho = grid_ds['mask_rho'].values

# Mask land in the background field
h_plot = np.where(
    mask_rho == 1,
    h,
    np.nan,
)

# ============================================================
# ASSIGN NEAREST WET-CELL BATHYMETRY TO EACH STATION
# ============================================================

valid_ocean = (
    (mask_rho == 1)
    & np.isfinite(lon_rho)
    & np.isfinite(lat_rho)
    & np.isfinite(h)
)

ocean_points = np.column_stack((
    lon_rho[valid_ocean],
    lat_rho[valid_ocean],
))

ocean_depths = h[
    valid_ocean
]

bathymetry_tree = cKDTree(
    ocean_points
)

station_points = station_df[
    ['lon', 'lat']
].to_numpy()

nearest_distance, nearest_ocean_index = (
    bathymetry_tree.query(station_points)
)

station_df['bathymetry_m'] = ocean_depths[
    nearest_ocean_index
]

# ============================================================
# PLOT
# ============================================================

plt.close('all')

fig, ax = plt.subplots(
    figsize=(6.5, 8),
    constrained_layout=True,
)

# Light bathymetry background
ax.pcolormesh(
    lon_rho,
    lat_rho,
    h_plot,
    cmap='Greys',
    shading='auto',
    alpha=0.25,
    zorder=0,
)

pfun.add_coast(ax)

scatter = ax.scatter(
    station_df['lon'],
    station_df['lat'],
    c=station_df['bathymetry_m'],
    cmap='viridis_r',
    s=90,
    edgecolor='black',
    linewidth=0.7,
    zorder=5,
)

ax.set_xlim(
    lon_low,
    lon_high,
)

ax.set_ylim(
    lat_low,
    lat_high,
)

pfun.dar(ax)

ax.set_xlabel(
    'Longitude',
    fontsize=10,
)

ax.set_ylabel(
    'Latitude',
    fontsize=10,
)

ax.tick_params(
    axis='both',
    which='major',
    labelsize=8,
)

ax.set_title(
    'Puget Sound station bathymetry',
    fontsize=13,
)

cbar = fig.colorbar(
    scatter,
    ax=ax,
    orientation='vertical',
    pad=0.03,
)

cbar.set_label(
    'LiveOcean grid depth (m)',
    fontsize=10,
)

cbar.ax.tick_params(
    labelsize=8
)

out_fn = out_dir / (
    f'{otype}_{year}_station_bathymetry.png'
)

fig.savefig(
    out_fn,
    dpi=300,
    bbox_inches='tight'
)
print(f'\nSaved figure to:\n{out_fn}')

plt.show()

grid_ds.close()