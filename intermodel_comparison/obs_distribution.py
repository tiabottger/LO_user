"""
Exploring the distribution statistics of observational DO data from bottle sample dataset
"""


import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial import cKDTree

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun

Ldir = Lfun.Lstart()

year = '2014'
otype = 'bottle'

# ============================================================
# LOAD DATA
# ============================================================

# Load the basin masks
mask_ds = xr.open_dataset(
    Ldir['LOu'] /
    'obsmod' /
    'basin_masks_from_pugetsoundDObox.nc'
)

in_dir = (
    Ldir['parent']
    / 'LO_output'
    / 'obsmod'
)

out_dir = (
    Ldir['parent']
    / 'LO_output'
    / 'obsmod_plots'
)

Lfun.make_dir(out_dir)

in_fn = in_dir / (
    f'combined_{otype}_{year}_cas7_t1_x11ab_ssc_ssmpnnl.pkl'
)

df0_dict = pickle.load(
    open(in_fn, 'rb')
)

# Remove non-DataFrame entries such as metadata
df0_dict = {
    key: df
    for key, df in df0_dict.items()
    if isinstance(df, pd.DataFrame)
}

data = pd.read_pickle(
    in_dir / "combined_bottle_2014_cas7_t1_x11ab_ssc.pkl")

obs = data['obs'].copy()

# ============================================================
# Apply Puget Sound basin mask
# ============================================================

mask_lon = mask_ds['lon_rho'].values
mask_lat = mask_ds['lat_rho'].values
puget_sound_mask = (
    mask_ds['mask_pugetsound'].values
)

# Build nearest-neighbor tree from valid grid coordinates
grid_points = np.column_stack((
    mask_lon.ravel(),
    mask_lat.ravel(),
))

valid_grid = (
    np.isfinite(grid_points[:, 0])
    & np.isfinite(grid_points[:, 1])
)

tree = cKDTree(
    grid_points[valid_grid]
)

# Observation/station sample coordinates
sample_points = np.column_stack((
    obs['lon'].to_numpy(),
    obs['lat'].to_numpy(),
))

# Find nearest valid grid point for each sample
_, nearest_valid_index = tree.query(
    sample_points
)

# Convert indices from valid subset back
# to flattened grid indices
valid_flat_indices = np.flatnonzero(
    valid_grid
)

nearest_flat_index = (
    valid_flat_indices[
        nearest_valid_index
    ]
)

eta_index, xi_index = np.unravel_index(
    nearest_flat_index,
    mask_lon.shape,
)

# Determine whether each sample is inside
# Puget Sound
inside_puget_sound = (
    puget_sound_mask[
        eta_index,
        xi_index
    ] == 1
)

obs = obs.loc[
    inside_puget_sound
].copy()

# ============================================================
# Find nearest bathymetry for each observation using LiveOcean grid
# ============================================================

# Load LiveOcean grid
Ldir = Lfun.Lstart(gridname='cas7')
grid_fn = Ldir['grid'] / 'grid.nc'
grid_ds = xr.open_dataset(grid_fn)

lon_rho = grid_ds['lon_rho'].values
lat_rho = grid_ds['lat_rho'].values
h = grid_ds['h'].values
mask_rho = grid_ds['mask_rho'].values

print('Grid shape:', h.shape)

# Identify wet cells
wet = (
    (mask_rho == 1) &
    np.isfinite(lon_rho) &
    np.isfinite(lat_rho) &
    np.isfinite(h)
)

# Convert lon/lat to 3D unit sphere coordinates
def lonlat_to_xyz(lon, lat):

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.column_stack([x, y, z])


# Flatten wet grid cells
wet_flat = np.where(wet.ravel())[0]

grid_xyz = lonlat_to_xyz(
    lon_rho.ravel()[wet_flat],
    lat_rho.ravel()[wet_flat]
)

# Build KD-tree
tree = cKDTree(grid_xyz)

# ------------------------------------------------------------
# Find nearest grid cell for each observation
# ------------------------------------------------------------

obs = df0_dict['obs'].copy()

# Valid observation coordinates
valid = (
    np.isfinite(obs['lon'].values) &
    np.isfinite(obs['lat'].values)
)

obs_xyz = lonlat_to_xyz(
    obs.loc[valid, 'lon'].values,
    obs.loc[valid, 'lat'].values
)

# Find nearest wet grid cell
distance, nearest_idx = tree.query(obs_xyz)

# Convert nearest KD-tree indices to original grid indices
nearest_flat = wet_flat[nearest_idx]

nearest_j, nearest_i = np.unravel_index(
    nearest_flat,
    h.shape
)

# Extract bathymetry
LO_h = np.full(len(obs), np.nan)

LO_h[valid] = h[nearest_j, nearest_i]

# Save to dataframe
obs['LO_h'] = LO_h

# Also save nearest grid coordinates
obs['LO_lon'] = np.nan
obs['LO_lat'] = np.nan

obs.loc[valid, 'LO_lon'] = lon_rho[nearest_j, nearest_i]
obs.loc[valid, 'LO_lat'] = lat_rho[nearest_j, nearest_i]

# Convert KD-tree chord distance to approximate distance in km
earth_radius = 6371.0

obs['grid_distance_km'] = np.nan

obs.loc[valid, 'grid_distance_km'] = (
    2 * earth_radius *
    np.arcsin(np.clip(distance / 2, 0, 1))
)

# ============================================================
# Calculate depth percentage of total water column
# ============================================================

# Positive-down observation depth
obs['depth_m'] = np.abs(obs['z'])

# Depth as percentage of total water column
obs['depth_pct'] = (
    obs['depth_m'] / obs['LO_h']
) * 100

# Inspect
print(
    obs[['depth_m', 'LO_h', 'depth_pct']].describe()
)

# Check observations deeper than local bathymetry
print(
    'Observations deeper than LO bathymetry:',
    (obs['depth_pct'] > 100).sum()
)

# ============================================================
# Plot histogram of observation depth values
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5)
)

# Absolute depth
depth_bins = np.arange(
    0,
    np.ceil(obs['depth_m'].max() / 10) * 10 + 10,
    10
)

axes[0].hist(
    obs['depth_m'].dropna(),
    bins=depth_bins,
    color='gray',
    orientation='horizontal',
    edgecolor='black',
    linewidth=0.7
)

axes[0].set_xlabel('Number of Observations')
axes[0].set_ylabel('Observation Depth (m)')
axes[0].set_title('Observational Depth Distribution')

axes[0].invert_yaxis()


# Relative depth
pct_bins = np.arange(0, 105, 2)

axes[1].hist(
    obs['depth_pct'].dropna(),
    bins=pct_bins,
    color='gray',
    orientation='horizontal',
    edgecolor='black',
    linewidth=0.7
)

axes[1].set_xlabel('Number of Observations')
axes[1].set_ylabel('Depth (% of Water Column)')
axes[1].set_title('Relative Observational Depth Distribution')

axes[1].set_ylim(0, 100)
axes[1].invert_yaxis()

plt.tight_layout()

plt.savefig(
    out_dir / 'obs_depth_distribution.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
plt.close()

# ============================================================
# HISTOGRAM OF WATER COLUMN DEPTHS AT OBSERVATION LOCATIONS
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

water_depth = obs['LO_h'].dropna()

bins = np.arange(
    0,
    np.ceil(water_depth.max() / 10) * 10 + 10,
    10
)

ax.hist(
    water_depth,
    bins=bins,
    color='gray',
    orientation='horizontal',
    edgecolor='black',
    linewidth=0.8
)

ax.set_xlabel('Number of Observations')
ax.set_ylabel('Total Water Column Depth (m)')
ax.set_title('Water Column Depth at Observational Locations')

ax.legend()
plt.tight_layout()
plt.savefig(
    out_dir / 'obs_bathy_distribution.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
plt.close()