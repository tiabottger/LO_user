"""
This script calculates hypoxic area and volume for Salish Sea Model,
reading netCDF files from SSM_pnnl/bottom_DO (on apogee). This file has also been
moved into LO_output/intermodel_comparison/bottom_DO.

SSM uses an unstructured grid, so this script has been adapted from
bottom_DO_plots.py, which was written for LiveOcean's structured grid.

This script searches for yearly box extractions in SSM_pnnl/ssm_box.
It uses the file created by ssm_get_bottom_DO_nc.py.
"""

# import things
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from pyproj import Transformer
from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

import sys
from pathlib import Path
pth = Path(__file__).absolute().parent.parent.parent / 'LO' / 'pgrid'
if str(pth) not in sys.path:
    sys.path.append(str(pth))

Ldir = Lfun.Lstart()

# ============================================================
# USER INPUTS
# ============================================================

region = 'pugetsoundDO'
year = '2014'

in_dir = Ldir['LOo'] / 'intermodel_comparison' / 'bottom_DO'

out_dir = Ldir['LOo'] / 'intermodel_comparison' / 'bottom_DO_plots'
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

# READ MODEL OUTPUT

ds_SSM = xr.open_dataset(
    in_dir / 'SSM_2014_pugetsound_bottom_DO_info.nc'
)

# READ Puget Sound BASIN MASK
basin_mask_ds = xr.open_dataset(
    Ldir['LOu'] /
    'obsmod' /
    'basin_masks_from_pugetsoundDObox.nc'
)
mask_ps = basin_mask_ds['mask_pugetsound'].values

# LIVE OCEAN GRID
grid_fn = Path('../../../LO_data/grids/cas7/grid.nc')
grid_ds = xr.open_dataset(grid_fn)

DO = ds_SSM['DO_bot'].values
hyp_thick = ds_SSM['hyp_thick'].values
thick_bot = ds_SSM['thick_bot'].values
depth_bot = ds_SSM['depth_bot'].values

lon = ds_SSM['lon'].values
lat = ds_SSM['lat'].values

n_time, n_node = DO.shape
n_elem = ds_SSM.sizes['elem']

time_SSM = pd.to_datetime(ds_SSM['time'].values)

lon_LO = grid_ds['lon_rho'].values
lat_LO = grid_ds['lat_rho'].values
mask_rho = grid_ds['mask_rho'].values

# ============================================================
# TRIANGULAR MESH CONNECTIVITY
# ============================================================

# nv should reference local_node indices.
nv = ds_SSM['nv'].values.astype(int)

print('nv minimum:', nv.min())
print('nv maximum:', nv.max())

if nv.min() == 0 and nv.max() < n_node:
    # Zero-based connectivity
    triangles = nv.copy()
elif nv.min() >= 1 and nv.max() == n_node:
    # One-based connectivity
    triangles = nv - 1
else:
    raise ValueError(
        'Cannot safely infer nv indexing. Inspect the extraction '
        'script and confirm that nv references local_node indices.'
    )

assert triangles.shape == (n_elem, 3)
assert triangles.min() >= 0
assert triangles.max() < n_node

# ============================================================
# PROJECT COORDINATES TO METERS
# ============================================================

# UTM zone 10N is suitable for this Puget Sound region.
transformer = Transformer.from_crs(
    'EPSG:4326',
    'EPSG:26910',
    always_xy=True
)

x, y = transformer.transform(lon, lat)

# Calculate the area of every triangular element.
x_tri = x[triangles]
y_tri = y[triangles]

element_area_m2 = 0.5 * np.abs(
    (x_tri[:, 1] - x_tri[:, 0]) *
    (y_tri[:, 2] - y_tri[:, 0])
    -
    (x_tri[:, 2] - x_tri[:, 0]) *
    (y_tri[:, 1] - y_tri[:, 0])
)

element_area_km2 = element_area_m2 / 1e6

print(
    'Element area range (km2):',
    element_area_km2.min(),
    element_area_km2.max()
)

# ============================================================
# MAP PUGET SOUND MASK ONTO SSM NODES
# ============================================================

mask_ps = basin_mask_ds['mask_pugetsound'].values
mask_lon = basin_mask_ds['lon_rho'].values
mask_lat = basin_mask_ds['lat_rho'].values

mask_points = np.column_stack((
    mask_lon.ravel(),
    mask_lat.ravel()
))

valid_mask_points = (
    np.isfinite(mask_points).all(axis=1)
)

tree = cKDTree(mask_points[valid_mask_points])

ssm_points = np.column_stack((lon, lat))
_, nearest = tree.query(ssm_points)

valid_flat = np.flatnonzero(valid_mask_points)
nearest_flat = valid_flat[nearest]

mask_j, mask_i = np.unravel_index(
    nearest_flat, mask_lon.shape
)

mask_node = mask_ps[mask_j, mask_i] == 1

# Classify an element as Puget Sound if at least two of
# its three vertices fall within the Puget Sound mask.
mask_elem = (
    mask_node[triangles].sum(axis=1) >= 2
)

print('SSM elements inside Puget Sound:', mask_elem.sum())

# Apply the basin mask to element areas.
area_km2 = element_area_km2 * mask_elem

# ============================================================
# NODE VALUES TO ELEMENT VALUES
# ============================================================

# The SSM variables are node-based. Approximate element values
# by averaging the three vertices of each triangular element.

def node_to_element(values, triangles):
    """
    Convert node-based data to element averages.

    Supports:
        (node,) -> (element,)
        (time, node) -> (time, element)
    """
    if values.ndim == 1:
        return np.nanmean(values[triangles], axis=1)

    return np.nanmean(values[:, triangles], axis=2)


DO_elem = node_to_element(DO, triangles)
hyp_thick_elem = node_to_element(hyp_thick, triangles)
depth_elem = node_to_element(depth_bot, triangles)
thick_bot_elem = node_to_element(thick_bot, triangles)

# Exclude elements outside Puget Sound.
DO_elem[:, ~mask_elem] = np.nan
hyp_thick_elem[:, ~mask_elem] = np.nan

# ============================================================
# HYPOXIC AREA: BOTTOM-LAYER DO <= 2 mg/L
# ============================================================

hypoxic_elem = DO_elem <= 2.0

# km2
hyp_area_ts = np.nansum(
    np.where(
        hypoxic_elem,
        area_km2[None, :], # adding None introduces time compatible dimension
        0.0
    ),
    axis=1
)

# ============================================================
# HYPOXIC WATER VOLUME
# ============================================================

# hyp_thick is the thickness of water with DO <= threshold.
# Units: m * km2 / 1000 = km3.
hyp_volume_ts = np.nansum(
    np.where(
        np.isfinite(hyp_thick_elem),
        hyp_thick_elem * area_km2[None, :],
        0.0
    ),
    axis=1
) / 1e3

# ============================================================
# TOTAL WATER VOLUME
# ============================================================

# Estimate the total volume by integrating depth over element area.
total_volume_km3 = np.nansum(
    depth_elem[0] * area_km2
) / 1e3

hyp_volume_pct = (
    100 * hyp_volume_ts / total_volume_km3
)

print(f'Total SSM water volume: {total_volume_km3:.2f} km3')
print(f'Maximum hypoxic area: {np.nanmax(hyp_area_ts):.2f} km2')
print(f'Maximum hypoxic volume: {np.nanmax(hyp_volume_ts):.4f} km3')

# ============================================================
# HYPOXIA OCCURRENCE
# ============================================================

# Percentage of valid daily time steps with hypoxic bottom DO.
hyp_count = np.sum(
    hypoxic_elem & np.isfinite(DO_elem),
    axis=0
)

valid_count = np.sum(
    np.isfinite(DO_elem),
    axis=0
)

hyp_occurrence = np.full(n_elem, np.nan)

np.divide(
    100 * hyp_count,
    valid_count,
    out=hyp_occurrence,
    where=valid_count > 0
)

# ============================================================
# MEAN BOTTOM-LAYER DO
# ============================================================

mean_DO_elem = np.nanmean(DO_elem, axis=0)

# ============================================================
# PLOT HYPOXIC AREA AND VOLUME TIME SERIES
# ============================================================

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(time_SSM, hyp_area_ts, color='green', linewidth=2)

ax.set_ylabel('Hypoxic bottom-layer area (km$^2$)')
ax.set_xlabel('Date')
ax.set_title('SSM: Bottom-layer hypoxic area (DO <= 2 mg/L)')
ax.grid(True, linestyle='--', color='silver')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=30)

fig.savefig(
    out_dir / 'SSM_hypoxic_area.png',
    dpi=300, bbox_inches='tight'
)
plt.show()

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(time_SSM, hyp_volume_ts, color='green', linewidth=2)

ax.set_ylabel('Hypoxic water volume (km$^3$)')
ax.set_xlabel('Date')
ax.set_title('SSM: Hypoxic water volume (DO <= 2 mg/L)')
ax.grid(True, linestyle='--', color='silver')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=30)

fig.savefig(
    out_dir / 'SSM_hypoxic_volume.png',
    dpi=300, bbox_inches='tight'
)
plt.show()

# hypoxic volume as percent of total volume
fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(time_SSM, hyp_volume_pct, color='green', linewidth=2)

ax.set_ylabel('Hypoxic volume (% of total volume)')
ax.set_xlabel('Date')
ax.set_title('SSM: Hypoxic volume fraction')
ax.grid(True, linestyle='--', color='silver')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=30)

fig.savefig(
    out_dir / 'SSM_hypoxic_volume_percent.png',
    dpi=300, bbox_inches='tight'
)
plt.show()

# ============================================================
# TRIANGULAR MESH MAPS
# ============================================================

triang = mtri.Triangulation(
    lon, lat, triangles=triangles
)

# Mask triangles outside Puget Sound.
triang.set_mask(~mask_elem)

# Common spatial extent based on the Puget Sound basin mask.
ps_lon = mask_lon[mask_ps == 1]
ps_lat = mask_lat[mask_ps == 1]

lon_min, lon_max = np.nanmin(ps_lon), np.nanmax(ps_lon)
lat_min, lat_max = np.nanmin(ps_lat), np.nanmax(ps_lat)

lon_pad = 0.02
lat_pad = 0.02

# Hypoxia occurrence
fig, ax = plt.subplots(figsize=(9, 8))

pcm = ax.tripcolor(
    triang,
    facecolors=hyp_occurrence,
    shading='flat',
    cmap='jet',
    vmin=0,
    vmax=100
)

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('Days hypoxic (%)')

ax.set_title('SSM: Bottom-layer hypoxia occurrence')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

pfun.add_coast(ax, color='black', linewidth=0.8)

fig.savefig(
    out_dir / 'SSM_hypoxia_occurrence.png',
    dpi=300, bbox_inches='tight'
)
plt.show()

# Mean bottom-layer DO
fig, ax = plt.subplots(figsize=(9, 8))

pcm = ax.tripcolor(
    triang,
    facecolors=mean_DO_elem,
    shading='flat',
    cmap='viridis_r'
)

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('Mean bottom-layer DO (mg/L)')

ax.set_title('SSM: Mean bottom-layer DO')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

pfun.add_coast(ax, color='black', linewidth=0.8)

fig.savefig(
    out_dir / 'SSM_mean_bottom_DO.png',
    dpi=300, bbox_inches='tight'
)
plt.show()

ds_SSM.close()