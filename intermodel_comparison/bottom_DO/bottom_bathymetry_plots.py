"""
This script compares bottom bathymetry for LiveOcean and SSC,
reading netCDF files from LO_output/intermodel_comparison/bottom_DO

This script searches for yearly box extractions in LO_output, for the
region "pugetsoundDO"
"""

# import things
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from lo_tools import Lfun

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


# ============================================================
# LOAD DATA
# ============================================================

# READ MODEL OUTPUT
ds_LO = xr.open_dataset(
    in_dir / 'cas7_t1_x11ab_pugetsound_2014_bottom_DO_info.nc'
)
ds_SSC = xr.open_dataset(
    in_dir / 'SSC_2014_pugetsound_bottom_DO_info.nc'
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

lon_LO = grid_ds['lon_rho'].values
lat_LO = grid_ds['lat_rho'].values
mask_rho = grid_ds['mask_rho'].values

# ============================================================
# GET CELL AREA FROM ORIGINAL LIVE OCEAN BOX FILE
# ============================================================

# get grid cell area from one of the original box files
fp_grid = Ldir['LOo'] / 'extract' / 'cas7_t1_x11ab' / 'box' / (region + '_2014.01.01_2014.12.31.nc')
box_ds = xr.open_dataset(fp_grid)

DX = box_ds.pm.values ** -1
DY = box_ds.pn.values ** -1
DA = DX * DY * 1e-6  # m2 to km2

# optional: mask out land if mask_rho exists
if 'mask_rho' in box_ds:
    mask_rho = box_ds['mask_rho'].values
else:
    mask_rho = np.ones_like(DA)

DA = DA * mask_rho
DA_ps = DA * mask_ps  # mask out non-Puget Sound areas

# Coordinates must match the bottom_DO / Puget Sound box grid
lon_LO = box_ds['lon_rho'].values
lat_LO = box_ds['lat_rho'].values

# ============================================================
# SSC GRID
# ============================================================

lon_SSC = ds_SSC['nav_lon'].values
lat_SSC = ds_SSC['nav_lat'].values


# ============================================================
# MAP Puget Sound MASK ONTO SSC GRID
# ============================================================

mask_lon = basin_mask_ds['lon_rho'].values
mask_lat = basin_mask_ds['lat_rho'].values

grid_points = np.column_stack((
    mask_lon.ravel(),
    mask_lat.ravel()
))

valid_grid = (
    np.isfinite(grid_points[:, 0]) &
    np.isfinite(grid_points[:, 1])
)

tree = cKDTree(grid_points[valid_grid])

ssc_points = np.column_stack((
    lon_SSC.ravel(),
    lat_SSC.ravel()
))

_, nearest_index = tree.query(ssc_points)

valid_flat_indices = np.flatnonzero(valid_grid)

nearest_flat_index = valid_flat_indices[nearest_index]

eta_index, xi_index = np.unravel_index(
    nearest_flat_index,
    mask_lon.shape
)

mask_ps_SSC = (
    mask_ps[eta_index, xi_index] == 1
).reshape(lon_SSC.shape)

# ============================================================
# BATHYMETRY — BOTTOM DEPTH
# ============================================================

depth_LO = ds_LO['depth_bot'].values
depth_SSC = ds_SSC['depth_bot'].values

# Apply Puget Sound masks
depth_LO_plot = np.where(mask_ps == 1, depth_LO, np.nan)
depth_SSC_plot = np.where(mask_ps_SSC, depth_SSC, np.nan)

# Common color scale
depth_min = np.nanmin([
    np.nanmin(depth_LO_plot),
    np.nanmin(depth_SSC_plot)
])

depth_max = np.nanmax([
    np.nanmax(depth_LO_plot),
    np.nanmax(depth_SSC_plot)
])

print("LiveOcean depth range:", np.nanmin(depth_LO_plot),
      np.nanmax(depth_LO_plot))

print("SalishSeaCast depth range:", np.nanmin(depth_SSC_plot),
      np.nanmax(depth_SSC_plot))

# ============================================================
# COMMON MAP EXTENT
# ============================================================

# for ax in axes.flat:
#     ax.set_xlim(
#         np.nanmin(mask_lon),
#         np.nanmax(mask_lon)
#     )
#     ax.set_ylim(
#         np.nanmin(mask_lat),
#         np.nanmax(mask_lat)
#     )
    
# ============================================================
# MAP EXTENT — PUGET SOUND ONLY
# ============================================================

ps_lon = mask_lon[mask_ps == 1]
ps_lat = mask_lat[mask_ps == 1]

lon_min = np.nanmin(ps_lon)
lon_max = np.nanmax(ps_lon)
lat_min = np.nanmin(ps_lat)
lat_max = np.nanmax(ps_lat)

# Add small padding
lon_pad = 0.02
lat_pad = 0.02

for ax in axes.flat:
    ax.set_xlim(
        lon_min - lon_pad,
        lon_max + lon_pad
    )
    ax.set_ylim(
        lat_min - lat_pad,
        lat_max + lat_pad
    )

# ============================================================
# Average DO concentration in bottom 14.6%
# ============================================================

mean_DO_LO = np.nanmean(ds_LO['DO_bot146'].values, axis=0)
mean_DO_SSC = np.nanmean(ds_SSC['DO_bot146'].values, axis=0)

# Apply Puget Sound mask
mean_DO_LO_plot = np.where(mask_ps == 1, mean_DO_LO, np.nan)
mean_DO_SSC_plot = np.where(mask_ps_SSC, mean_DO_SSC, np.nan)

# Common color scale for both models
DO_min = np.nanmin([
    np.nanmin(mean_DO_LO_plot),
    np.nanmin(mean_DO_SSC_plot)
])

DO_max = np.nanmax([
    np.nanmax(mean_DO_LO_plot),
    np.nanmax(mean_DO_SSC_plot)
])

print("Mean bottom 14.6% DO range:")
print("Minimum:", DO_min)
print("Maximum:", DO_max)

# ============================================================
# Plot spatially averaged bottom 14.6% DO
# ============================================================

print('Plotting average DO in bottom 14.6% of water column')

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 6),
    constrained_layout=True
)

# ------------------------------------------------------------
# LiveOcean
# ------------------------------------------------------------

pcm1 = axes[0].pcolormesh(
    lon_LO,
    lat_LO,
    mean_DO_LO_plot,
    shading='auto',
    cmap='viridis_r',
    vmin=DO_min,
    vmax=DO_max
)

axes[0].set_title('LiveOcean')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

# ------------------------------------------------------------
# SalishSeaCast
# ------------------------------------------------------------

pcm2 = axes[1].pcolormesh(
    lon_SSC,
    lat_SSC,
    mean_DO_SSC_plot,
    shading='auto',
    cmap='viridis_r',
    vmin=DO_min,
    vmax=DO_max
)

axes[1].set_title('SalishSeaCast')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')

# ------------------------------------------------------------
# Same colorbar for both
# ------------------------------------------------------------

cbar = fig.colorbar(
    pcm2,
    ax=axes,
    shrink=0.85,
    pad=0.02
)

cbar.set_label('Mean DO in bottom 14.6% (mg/L)')

# limits from map extent above
for ax in axes:
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

plt.savefig(
    'mean_DO_bottom146_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()