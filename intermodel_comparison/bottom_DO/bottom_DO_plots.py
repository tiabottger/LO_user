"""
This script calculates hypoxic area and volume for LiveOcean and SSC output,
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
# SSC CELL AREA
#
# so estimate horizontal area from lon/lat.
# ============================================================

R = 6371000.0

lat_rad = np.deg2rad(lat_SSC)
lon_rad = np.deg2rad(lon_SSC)

dlat = np.gradient(lat_rad, axis=0)
dlon = np.gradient(lon_rad, axis=1)

DY_SSC = R * np.abs(dlat)
DX_SSC = (
    R *
    np.abs(dlon) *
    np.cos(lat_rad)
)

DA_SSC = DX_SSC * DY_SSC * 1e-6  # m2 to km2

DA_SSC = DA_SSC * mask_ps_SSC 

# ============================================================
# LIVE OCEAN
# Bottom 14.6% of water column
# ============================================================

DO_LO = ds_LO['DO_bot146'].values
hyp_thick_LO = ds_LO['hyp_thick'].values

# Hypoxic cells
hypoxic_LO = DO_LO <= 2.0

# ------------------------------------------------------------
# Hypoxic area
# ------------------------------------------------------------

# DA_ps is already masked to Puget Sound, so multiplying by DA applies mask
hyp_area_LO_ts = np.sum(
    hypoxic_LO * DA_ps[None, :, :],
    axis=(1, 2)
)

# ------------------------------------------------------------
# Hypoxic volume
# ------------------------------------------------------------

# hyp_thick = m
# DA_ps = km2
# m × km2 = 10^6 m3
hyp_volume_LO_ts = np.nansum(
    hypoxic_LO *
    hyp_thick_LO *
    DA_ps[None, :, :],
    axis=(1, 2)
)

# ============================================================
# SALISHSEACAST
# Bottom 14.6% of water column
# ============================================================

DO_SSC = ds_SSC['DO_bot146'].values
hyp_thick_SSC = ds_SSC['hyp_thick'].values

# Hypoxic cells
hypoxic_SSC = DO_SSC <= 2.0

# ------------------------------------------------------------
# Hypoxic area
# ------------------------------------------------------------

# DA_SSC is already masked to Puget Sound
hyp_area_SSC_ts = np.sum(
    hypoxic_SSC *
    DA_SSC[None, :, :],
    axis=(1, 2)
)

# ------------------------------------------------------------
# Hypoxic volume
# ------------------------------------------------------------

# hyp_thick = m
# DA_SSC = km2
# m × km2 = 10^6 m3
hyp_volume_SSC_ts = np.nansum(
    hypoxic_SSC *
    hyp_thick_SSC *
    DA_SSC[None, :, :],
    axis=(1, 2)
)
       
# ============================================================
# PLOT HYPOXIC AREA AND VOLUME TS
# ============================================================

# ------------------------------------------------------------
# Hypoxic area
# ------------------------------------------------------------

print('Plotting hypoxic area time series')

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(
    pd.to_datetime(ds_LO['ocean_time'].values),
    hyp_area_LO_ts, linewidth=2, color= 'red', label='LiveOcean'
)

ax.plot(
    pd.to_datetime(ds_LO['ocean_time'].values),
    hyp_area_SSC_ts, linewidth=2, color= 'blue', label='SalishSeaCast'
)

ax.set_ylabel(
    'Hypoxic Area (km$^2$)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.grid(visible=True, axis='both', color='silver', linestyle='--')
ax.legend(loc='upper right')

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m')
)

ax.tick_params(
    axis='x', rotation=30)
ax.tick_params(
    axis='both', labelsize=11)

ax.set_title('Hypoxic Area — Bottom 14.6%', fontsize=13)

plt.savefig(
    'hypoxic_area_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ------------------------------------------------------------
# Hypoxic volume
# ------------------------------------------------------------

print('Plotting hypoxic volume time series')

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(
    pd.to_datetime(ds_LO['ocean_time'].values),
    hyp_volume_LO_ts, linewidth=2, color= 'red', label='LiveOcean'
)

ax.plot(
    pd.to_datetime(ds_LO['ocean_time'].values),
    hyp_volume_SSC_ts, linewidth=2, color= 'blue', label='SalishSeaCast'
)

ax.set_ylabel(
    'Hypoxic Volume (km$^3$)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.grid(visible=True, axis='both', color='silver', linestyle='--')
ax.legend(loc='upper right')

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m')
)

ax.tick_params(
    axis='x', rotation=30)
ax.tick_params(
    axis='both', labelsize=11)

ax.set_title('Hypoxic Volume — Bottom 14.6%', fontsize=13)

plt.savefig(
    'hypoxic_volume_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ============================================================
# HYPOXIA OCCURRENCE
# ============================================================

# ------------------------------------------------------------
# LiveOcean
# ------------------------------------------------------------

# Coordinates must match the bottom_DO / Puget Sound box grid
lon_LO = box_ds['lon_rho'].values
lat_LO = box_ds['lat_rho'].values

DO_LO = ds_LO['DO_bot146'].values

# Hypoxic = DO <= 2 mg/L
hypoxic_LO = DO_LO <= 2.0
# hypoxic_LO = DO_LO <= 3.0

# Count hypoxic occurrences
hyp_count_LO = np.sum(hypoxic_LO, axis=0)

# Count valid model output times
valid_count_LO = np.sum(np.isfinite(DO_LO), axis=0)

# Occurrence as percentage of valid time steps
hyp_occurrence_LO = np.full_like(
    valid_count_LO,
    np.nan,
    dtype=float
)

np.divide(
    100 * hyp_count_LO,
    valid_count_LO,
    out=hyp_occurrence_LO,
    where=valid_count_LO > 0
)

# ------------------------------------------------------------
# SSC
# ------------------------------------------------------------

DO_SSC = ds_SSC['DO_bot146'].values

hypoxic_SSC = DO_SSC <= 2.0
# hypoxic_SSC = DO_SSC <= 3.0

hyp_count_SSC = np.sum(hypoxic_SSC, axis=0)

valid_count_SSC = np.sum(np.isfinite(DO_SSC), axis=0)

hyp_occurrence_SSC = np.full_like(
    valid_count_SSC,
    np.nan,
    dtype=float
)

np.divide(
    100 * hyp_count_SSC,
    valid_count_SSC,
    out=hyp_occurrence_SSC,
    where=valid_count_SSC > 0
)


# ============================================================
# PLOT HYPOXIA OCCURRENCE
# ============================================================

print('Plotting hypoxic ocurrence in bottom 14.6% of water column')

occurrence_max = np.nanmax([
    np.nanmax(hyp_occurrence_LO),
    np.nanmax(hyp_occurrence_SSC)
])

hyp_occurrence_LO_plot = np.where(
    mask_ps == 1,
    hyp_occurrence_LO,
    np.nan
)

hyp_occurrence_SSC_plot = np.where(
    mask_ps_SSC,
    hyp_occurrence_SSC,
    np.nan
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 7),
    constrained_layout=True
)


# ------------------------------------------------------------
# LiveOcean
# ------------------------------------------------------------

pcm1 = axes[0].pcolormesh(
    lon_LO,
    lat_LO,
    # hyp_occurrence_LO, # as percentage
    hyp_occurrence_LO_plot,
    shading='auto',
    cmap= 'jet',
    vmin=0,
    # vmax=occurrence_max
    vmax= 80
)

axes[0].set_title(
    'LiveOcean — Hypoxia Occurrence DO ≤ 2.0 mg/L',
    #'LiveOcean — Hypoxia Occurrence DO ≤ 3.0 mg/L',
    fontsize=14
)

axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

cbar = fig.colorbar(
    pcm1,
    ax=axes[0]
)

cbar.set_label(
    #'Hypoxic occurrence (%)'
    'Hypoxic days count'
)

# ------------------------------------------------------------
# SSC
# ------------------------------------------------------------

pcm2 = axes[1].pcolormesh(
    lon_SSC,
    lat_SSC,
    # hyp_occurrence_SSC,
    hyp_occurrence_SSC_plot,
    shading='auto',
    cmap= 'jet',
    vmin=0,
    # vmax=occurrence_max
    vmax = 80
)

axes[1].set_title(
    'SalishSeaCast — Hypoxia Occurrence DO ≤ 2.0 mg/L',
    #'SalishSeaCast — Hypoxia Occurrence DO ≤ 3.0 mg/L',
    fontsize=14
)

axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')

cbar = fig.colorbar(
    pcm2,
    ax=axes[1]
)

cbar.set_label(
    #'Hypoxic occurrence (%)'
    'Hypoxic days count'
)


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


plt.savefig(
    'hypoxic_occurrence_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

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