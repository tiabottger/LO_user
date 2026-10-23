"""
This script calculates hypoxic area and volume for LiveOcean SSC and SSM output,
reading netCDF files from LO_output/intermodel_comparison/bottom_DO

This script searches for yearly box extractions in LO_output, for the
region "pugetsoundDO". It uses the file created by get_bottom_DO_nc.py.
"""

# import things
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import FuncNorm, ListedColormap
from scipy.spatial import cKDTree
from pyproj import Transformer
from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
import cmocean
import cmocean.tools

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
ds_LO = xr.open_dataset(
    in_dir / 'cas7_t1_x11ab_pugetsound_2014_bottom_DO_info.nc'
)
ds_SSC = xr.open_dataset(
    in_dir / 'SSC_2014_pugetsound_bottom_DO_info.nc'
)

ds_SSM= xr.open_dataset(
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
# SSM CELL AREA
# ============================================================
DO_SSM = ds_SSM['DO_bot'].values
hyp_thick_SSM = ds_SSM['hyp_thick'].values
thick_bot_SSM = ds_SSM['thick_bot'].values
depth_bot_SSM = ds_SSM['depth_bot'].values

n_time, n_node = DO_SSM.shape
n_elem = ds_SSM.sizes['elem']

lon_SSM = ds_SSM['lon'].values
lat_SSM = ds_SSM['lat'].values

# ------------------------------------------------------------
# TRIANGULAR MESH CONNECTIVITY
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# PROJECT COORDINATES TO METERS
# ------------------------------------------------------------

# UTM zone 10N is suitable for this Puget Sound region.
transformer = Transformer.from_crs(
    'EPSG:4326',
    'EPSG:26910',
    always_xy=True
)

x, y = transformer.transform(lon_SSM, lat_SSM)

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

ssm_points = np.column_stack((lon_SSM, lat_SSM))
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


DO_elem = node_to_element(DO_SSM, triangles)
hyp_thick_elem = node_to_element(hyp_thick_SSM, triangles)
depth_elem = node_to_element(depth_bot_SSM, triangles)
thick_bot_elem = node_to_element(thick_bot_SSM, triangles)

# Exclude elements outside Puget Sound.
DO_elem[:, ~mask_elem] = np.nan
hyp_thick_elem[:, ~mask_elem] = np.nan

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
hyp_volume_LO_ts = np.nansum(
    hypoxic_LO *
    hyp_thick_LO *
    DA_ps[None, :, :],
    axis=(1, 2) 
)/1e3

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
hyp_volume_SSC_ts = np.nansum(
    hypoxic_SSC *
    hyp_thick_SSC *
    DA_SSC[None, :, :],
    axis=(1, 2)
)/ 1e3 # convert m hyp thick to km --> km3

# ============================================================
# SALISH SEA MODEL
# Bottom cell
# ============================================================

# ------------------------------------------------------------
# Hypoxic area
# ------------------------------------------------------------

hypoxic_elem = DO_elem <= 2.0

# km2
hyp_area_SSM_ts = np.nansum(
    np.where(
        hypoxic_elem,
        area_km2[None, :], # adding None introduces time compatible dimension
        0.0
    ),
    axis=1
)

# ------------------------------------------------------------
# Hypoxic volume
# ------------------------------------------------------------

# hyp_thick is the thickness of water with DO <= threshold.
# Units: m * km2 / 1000 = km3.
hyp_volume_SSM_ts = np.nansum(
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

# ------------------------------------------------------------
# LIVE OCEAN
# ------------------------------------------------------------

depth_bot_LO = ds_LO['depth_bot'].values

total_volume_LO = np.nansum(
    depth_bot_LO * DA_ps
) / 1e3   # m * km2 -> km3


# ------------------------------------------------------------
# SALISHSEACAST
# ------------------------------------------------------------

depth_bot_SSC = ds_SSC['depth_bot'].values

total_volume_SSC = np.nansum(
    depth_bot_SSC * DA_SSC
) / 1e3   # m * km2 -> km3

# ------------------------------------------------------------
# SALISH SEA MODEL
# ------------------------------------------------------------

total_volume_SSM = np.nansum(
    depth_elem[0] * area_km2
) / 1e3


print(f'LiveOcean total volume: {total_volume_LO:.2f} km3')
print(f'SalishSeaCast total volume: {total_volume_SSC:.2f} km3')
print(f'Salish Sea Model total volume: {total_volume_SSM:.2f} km3')

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

ax.plot(
    pd.to_datetime(ds_SSM['time'].values), 
    hyp_area_SSM_ts, linewidth=2, color='green', label='Salish Sea Model')

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
    out_dir / 'hypoxic_area_comparison.png',
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

ax.plot(
    pd.to_datetime(pd.to_datetime(ds_SSM['time'].values)),
    hyp_volume_SSM_ts, linewidth=2, color= 'green', label='Salish Sea Model'
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

ax.set_title('Hypoxic Volume', fontsize=13)

plt.savefig(
    out_dir / 'hypoxic_volume_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ============================================================
# PLOT HYPOXIC VOLUME (% OF TOTAL VOLUME)
# ============================================================

hyp_volume_pct_LO = (
    hyp_volume_LO_ts / total_volume_LO
) * 100

hyp_volume_pct_SSC = (
    hyp_volume_SSC_ts / total_volume_SSC
) * 100

hyp_volume_pct_SSM = (
    hyp_volume_SSM_ts / total_volume_SSM
) * 100

print('Plotting hypoxic volume as percent of total volume')

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(
    pd.to_datetime(ds_LO['ocean_time'].values),
    hyp_volume_pct_LO,
    linewidth=2,
    color='red',
    label='LiveOcean'
)

ax.plot(
    pd.to_datetime(ds_LO['ocean_time'].values),
    hyp_volume_pct_SSC,
    linewidth=2,
    color='blue',
    label='SalishSeaCast'
)

ax.plot(
    pd.to_datetime(pd.to_datetime(ds_SSM['time'].values)),
    hyp_volume_pct_SSM,
    linewidth=2,
    color='green',
    label='Salish Sea Model'
)

ax.set_ylabel(
    'Hypoxic Volume (% of Total Volume)',
    fontsize=12
)

ax.set_xlabel('Date', fontsize=12)

ax.grid(
    visible=True,
    axis='both',
    color='silver',
    linestyle='--'
)

ax.legend(loc='upper right')

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m')
)

ax.tick_params(
    axis='x',
    rotation=30
)

ax.tick_params(
    axis='both',
    labelsize=11
)

ax.set_title(
    'Hypoxic Volume (% of Total Volume)',
    fontsize=13
)

plt.savefig(
    out_dir / 'hypoxic_volume_percent_comparison.png',
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
# DO_LO = ds_LO['DO_bot'].values

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
# DO_SSC = ds_SSC['DO_bot'].values

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

# ------------------------------------------------------------
# SSM
# ------------------------------------------------------------

# Percentage of valid daily time steps with hypoxic bottom DO.
hyp_count_SSM = np.sum(
    hypoxic_elem & np.isfinite(DO_elem),
    axis=0
)

valid_count_SSM = np.sum(
    np.isfinite(DO_elem),
    axis=0
)

hyp_occurrence_SSM = np.full(n_elem, np.nan)

np.divide(
    100 * hyp_count_SSM,
    valid_count_SSM,
    out=hyp_occurrence_SSM,
    where=valid_count_SSM > 0
)

# ============================================================
# PLOT HYPOXIA OCCURRENCE
# ============================================================

print('Plotting hypoxic ocurrence in bottom 14.6% of water column')

occurrence_max = np.nanmax([
    np.nanmax(hyp_occurrence_LO),
    np.nanmax(hyp_occurrence_SSC),
    np.nanmax(hyp_occurrence_SSM)
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

hyp_occurrence_SSM_plot = np.where(
    mask_elem,
    hyp_occurrence_SSM,
    np.nan
)

# ------------------------------------------------------------
# Color map
# ------------------------------------------------------------

# Color map
# cmap = "jet"
cmap= plt.cm.jet.copy()
cmap.set_under('white'),


# ------------------------------------------------------------
# SSM triangular mesh
# ------------------------------------------------------------

triang = mtri.Triangulation(
    lon_SSM, lat_SSM, triangles=triangles
)

# Mask triangles outside Puget Sound
triang.set_mask(~mask_elem)

# Common spatial extent
ps_lon = mask_lon[mask_ps == 1]
ps_lat = mask_lat[mask_ps == 1]

lon_min, lon_max = np.nanmin(ps_lon), np.nanmax(ps_lon)
lat_min, lat_max = np.nanmin(ps_lat), np.nanmax(ps_lat)

lon_pad = 0.02
lat_pad = 0.02

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(20, 7),
    constrained_layout=True
)

# ------------------------------------------------------------
# LO
# ------------------------------------------------------------

pcm1 = axes[0].pcolormesh(
    lon_LO,
    lat_LO,
    # hyp_occurrence_LO, # as percentage
    hyp_occurrence_LO_plot,
    shading='auto',
    cmap = cmap,
    vmin=0.001,
    # vmax=occurrence_max
    vmax= 140
)

axes[0].set_title('LO: Bottom Hypoxia (DO ≤ 2.0 mg/L) Occurrence [Days]')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_xlim(lon_min - lon_pad, lon_max + lon_pad)
axes[0].set_ylim(lat_min - lat_pad, lat_max + lat_pad)

pfun.add_coast(axes[0], color='black', linewidth=0.8)

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
    cmap= cmap,
    vmin=0.001,
    # vmax=occurrence_max
    vmax = 140
)

axes[1].set_title('SSC: Bottom Hypoxia (DO ≤ 2.0 mg/L) Occurrence [Days]')
axes[1].set_xlabel('Longitude')
axes[1].set_xlim(lon_min - lon_pad, lon_max + lon_pad)
axes[1].set_ylim(lat_min - lat_pad, lat_max + lat_pad)

pfun.add_coast(axes[1], color='black', linewidth=0.8)

cbar = fig.colorbar(
    pcm2,
    ax=axes[1]
)

cbar.set_label(
    #'Hypoxic occurrence (%)'
    'Hypoxic days count'
)

# ------------------------------------------------------------
# SSM
# ------------------------------------------------------------

pcm3 = axes[2].tripcolor(
    triang,
    facecolors=hyp_occurrence_SSM_plot,
    shading='flat',
    cmap=cmap,
    vmin=0.001,
    # vmax=occurrence_max
    vmax=140
)

axes[2].set_title('SSM: Bottom Hypoxia (DO ≤ 2.0 mg/L) Occurrence [Days]')
axes[2].set_xlabel('Longitude')
axes[2].set_xlim(lon_min - lon_pad, lon_max + lon_pad)
axes[2].set_ylim(lat_min - lat_pad, lat_max + lat_pad)

pfun.add_coast(axes[2], color='black', linewidth=0.8)

cbar = fig.colorbar(
    pcm3,
    ax=axes[2]
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
    out_dir / 'hypoxic_occurrence_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ============================================================
# Average DO concentration in bottom 14.6%
# ============================================================

mean_DO_LO = np.nanmean(ds_LO['DO_bot146'].values, axis=0)
mean_DO_SSC = np.nanmean(ds_SSC['DO_bot146'].values, axis=0)
mean_DO_SSM = np.nanmean(DO_elem, axis=0)

# Apply Puget Sound mask
mean_DO_LO_plot = np.where(mask_ps == 1, mean_DO_LO, np.nan)
mean_DO_SSC_plot = np.where(mask_ps_SSC, mean_DO_SSC, np.nan)
mean_DO_SSM_plot = np.where(mask_elem, mean_DO_SSM, np.nan)

# Common color scale for both models
DO_min = np.nanmin([
    np.nanmin(mean_DO_LO_plot),
    np.nanmin(mean_DO_SSC_plot),
    np.nanmin(mean_DO_SSM_plot)
])

DO_max = np.nanmax([
    np.nanmax(mean_DO_LO_plot),
    np.nanmax(mean_DO_SSC_plot),
    np.nanmax(mean_DO_SSM_plot)
])

print("Mean bottom 14.6% DO range:")
print("Minimum:", DO_min)
print("Maximum:", DO_max)

# ============================================================
# Plot spatially averaged bottom 14.6% DO
# ============================================================

print('Plotting average DO in bottom 14.6% of water column')

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6),
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

pfun.add_coast(axes[0], color='black', linewidth=0.8)

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

pfun.add_coast(axes[1], color='black', linewidth=0.8)

# ------------------------------------------------------------
# Salish Sea Model
# ------------------------------------------------------------

pcm3 = axes[2].tripcolor(
    triang,
    facecolors= mean_DO_SSM_plot,
    shading='flat',
    cmap='viridis_r',
    vmin=DO_min,
    vmax=DO_max
)

axes[2].set_title('Salish Sea Model')
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('Latitude')

pfun.add_coast(axes[2], color='black', linewidth=0.8)
# ------------------------------------------------------------
# Same colorbar for all
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
    out_dir / 'mean_DO_bottom146_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# Daily DO concentration in bottom 14.6% animation
# with hypoxic area time series
# ============================================================

# Apply Puget Sound mask
DO_LO_plot = np.where(
    mask_ps[None, :, :] == 1,
    DO_LO,
    np.nan
)

DO_SSC_plot = np.where(
    mask_ps_SSC[None, :, :],
    DO_SSC,
    np.nan
)

# mask already applied to DO_elem
# LO and SSC had masks applied to DA for hypoxic area and volume calculations, but DO_elem is already masked to Puget Sound elements.
# DO_SSM_plot = np.where(
#     mask_elem[None, :], #2D because [time, element] rather than [time, lat, lon]
#     DO_elem,
#     np.nan
# )

# ------------------------------------------------------------
# Common color scale for both models and all days
# ------------------------------------------------------------

# DO_min = np.nanmin([
#     np.nanmin(DO_LO_plot),
#     np.nanmin(DO_SSC_plot),
#     np.nanmin(DO_elem)
# ])

# DO_max = np.nanmax([
#     np.nanmax(DO_LO_plot),
#     np.nanmax(DO_SSC_plot),
#     np.nanmax(DO_elem)
# ])

# Original oxygen colormap
cmap = cmocean.cm.oxy

# Crop the upper 20% of the colormap (supersaturated colors)
cmap_oxy = cmocean.tools.crop_by_percent(
    cmap,
    20,
    which='max',
    N=None
)

# Common oxygen limits for LO, SSC, and SSM
DO_min = 0.0
DO_max = 20.0  

# Map DO = 2 mg/L to 20% of the colormap
def forward(x):
    return np.where(
        x <= 2.0,
        0.2 * x / 2.0,
        0.2 + 0.8 * (x - 2.0) / (DO_max - 2.0)
    )

def inverse(y):
    return np.where(
        y <= 0.2,
        2.0 * y / 0.2,
        2.0 + (DO_max - 2.0) * (y - 0.2) / 0.8
    )

norm = FuncNorm(
    (forward, inverse),
    vmin=DO_min,
    vmax=DO_max
)

# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

time_LO = pd.to_datetime(ds_LO['ocean_time'].values)
time_SSC = pd.to_datetime(ds_LO['ocean_time'].values)
time_SSM = pd.to_datetime(ds_SSM['time'].values)

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig = plt.figure(figsize=(22, 10))

gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1, 3],
    hspace=0.25,
    wspace=0.05
)

# Time series spans all three columns
ax_ts = fig.add_subplot(gs[0, :])

# Map panels
axes = [
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2])
]
# ------------------------------------------------------------
# Initial frame
# ------------------------------------------------------------

pcm1 = axes[0].pcolormesh(
    lon_LO,
    lat_LO,
    DO_LO_plot[0, :, :],
    shading='auto',
    #cmap='viridis_r',
    cmap=cmap_oxy,
    norm=norm,
    # vmin=DO_min,
    # vmax=DO_max
)

pcm2 = axes[1].pcolormesh(
    lon_SSC,
    lat_SSC,
    DO_SSC_plot[0, :, :],
    shading='auto',
    #cmap='viridis_r',
    cmap=cmap_oxy,
    norm=norm,
    # vmin=DO_min,
    # vmax=DO_max
)

pcm3 = axes[2].tripcolor(
    triang,
    facecolors= DO_elem[0, :],
    shading='flat',
    #cmap='viridis_r',
    cmap=cmap_oxy,
    norm=norm,
    # vmin=DO_min,
    # vmax=DO_max
)
pfun.add_coast(axes[0], color='black', linewidth=0.8)
pfun.add_coast(axes[1], color='black', linewidth=0.8)
pfun.add_coast(axes[2], color='black', linewidth=0.8)

# ------------------------------------------------------------
# Titles and labels
# ------------------------------------------------------------

axes[0].set_title('LiveOcean')
axes[1].set_title('SalishSeaCast')
axes[2].set_title('Salish Sea Model')
for ax in axes:
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

# ------------------------------------------------------------
# Plot hypoxic area time series
# ------------------------------------------------------------

# Use the same dates as the animation frames
time = pd.to_datetime(time_LO)

ax_ts.plot(
    time, hyp_area_LO_ts,
    color='red', label='LiveOcean'
)
ax_ts.plot(
    time, hyp_area_SSC_ts,
    color='blue', label='SalishSeaCast'
)
ax_ts.plot(
    time, hyp_area_SSM_ts,
    color='green', label='SSM'
)

ax_ts.set_ylabel('Hypoxic area (km²)')
ax_ts.set_title('Daily hypoxic area (DO ≤ 2 mg/L)')
ax_ts.legend(loc='upper right', ncol=3)
ax_ts.grid(True, alpha=0.3)

ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax_ts.set_xlim(time[0], time[-1])

# Animated vertical date marker
time_marker = ax_ts.axvline(
    time[0],
    color='black',
    linestyle='-',
    linewidth=2
)

# Moving date label
date_label = ax_ts.text(
    0.01, 0.90, '',
    transform=ax_ts.transAxes,
    ha='left',
    va='top'
)

# ------------------------------------------------------------
# Common colorbar
# ------------------------------------------------------------

cbar = fig.colorbar(
    pcm2,
    ax=axes,
    shrink=0.85,
    pad=0.02
)

cbar.set_label('DO in bottom 14.6% (mg/L)')

# ------------------------------------------------------------
# Align hypoxic area time-series panel with the three map panels
# ------------------------------------------------------------

# Align time-series panel horizontally with the three map panels
fig.canvas.draw()

map_pos_left = axes[0].get_position().x0
map_pos_right = axes[2].get_position().x1
ts_pos = ax_ts.get_position()

ax_ts.set_position([
    map_pos_left,
    ts_pos.y0,
    map_pos_right - map_pos_left,
    ts_pos.height
])

# Keep latitude labels only on the leftmost map
for ax in axes[1:]:
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelleft=False)

# Add a little space between map panels
fig.subplots_adjust(wspace=0.12)

# ============================================================
# Animation function
# ============================================================

def update(frame):
    global pcm3

    # Update LiveOcean
    pcm1.set_array(
        DO_LO_plot[frame, :, :].ravel()
    )

    # Update SalishSeaCast
    pcm2.set_array(
        DO_SSC_plot[frame, :, :].ravel()
    )

    # Remove the previous SSM collection
    pcm3.remove()

    # Recreate the SSM plot for this frame
    pcm3 = axes[2].tripcolor(
        triang,
        facecolors=DO_elem[frame, :],
        shading='flat',
        cmap=cmap_oxy,
        norm=norm
    )
    
    # Move the time-series marker to the current animation date
    current_time = time[frame]

    time_marker.set_xdata([current_time, current_time])
    date_label.set_text(current_time.strftime('%Y-%m-%d'))

    return pcm1, pcm2, pcm3, time_marker, date_label


# ============================================================
# Create animation
# ============================================================

print('Creating animation...')

ani = FuncAnimation(
    fig,
    update,
    frames=len(time),
    interval=150,      # milliseconds between frames
    blit=False
)

# TESTING
# update(180)
# fig.canvas.draw()
# plt.show()
# plt.savefig(
#     out_dir / 'DO_bottom146_frame_test_update.png',
#     dpi=300,
#     bbox_inches='tight'
# )


ani.save(
    out_dir / 'daily_DO_bottom146_comparison.mp4',
    writer='ffmpeg',
    fps=6,
    dpi=200
)

# TESTING 
# frame = 180

# fig_test, ax_test = plt.subplots(figsize=(10, 8))

# pcm_test = ax_test.tripcolor(
#     triang,
#     facecolors=DO_elem[frame, :],
#     shading='flat',
#     cmap='viridis_r',
#     vmin=DO_min,
#     vmax=DO_max
# )

# pfun.add_coast(ax_test, color='black', linewidth=0.8)
# ax_test.set_xlim(lon_min, lon_max)
# ax_test.set_ylim(lat_min, lat_max)
# ax_test.set_title(time_SSM[frame].strftime('%Y-%m-%d'))

# fig_test.colorbar(pcm_test, ax=ax_test, label='DO (mg/L)')
# plt.savefig(
#     out_dir / 'DO_bottom146_frame_test.png',
#     dpi=300,
#     bbox_inches='tight'
# )