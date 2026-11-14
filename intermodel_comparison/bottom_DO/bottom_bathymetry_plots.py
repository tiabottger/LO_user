"""
This script compares bottom bathymetry for LiveOcean and SSC,
reading netCDF files from LO_output/intermodel_comparison/bottom_DO

This script searches for yearly box extractions in LO_output, for the
region "pugetsoundDO"
"""

# import things
import numpy as np
from scipy.stats import triang
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree
from lo_tools import Lfun

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep

import cmocean

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

# Coordinates must match the bottom_DO / Puget Sound box grid
lon_LO = box_ds['lon_rho'].values
lat_LO = box_ds['lat_rho'].values

# ============================================================
# SSC GRID
# ============================================================

lon_SSC = ds_SSC['nav_lon'].values
lat_SSC = ds_SSC['nav_lat'].values

# ============================================================
# SSM GRID
# ============================================================

lon_SSM = ds_SSM['lon'].values
lat_SSM = ds_SSM['lat'].values

# ============================================================
# SSM MESH FOOTPRINT
# ============================================================

nv = ds_SSM['nv'].values.copy()

# Check whether connectivity is zero-based or one-based
if nv.min() == 0 and nv.max() < len(lon_SSM):
    triangles = nv.copy()

elif nv.min() >= 1 and nv.max() <= len(lon_SSM):
    triangles = nv - 1

else:
    raise ValueError('Cannot safely infer connectivity indexing')

# Construct triangular polygons separately
SSM_polygons = []

for nodes in triangles:

    triangle_lon = lon_SSM[nodes]
    triangle_lat = lat_SSM[nodes]

    if (
        np.all(np.isfinite(triangle_lon)) and
        np.all(np.isfinite(triangle_lat))
    ):
        SSM_polygons.append(
            Polygon(zip(triangle_lon, triangle_lat))
        )

# Combine triangles into one mesh footprint
SSM_footprint = unary_union(SSM_polygons)

# Prepare geometry for efficient point-in-polygon testing
SSM_footprint_prepared = prep(SSM_footprint)

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
# MAP Puget Sound MASK ONTO SSM GRID
# ============================================================

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

mask_node_SSM_ps = mask_ps[mask_j, mask_i] == 1

# Classify an element as Puget Sound if at least two of
# its three vertices fall within the Puget Sound mask.
mask_elem_SSM_ps = (
    mask_node_SSM_ps[triangles].sum(axis=1) >= 2
)

triang = mtri.Triangulation(
    lon_SSM, lat_SSM, triangles=triangles
)

# Mask triangles outside Puget Sound
triang.set_mask(~mask_elem_SSM_ps)

# ============================================================
# BATHYMETRY — BOTTOM DEPTH
# ============================================================

depth_LO = ds_LO['depth_bot'].values
depth_SSC = ds_SSC['depth_bot'].values
depth_SSM = ds_SSM['depth_bot'].values

depth_elem_SSM = node_to_element(
    depth_SSM,
    triangles
)

# Apply Puget Sound masks
depth_LO_plot = np.where(mask_ps == 1, depth_LO, np.nan)
depth_SSC_plot = np.where(mask_ps_SSC, depth_SSC, np.nan)

depth_SSM_plot = np.where(
    mask_elem_SSM_ps,
    depth_elem_SSM,
    np.nan
)

# Common color scale
depth_min = np.nanmin([
    np.nanmin(depth_LO_plot),
    np.nanmin(depth_SSC_plot),
    np.nanmin(depth_SSM_plot)
])

depth_max = np.nanmax([
    np.nanmax(depth_LO_plot),
    np.nanmax(depth_SSC_plot),
    np.nanmax(depth_SSM_plot)
])

print("LiveOcean depth range:", np.nanmin(depth_LO_plot),
      np.nanmax(depth_LO_plot))

print("SalishSeaCast depth range:", np.nanmin(depth_SSC_plot),
      np.nanmax(depth_SSC_plot))

print("Salish Sea Model depth range:", np.nanmin(depth_SSM_plot),
      np.nanmax(depth_SSM_plot))

# ============================================================
# PLOT BATHYMETRY
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 6),
    constrained_layout=True
)

# ------------------------------------------------------------
# LiveOcean
# ------------------------------------------------------------

pcm1 = axes[0].pcolormesh(
    lon_LO,
    lat_LO,
    depth_LO_plot,
    shading='auto',
    cmap=cmocean.cm.deep,
    vmin=depth_min,
    vmax=depth_max
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
    depth_SSC_plot,
    shading='auto',
    cmap=cmocean.cm.deep,
    vmin=depth_min,
    vmax=depth_max
)

axes[1].set_title('SalishSeaCast')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')

# ------------------------------------------------------------
# Salish Sea Model
# ------------------------------------------------------------

pcm3 = axes[2].tripcolor(
    triang,
    facecolors=depth_SSM_plot,
    shading='flat',
    cmap=cmocean.cm.deep,
    vmin=depth_min,
    vmax=depth_max
)

axes[2].set_title('Salish Sea Model')
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('Latitude')

# ------------------------------------------------------------
# Shared colorbar
# ------------------------------------------------------------

cbar = fig.colorbar(
    pcm3,
    ax=axes,
    shrink=0.85,
    pad=0.02
)

cbar.set_label('Bottom depth (m)')

# ------------------------------------------------------------
# MAP EXTENT — PUGET SOUND ONLY
# ------------------------------------------------------------

ps_lon = mask_lon[mask_ps == 1]
ps_lat = mask_lat[mask_ps == 1]

lon_min = np.nanmin(ps_lon)
lon_max = np.nanmax(ps_lon)
lat_min = np.nanmin(ps_lat)
lat_max = np.nanmax(ps_lat)

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
    
# Dark grey background
fig.patch.set_facecolor('0.35')

for ax in axes:
    ax.set_facecolor('0.35')

plt.savefig(
    'bathymetry_bottom_depth_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# MAP SSC BATHYMETRY ONTO LIVE OCEAN GRID FOR DIFFERENCE CALCULATION
# ============================================================

# ------------------------------------------------------------
# Build nearest-neighbor tree from SSC grid
# ------------------------------------------------------------

ssc_points = np.column_stack((
    lon_SSC.ravel(),
    lat_SSC.ravel()
))

valid_ssc = (
    np.isfinite(ssc_points[:, 0]) &
    np.isfinite(ssc_points[:, 1])
)

tree_SSC = cKDTree(ssc_points[valid_ssc])

# ------------------------------------------------------------
# Find nearest SSC cell for every LiveOcean cell
# ------------------------------------------------------------

lo_points = np.column_stack((
    lon_LO.ravel(),
    lat_LO.ravel()
))

_, nearest_index = tree_SSC.query(lo_points)

valid_flat_indices = np.flatnonzero(valid_ssc)
nearest_flat_index = valid_flat_indices[nearest_index]

# ============================================================
# MAP SSC BATHYMETRY ONTO LIVE OCEAN GRID
# ============================================================

ssc_points = np.column_stack((
    lon_SSC.ravel(),
    lat_SSC.ravel()
))

valid_ssc = (
    np.isfinite(ssc_points[:, 0]) &
    np.isfinite(ssc_points[:, 1])
)

tree_SSC = cKDTree(ssc_points[valid_ssc])

lo_points = np.column_stack((
    lon_LO.ravel(),
    lat_LO.ravel()
))

# Find nearest SSC cell for every LO cell
distance_to_SSC, nearest_index = tree_SSC.query(lo_points)

valid_flat_indices = np.flatnonzero(valid_ssc)
nearest_flat_index = valid_flat_indices[nearest_index]

# Map SSC bathymetry onto LO grid
depth_SSC_on_LO = (
    depth_SSC.ravel()[nearest_flat_index]
    .reshape(depth_LO.shape)
)

# ============================================================
# MAP SSM BATHYMETRY ONTO LIVE OCEAN GRID
# ============================================================

ssm_points = np.column_stack((
    lon_SSM,
    lat_SSM
))

valid_ssm = (
    np.isfinite(lon_SSM) &
    np.isfinite(lat_SSM) &
    np.isfinite(depth_SSM)
)

tree_SSM = cKDTree(ssm_points[valid_ssm])

lo_points = np.column_stack((
    lon_LO.ravel(),
    lat_LO.ravel()
))

# Find nearest SSM cell for every LO cell
distance_to_SSM, nearest_index = tree_SSM.query(lo_points)

valid_flat_indices_ssm = np.flatnonzero(valid_ssm)
nearest_flat_index_ssm = valid_flat_indices_ssm[nearest_index]

# Map SSM bathymetry onto LO grid
depth_SSM_on_LO = (
    depth_SSM[nearest_flat_index_ssm]
    .reshape(depth_LO.shape)
)

# ============================================================
# IDENTIFY LO CELLS WITHOUT SSC BATHYMETRY
# ============================================================

# LO wet cells where mapped SSC bathymetry is missing
missing_SSC = (
    np.isfinite(depth_LO) &
    ~np.isfinite(depth_SSC_on_LO) &
    (mask_ps == 1)
)

print(
    'LO wet cells missing from SSC:',
    np.sum(missing_SSC)
)

# Assign zero SSC bathymetry to these cells
depth_SSC_on_LO[missing_SSC] = 0

# ============================================================
# BATHYMETRY DIFFERENCE
# ============================================================

depth_difference = depth_LO - depth_SSC_on_LO

depth_difference_plot = np.where(
    mask_ps == 1,
    depth_difference,
    np.nan
)

depth_difference_LO_SSM = depth_LO - depth_SSM_on_LO

depth_difference_LO_SSM_plot = np.where(
    mask_ps == 1,
    depth_difference_LO_SSM,
    np.nan
)

# ============================================================
# BATHYMETRY DIFFERENCE
# ============================================================

depth_difference = depth_LO - depth_SSC_on_LO

depth_difference_plot = np.where(
    mask_ps == 1,
    depth_difference,
    np.nan
)

print(
    'LO bathymetry in missing SSC cells:',
    np.nanmin(depth_LO[missing_SSC]),
    np.nanmax(depth_LO[missing_SSC])
)

print(
    'SSC bathymetry in missing SSC cells:',
    np.nanmin(depth_SSC_on_LO[missing_SSC]),
    np.nanmax(depth_SSC_on_LO[missing_SSC])
)

print('Total missing SSC cells:', np.sum(missing_SSC))

print(
    'Missing SSC cells inside Puget Sound mask:',
    np.sum(missing_SSC & (mask_ps == 1))
)

print(
    'Missing SSC cells outside Puget Sound mask:',
    np.sum(missing_SSC & (mask_ps == 0))
)

print(
    'Difference in missing SSC cells:',
    np.nanmin(depth_difference[missing_SSC]),
    np.nanmax(depth_difference[missing_SSC])
)

print(
    'Difference plot in missing SSC cells:',
    np.sum(np.isfinite(depth_difference_plot[missing_SSC]))
)


# ============================================================
# PLOT BATHYMETRY DIFFERENCE
# ============================================================

# LO - SSC
fig, ax = plt.subplots(
    figsize=(8, 7),
    constrained_layout=True
)

# Use a symmetric color scale around zero
diff_max = np.nanmax(np.abs(depth_difference_plot))

pcm = ax.pcolormesh(
    lon_LO,
    lat_LO,
    depth_difference_plot,
    shading='auto',
    cmap='RdBu_r',
    vmin=-diff_max,
    vmax=diff_max
)

# Overlay cross-hatching where LO cells are absent from SSC
ax.contourf(
    lon_LO,
    lat_LO,
    missing_SSC.astype(int),
    levels=[0.5, 1.5],
    colors='none',
    hatches=['////']
)

ax.set_title('LiveOcean − SalishSeaCast Bathymetry')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

cbar = fig.colorbar(
    pcm,
    ax=ax,
    shrink=0.85,
    pad=0.02
)

cbar.set_label('Depth difference (m)')

# ------------------------------------------------------------
# MAP EXTENT — PUGET SOUND ONLY
# ------------------------------------------------------------

ps_lon = mask_lon[mask_ps == 1]
ps_lat = mask_lat[mask_ps == 1]

lon_min = np.nanmin(ps_lon)
lon_max = np.nanmax(ps_lon)
lat_min = np.nanmin(ps_lat)
lat_max = np.nanmax(ps_lat)

lon_pad = 0.02
lat_pad = 0.02

ax.set_xlim(
    lon_min - lon_pad,
    lon_max + lon_pad
)

ax.set_ylim(
    lat_min - lat_pad,
    lat_max + lat_pad
)

plt.savefig(
    'bathymetry_difference_LO_minus_SSC.png',
    dpi=300,
    bbox_inches='tight'
)

print('Figure saved to:', Path('bathymetry_difference_LO_minus_SSC.png').resolve())

# LO - SSM
fig, ax = plt.subplots(figsize=(10, 8))

pcm = ax.pcolormesh(
    lon_LO,
    lat_LO,
    depth_difference_LO_SSM_plot,
    cmap='RdBu_r',
    shading='auto',
    vmin=-diff_max,
    vmax=diff_max
)

plt.colorbar(
    pcm,
    ax=ax,
    label='LO - SSM bottom depth (m)'
)

ax.set_title('Bottom Bathymetry Difference: LO - SSM')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

ax.set_xlim(
    lon_min - lon_pad,
    lon_max + lon_pad
)

ax.set_ylim(
    lat_min - lat_pad,
    lat_max + lat_pad
)

plt.savefig(
    'bathymetry_difference_LO_minus_SSM.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
print('Figure saved to:', Path('bathymetry_difference_LO_minus_SSM.png').resolve())

# ============================================================
# BOTTOM 14.6% THICKNESS
# ============================================================

thick_LO = ds_LO['thick_bot146'].values
thick_SSC = ds_SSC['thick_bot146'].values

# ============================================================
# MAP SSC THICKNESS ONTO LIVE OCEAN GRID
# ============================================================

# SSC grid points
ssc_points = np.column_stack((
    lon_SSC.ravel(),
    lat_SSC.ravel()
))

valid_ssc = (
    np.isfinite(ssc_points[:, 0]) &
    np.isfinite(ssc_points[:, 1])
)

tree_SSC = cKDTree(ssc_points[valid_ssc])

# LiveOcean grid points
lo_points = np.column_stack((
    lon_LO.ravel(),
    lat_LO.ravel()
))

distance_to_SSC, nearest_index = tree_SSC.query(lo_points)

valid_flat_indices = np.flatnonzero(valid_ssc)
nearest_flat_index = valid_flat_indices[nearest_index]

# Identify LO cells not represented by SSC
missing_SSC = (
    distance_to_SSC.reshape(depth_LO.shape) > 0.01
)

# ------------------------------------------------------------
# Map every SSC timestep onto LO grid
# ------------------------------------------------------------

thick_SSC_on_LO = np.empty_like(thick_LO)

for t in range(thick_SSC.shape[0]):

    thick_SSC_on_LO[t, :, :] = (
        thick_SSC[t, :, :].ravel()[nearest_flat_index]
        .reshape(thick_LO.shape[1:])
    )

# ------------------------------------------------------------
# Assign zero thickness where SSC has no grid coverage
# ------------------------------------------------------------

thick_SSC_on_LO[:, missing_SSC] = 0

print(
    'LO cells absent from SSC:',
    np.sum(missing_SSC)
)

# ============================================================
# THICKNESS DIFFERENCE
# ============================================================

thick_difference = thick_LO - thick_SSC_on_LO

mean_thick_difference = np.nanmean(
    thick_difference,
    axis=0
)

# Apply Puget Sound mask
mean_thick_difference_plot = np.where(
    mask_ps == 1,
    mean_thick_difference,
    np.nan
)

# ============================================================
# PLOT THICK_BOT146 DIFFERENCE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7),
    constrained_layout=True
)

# Symmetric color scale centered at zero
diff_max = np.nanmax(np.abs(mean_thick_difference_plot))

pcm = ax.pcolormesh(
    lon_LO,
    lat_LO,
    mean_thick_difference_plot,
    shading='auto',
    cmap='RdBu_r',
    vmin=-diff_max,
    vmax=diff_max
)

ax.set_title(
    'LiveOcean − SalishSeaCast\n'
    'Mean Bottom 14.6% Layer Thickness Difference'
)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Colorbar
cbar = fig.colorbar(
    pcm,
    ax=ax,
    shrink=0.85,
    pad=0.02
)

cbar.set_label(
    'thick_bot146 difference (m)'
)

# ============================================================
# MAP EXTENT — PUGET SOUND ONLY
# ============================================================

ps_lon = mask_lon[mask_ps == 1]
ps_lat = mask_lat[mask_ps == 1]

lon_min = np.nanmin(ps_lon)
lon_max = np.nanmax(ps_lon)
lat_min = np.nanmin(ps_lat)
lat_max = np.nanmax(ps_lat)

lon_pad = 0.02
lat_pad = 0.02

ax.set_xlim(
    lon_min - lon_pad,
    lon_max + lon_pad
)

ax.set_ylim(
    lat_min - lat_pad,
    lat_max + lat_pad
)

plt.savefig(
    'thick_bot146_difference_LO_minus_SSC.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print('Figure saved to:', Path('thick_bot146_difference_LO_minus_SSC.png').resolve())