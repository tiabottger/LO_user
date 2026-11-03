"""
This script calculates hypoxic area and volume for SSM PNNL and Ecology output,
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
ds_SSM= xr.open_dataset(
    in_dir / 'SSM_2014_pugetsound_bottom_DO_info.nc'
)

ds_SSM_ecology= xr.open_dataset(
    in_dir / 'SSM_ecology_2014_pugetsound_bottom_DO_info.nc'
)

# READ Puget Sound BASIN MASK
basin_mask_ds = xr.open_dataset(
    Ldir['LOu'] /
    'obsmod' /
    'basin_masks_from_pugetsoundDObox.nc'
)

mask_ps = basin_mask_ds['mask_pugetsound'].values
mask_lon = basin_mask_ds['lon_rho'].values
mask_lat = basin_mask_ds['lat_rho'].values

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
# EXTRACT SSM PNNL 
# ============================================================
DO_SSM = ds_SSM['DO_bot'].values
hyp_thick_SSM = ds_SSM['hyp_thick'].values
thick_bot_SSM = ds_SSM['thick_bot'].values
depth_bot_SSM = ds_SSM['depth_bot'].values

time_SSM = pd.to_datetime(ds_SSM['time'].values)

n_time, n_node = DO_SSM.shape
n_elem = ds_SSM.sizes['elem']

lon_SSM = ds_SSM['lon'].values
lat_SSM = ds_SSM['lat'].values

# ============================================================
# EXTRACT SSM ECOLOGY
# ============================================================
DO_SSM_ecology = ds_SSM_ecology['DO_bot'].values
hyp_thick_SSM_ecology = ds_SSM_ecology['hyp_thick'].values
thick_bot_SSM_ecology = ds_SSM_ecology['thick_bot'].values
depth_bot_SSM_ecology = ds_SSM_ecology['depth_bot'].values

lon_SSM_ecology = ds_SSM_ecology['lon'].values
lat_SSM_ecology = ds_SSM_ecology['lat'].values

time_SSM_ecology = pd.to_datetime(
    ds_SSM_ecology['time'].values
)

print("PNNL node count:", len(lon_SSM))
print("Ecology node count:", len(lon_SSM_ecology))

assert len(lon_SSM) == len(lon_SSM_ecology)

# ============================================================
# MATCH PNNL AND ECOLOGY NODES
# ============================================================

# Project coordinates to UTM zone 10N so distances are in meters.
transformer = Transformer.from_crs(
    'EPSG:4326',
    'EPSG:26910',
    always_xy=True
)

x_SSM, y_SSM = transformer.transform(
    lon_SSM, lat_SSM
)

x_ecology, y_ecology = transformer.transform(
    lon_SSM_ecology, lat_SSM_ecology
)

pnnl_points = np.column_stack((x_SSM, y_SSM))
ecology_points = np.column_stack((x_ecology, y_ecology))

# Find the nearest Ecology node for each PNNL node.
tree_ecology = cKDTree(ecology_points)

node_distances, pnnl_to_ecology = tree_ecology.query(
    pnnl_points,
    k=1
)

print(
    "Maximum PNNL-Ecology node separation (m):",
    node_distances.max()
)

print(
    "Mean PNNL-Ecology node separation (m):",
    node_distances.mean()
)

# Require a one-to-one node match.
assert len(pnnl_points) == len(ecology_points), \
    "PNNL and Ecology have different node counts."

assert len(np.unique(pnnl_to_ecology)) == len(pnnl_points), \
    "PNNL nodes do not map uniquely to Ecology nodes."

# Require coordinates to agree
coordinate_tolerance_m = 10

assert np.all(node_distances < coordinate_tolerance_m), \
    "Some PNNL and Ecology nodes differ by more than tolerance."

print("All PNNL nodes have a unique Ecology match.")


# ------------------------------------------------------------
# PNNL TRIANGULAR MESH CONNECTIVITY
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

# ============================================================
# TRANSFER PNNL CONNECTIVITY TO ECOLOGY
# ============================================================

# pnnl_to_ecology maps each PNNL local node index to the
# corresponding Ecology local node index.
#
# Translate all PNNL triangles into Ecology local node indices.
triangles_ecology = pnnl_to_ecology[triangles]

assert triangles_ecology.shape == triangles.shape
assert triangles_ecology.min() >= 0
assert triangles_ecology.max() < len(lon_SSM_ecology)

# Add transferred connectivity to the Ecology dataset.
ds_SSM_ecology = ds_SSM_ecology.assign(
    nv=(
        ('elem', 'three'),
        triangles_ecology.astype(np.int32)
    )
)

ds_SSM_ecology['nv'].attrs['description'] = (
    'PNNL triangular connectivity remapped to Ecology local '
    'node indices using coordinate-based node matching.'
)

print("Transferred PNNL connectivity to Ecology.")
print("Ecology connectivity shape:", triangles_ecology.shape)

# ============================================================
# CALCULATE TRIANGULAR ELEMENT AREAS
# ============================================================

# Calculate area using PNNL coordinates and connectivity.
# Because the node coordinates and topology have been matched,
# these same element areas apply to Ecology.

x_tri = x_SSM[triangles]
y_tri = y_SSM[triangles]

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

mask_node_SSM = mask_ps[mask_j, mask_i] == 1

# Transfer the node mask to Ecology indexing.
mask_node_ecology = np.empty(
    len(lon_SSM_ecology),
    dtype=bool
)

mask_node_ecology[pnnl_to_ecology] = mask_node_SSM

# Classify an element as Puget Sound if at least two of
# its three vertices fall within the Puget Sound mask.
mask_elem = (
    mask_node_SSM[triangles].sum(axis=1) >= 2
)

mask_elem_ecology = (
    mask_node_ecology[triangles_ecology].sum(axis=1) >= 2
)

# Verify that both models use the same element mask.
assert np.array_equal(mask_elem, mask_elem_ecology)

print('SSM elements inside Puget Sound:', mask_elem.sum())

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

# ------------------------------------------------------------
# PNNL element values
# ------------------------------------------------------------

DO_elem_SSM = node_to_element(DO_SSM, triangles)
hyp_thick_elem_SSM = node_to_element(hyp_thick_SSM, triangles)
depth_elem_SSM = node_to_element(depth_bot_SSM, triangles)
thick_bot_elem_SSM = node_to_element(thick_bot_SSM, triangles)

# ------------------------------------------------------------
# Ecology element values
# ------------------------------------------------------------

DO_elem_ecology = node_to_element(
    DO_SSM_ecology,
    triangles_ecology
)

hyp_thick_elem_ecology = node_to_element(
    hyp_thick_SSM_ecology,
    triangles_ecology
)

depth_elem_ecology = node_to_element(
    depth_bot_SSM_ecology,
    triangles_ecology
)

thick_bot_elem_ecology = node_to_element(
    thick_bot_SSM_ecology,
    triangles_ecology
)

# ------------------------------------------------------------
# Exclude elements outside Puget Sound
# ------------------------------------------------------------

DO_elem_SSM[:, ~mask_elem] = np.nan
hyp_thick_elem_SSM[:, ~mask_elem] = np.nan

DO_elem_ecology[:, ~mask_elem_ecology] = np.nan
hyp_thick_elem_ecology[:, ~mask_elem_ecology] = np.nan

depth_elem_SSM[:, ~mask_elem] = np.nan
depth_elem_ecology[:, ~mask_elem_ecology] = np.nan

# ============================================================
# PNNL SALISH SEA MODEL
# HYPOXIC AREA AND VOLUME
# ============================================================

# ------------------------------------------------------------
# Hypoxic area
# ------------------------------------------------------------

hypoxic_elem_SSM = DO_elem_SSM <= 2.0

# km2
hyp_area_SSM_ts = np.nansum(
    np.where(
        hypoxic_elem_SSM,
        element_area_km2[None, :], # adding None introduces time compatible dimension
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
        np.isfinite(hyp_thick_elem_SSM),
        hyp_thick_elem_SSM * element_area_km2[None, :],
        0.0
    ),
    axis=1
) / 1e3

# ============================================================
# ECOLOGY SALISH SEA MODEL
# HYPOXIC AREA AND VOLUME
# ============================================================

# ------------------------------------------------------------
# Hypoxic area
# ------------------------------------------------------------

hypoxic_elem_ecology = DO_elem_ecology <= 2.0

hyp_area_SSM_ecology_ts = np.nansum(
    np.where(
        hypoxic_elem_ecology,
        element_area_km2[None, :],
        0.0
    ),
    axis=1
)


# ------------------------------------------------------------
# Hypoxic volume
# ------------------------------------------------------------

hyp_volume_SSM_ecology_ts = np.nansum(
    np.where(
        np.isfinite(hyp_thick_elem_ecology),
        hyp_thick_elem_ecology * element_area_km2[None, :],
        0.0
    ),
    axis=1
) / 1e3


# ============================================================
# TOTAL WATER VOLUME
# ============================================================

# ------------------------------------------------------------
# PNNL SALISH SEA MODEL
# ------------------------------------------------------------

total_volume_SSM = np.nansum(
    depth_elem_SSM[0] * element_area_km2
) / 1e3

print(f'Salish Sea Model PNNL total volume: {total_volume_SSM:.2f} km3')

# ------------------------------------------------------------
# Ecology
# ------------------------------------------------------------

total_volume_SSM_ecology = np.nansum(
    depth_elem_ecology[0] * element_area_km2
) / 1e3

print(
    f'Salish Sea Model Ecology total volume: '
    f'{total_volume_SSM_ecology:.2f} km3'
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
    time_SSM,
    hyp_area_SSM_ts,
    linewidth=2,
    color='green',
    label='SSM PNNL'
)

ax.plot(
    time_SSM_ecology,
    hyp_area_SSM_ecology_ts,
    linewidth=2,
    color='purple',
    label='SSM Ecology'
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

ax.set_title('Hypoxic Area — Bottom Cell%', fontsize=13)

plt.savefig(
    out_dir / 'ssm_ecology_hypoxic_area_comparison.png',
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
    time_SSM,
    hyp_volume_SSM_ts,
    linewidth=2,
    color='green',
    label='SSM PNNL'
)

ax.plot(
    time_SSM_ecology,
    hyp_volume_SSM_ecology_ts,
    linewidth=2,
    color='purple',
    label='SSM Ecology'
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
    out_dir / 'ssm_ecology_hypoxic_volume_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ============================================================
# PLOT HYPOXIC VOLUME (% OF TOTAL VOLUME)
# ============================================================

hyp_volume_pct_SSM = (
    hyp_volume_SSM_ts / total_volume_SSM
) * 100

hyp_volume_pct_SSM_ecology = (
    hyp_volume_SSM_ecology_ts / total_volume_SSM_ecology
) * 100

print('Plotting hypoxic volume as percent of total volume')

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(
    pd.to_datetime(pd.to_datetime(ds_SSM['time'].values)),
    hyp_volume_pct_SSM,
    linewidth=2,
    color='green',
    label='Salish Sea Model'
)

ax.plot(
    time_SSM_ecology,
    hyp_volume_pct_SSM_ecology,
    linewidth=2,
    color='purple',
    label='SSM Ecology'
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
    out_dir / 'ssm_ecology_hypoxic_volume_percent_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# HYPOXIA OCCURRENCE
# ============================================================

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
    np.nanmax(hyp_occurrence_SSM)
])

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
    1, 2,
    figsize=(20, 7),
    constrained_layout=True
)

# ------------------------------------------------------------
# SSM
# ------------------------------------------------------------

pcm1 = axes[0].tripcolor(
    triang,
    facecolors=hyp_occurrence_SSM_plot,
    shading='flat',
    cmap=cmap,
    vmin=0.001,
    # vmax=occurrence_max
    vmax=140
)

axes[0].set_title('SSM: Bottom Hypoxia (DO ≤ 2.0 mg/L) Occurrence [Days]')
axes[0].set_xlabel('Longitude')
axes[0].set_xlim(lon_min - lon_pad, lon_max + lon_pad)
axes[0].set_ylim(lat_min - lat_pad, lat_max + lat_pad)

pfun.add_coast(axes[2], color='black', linewidth=0.8)

cbar = fig.colorbar(
    pcm1,
    ax=axes[0]
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

mean_DO_SSM = np.nanmean(DO_elem, axis=0)

# Apply Puget Sound mask
mean_DO_SSM_plot = np.where(mask_elem, mean_DO_SSM, np.nan)

# Common color scale for both models
DO_min = np.nanmin([
    np.nanmin(mean_DO_SSM_plot)
])

DO_max = np.nanmax([
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
    1, 2,
    figsize=(18, 6),
    constrained_layout=True
)

# ------------------------------------------------------------
# Salish Sea Model
# ------------------------------------------------------------

pcm1 = axes[0].tripcolor(
    triang,
    facecolors= mean_DO_SSM_plot,
    shading='flat',
    cmap='viridis_r',
    vmin=DO_min,
    vmax=DO_max
)

axes[0].set_title('Salish Sea Model')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

pfun.add_coast(axes[0], color='black', linewidth=0.8)
# ------------------------------------------------------------
# Same colorbar for all
# ------------------------------------------------------------

cbar = fig.colorbar(
    pcm1,
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

# ------------------------------------------------------------
# Common color scale for both models and all days
# ------------------------------------------------------------

# DO_min = np.nanmin([
#     np.nanmin(DO_elem)
# ])

# DO_max = np.nanmax([
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
).copy()

# Missing/invalid cells should not be colored as low oxygen
cmap_oxy.set_bad('white')

# Common oxygen limits
DO_min = 0.0
DO_max = 15.0  
DO_threshold = 2.0  # mg/L

# Map DO = 2 mg/L to 20% of the colormap
# Since we cropped the top 20% of the colormap, this becomes 25% of cropped colormap
hypoxic_fraction = 0.2 / 0.8 # 0.25

# explicitly preserve invalid values (NaN) 
def forward(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)

    valid = np.isfinite(x)
    hypoxic = valid & (x <= DO_threshold)
    oxic = valid & (x > DO_threshold)

    out[hypoxic] = (
        hypoxic_fraction * x[hypoxic] / DO_threshold
    )

    out[oxic] = (
        hypoxic_fraction
        + (1 - hypoxic_fraction)
        * (x[oxic] - DO_threshold)
        / (DO_max - DO_threshold)
    )

    return out


def inverse(y):
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)

    valid = np.isfinite(y)
    hypoxic = valid & (y <= hypoxic_fraction)
    oxic = valid & (y > hypoxic_fraction)

    out[hypoxic] = (
        DO_threshold * y[hypoxic] / hypoxic_fraction
    )

    out[oxic] = (
        DO_threshold
        + (DO_max - DO_threshold)
        * (y[oxic] - hypoxic_fraction)
        / (1 - hypoxic_fraction)
    )

    return out

norm = FuncNorm(
    (forward, inverse),
    vmin=DO_min,
    vmax=DO_max,
    clip=True
)

# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

time_SSM = pd.to_datetime(ds_SSM['time'].values)

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig = plt.figure(figsize=(22, 10))

gs = fig.add_gridspec(
    2, 4,
    width_ratios=[1, 1, 1, 0.045],
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

# Dedicated colorbar axis
cax = fig.add_subplot(gs[1, 3])
# ------------------------------------------------------------
# Initial frame
# ------------------------------------------------------------

pcm1 = axes[0].tripcolor(
    triang,
    facecolors= DO_elem[0, :],
    shading='flat',
    # cmap='viridis_r',
    # vmin=DO_min,
    # vmax=DO_max,
    cmap=cmap_oxy,
    norm=norm,
)

pfun.add_coast(axes[0], color='black', linewidth=0.8)


# ------------------------------------------------------------
# Titles and labels
# ------------------------------------------------------------

axes[0].set_title('Salish Sea Model: PNNL')
for ax in axes:
    ax.set_facecolor('white')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

# ------------------------------------------------------------
# Plot hypoxic area time series
# ------------------------------------------------------------

# Use the same dates as the animation frames
time = pd.to_datetime(time_SSM)

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
    pcm1,
    cax=cax
)

cbar.set_label('DO in bottom 14.6% (mg/L)')

# Keep latitude labels only on the leftmost map
for ax in axes[1:]:
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelleft=False)

# ============================================================
# Animation function
# ============================================================

def update(frame):
    global pcm1

    # Remove the previous SSM collection
    pcm1.remove()

    # Recreate the SSM plot for this frame
    pcm1 = axes[0].tripcolor(
        triang,
        facecolors=DO_elem[frame, :],
        shading='flat',
        # cmap='viridis_r',
        # vmin=DO_min,
        # vmax=DO_max
        cmap=cmap_oxy,
        norm=norm,
    )
    
    # Move the time-series marker to the current animation date
    current_time = time[frame]

    time_marker.set_xdata([current_time, current_time])
    date_label.set_text(current_time.strftime('%Y-%m-%d'))

    return pcm1, time_marker, date_label


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