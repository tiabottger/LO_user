"""
Plot station-mean model errors for LiveOcean and SalishSeaCast.

Panels:
1. LiveOcean RMSE
2. SalishSeaCast RMSE
3. Difference in model error: SSC RMSE - LO RMSE
this third panel shows which model has the least error at each station
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

# Variable as stored in df0_dict
vn = 'DO'

# Label and optional conversion
var_label = 'Dissolved oxygen'
units = r'mg L$^{-1}$'

# Example conversion for DO from mmol m-3 to mg L-1.
# Change to 1 if the variable is already in the desired units.
conversion_factor = 0.032

# Minimum number of matched samples required at a station
min_samples = 3

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

obs_df = df0_dict[obs_key].copy()
lo_df = df0_dict[lo_key].copy()
ssc_df = df0_dict[ssc_key].copy()

# Confirm that corresponding rows are aligned
if not (
    len(obs_df) == len(lo_df) == len(ssc_df)
):
    raise ValueError(
        'Observation, LiveOcean, and SalishSeaCast '
        'DataFrames have different lengths.'
    )

# ============================================================
# CREATE SAMPLE-LEVEL DATAFRAME
# ============================================================

sample_df = pd.DataFrame({
    'name': obs_df['name'],
    'lon': obs_df['lon'],
    'lat': obs_df['lat'],
    'obs': obs_df[vn] * conversion_factor,
    'lo': lo_df[vn] * conversion_factor,
    'ssc': ssc_df[vn] * conversion_factor,
})

sample_df['name'] = sample_df['name'].astype(str)

sample_df = (
    sample_df
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=['obs', 'lo', 'ssc'])
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
sample_points = np.column_stack((
    sample_df['lon'].to_numpy(),
    sample_df['lat'].to_numpy(),
))

# Find nearest valid grid point for each sample
_, nearest_valid_index = tree.query(sample_points)

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

sample_df = sample_df.loc[
    inside_puget_sound
].copy()

# ------------------------------------------------------------
# Squared errors
# ------------------------------------------------------------

sample_df['lo_sq_error'] = (
    sample_df['lo'] - sample_df['obs']
)**2

sample_df['ssc_sq_error'] = (
    sample_df['ssc'] - sample_df['obs']
)**2

# ============================================================
# STATION RMSE
# ============================================================

station_rmse = (
    sample_df
    .groupby('name', as_index=False)
    .agg(
        lon=('lon', 'mean'),
        lat=('lat', 'mean'),
        n=('obs', 'size'),
        lo_mse=('lo_sq_error', 'mean'),
        ssc_mse=('ssc_sq_error', 'mean'),
    )
)

station_rmse = station_rmse[
    station_rmse['n'] >= min_samples
].copy()

station_rmse['lo_rmse'] = np.sqrt(
    station_rmse['lo_mse']
)

station_rmse['ssc_rmse'] = np.sqrt(
    station_rmse['ssc_mse']
)

station_rmse['rmse_difference'] = (
    station_rmse['ssc_rmse']
    - station_rmse['lo_rmse']
)

station_rmse = station_rmse.drop(
    columns=['lo_mse', 'ssc_mse']
)

# print(
#     station_rmse.sort_values('name').to_string(index=False)
# )

# ============================================================
# LOAD GRID FOR MAP BACKGROUND
# ============================================================
Ldir = Lfun.Lstart(gridname='cas7')
grid_fn = Ldir['grid'] / 'grid.nc'
grid_ds = xr.open_dataset(grid_fn)

lon_rho = grid_ds['lon_rho'].values
lat_rho = grid_ds['lat_rho'].values
h = grid_ds['h'].values
mask_rho = grid_ds['mask_rho'].values

# Mask land
h_plot = np.where(mask_rho == 1, h, np.nan)

# ============================================================
# COLOR LIMITS
# ============================================================

rmse_max = max(
    station_rmse['lo_rmse'].max(),
    station_rmse['ssc_rmse'].max()
)

diff_max = np.max(
    np.abs(station_rmse['rmse_difference'])
)

diff_norm = TwoSlopeNorm(
    vmin=-diff_max,
    vcenter=0,
    vmax=diff_max
)


# ============================================================
# PLOT
# ============================================================

plt.close('all')

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14,6),
    constrained_layout=True,
    sharex=True,
    sharey=True
)

# ------------------------------------------------------------
# Panel 1: LiveOcean RMSE
# ------------------------------------------------------------

sc1 = axes[0].scatter(
    station_rmse['lon'],
    station_rmse['lat'],
    c=station_rmse['lo_rmse'],
    cmap='viridis',
    vmin=0,
    #vmax=rmse_max,
    vmax = 2,# set max to 2 since if we're off by 2 we're not capturing hypoxia
    s=90,
    edgecolor='k',
    linewidth=0.6,
    zorder=10
)

# ------------------------------------------------------------
# Panel 2: SalishSeaCast RMSE
# ------------------------------------------------------------

sc2 = axes[1].scatter(
    station_rmse['lon'],
    station_rmse['lat'],
    c=station_rmse['ssc_rmse'],
    cmap='viridis',
    vmin=0,
    #vmax=rmse_max,
    vmax = 2,# set max to 2 since if we're off by 2 we're not capturing hypoxia
    s=90,
    edgecolor='k',
    linewidth=0.6,
    zorder=10
)

# ------------------------------------------------------------
# Panel 3: Difference
# ------------------------------------------------------------

sc3 = axes[2].scatter(
    station_rmse['lon'],
    station_rmse['lat'],
    c=station_rmse['rmse_difference'],
    cmap='RdBu_r',
    norm=diff_norm,
    s=90,
    edgecolor='k',
    linewidth=0.6,
    zorder=10
)

titles = [
    'LiveOcean RMSE',
    'SalishSeaCast RMSE',
    'SSC RMSE − LO RMSE'
]

for ax, title in zip(axes, titles):

    ax.pcolormesh(
        lon_rho,
        lat_rho,
        h_plot,
        cmap='Greys',
        shading='auto',
        alpha=0.25
    )

    pfun.add_coast(ax)

    ax.set_xlim(lon_low, lon_high)
    ax.set_ylim(lat_low, lat_high)

    pfun.dar(ax)

    ax.set_title(title, fontsize=12)

    ax.tick_params(labelsize=8)

# ------------------------------------------------------------
# Colorbars
# ------------------------------------------------------------

cb1 = fig.colorbar(
    sc1,
    ax=axes[:2],
    orientation='horizontal',
    fraction=0.05,
    pad=0.08
)

cb1.set_label(f'RMSE ({units})')

cb2 = fig.colorbar(
    sc3,
    ax=axes[2],
    orientation='horizontal',
    fraction=0.05,
    pad=0.08
)

cb2.set_label('RMSE Difference (SSC − LO)')

fig.suptitle(
    f'{var_label} Station RMSE ({year})',
    fontsize=15
)

out_fn = out_dir / (
    f'{otype}_{year}_{vn}_station_rmse_error_map.png'
)

fig.savefig(
    out_fn,
    dpi=300,
    bbox_inches='tight'
)

print(f'\nSaved figure to:\n{out_fn}')

plt.show()

grid_ds.close()