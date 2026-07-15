"""
Plot station-mean model errors for LiveOcean and SalishSeaCast.

Panels:
1. LiveOcean mean error: LO - observation
2. SalishSeaCast mean error: SSC - observation
3. Difference in models: SSC - LO
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
    'name': obs_df['name'].to_numpy(),
    'lon': obs_df['lon'].to_numpy(),
    'lat': obs_df['lat'].to_numpy(),
    'obs': obs_df[vn].to_numpy() * conversion_factor,
    'lo': lo_df[vn].to_numpy() * conversion_factor,
    'ssc': ssc_df[vn].to_numpy() * conversion_factor,
})

# Ensure station identifiers all have the same data type
sample_df['name'] = sample_df['name'].astype(str)

# Retain only rows where the observation and both models are finite.
# This ensures that LO and SSC are evaluated using identical samples.
sample_df = sample_df.replace([np.inf, -np.inf], np.nan)

sample_df = sample_df.dropna(
    subset=['name', 'lon', 'lat', 'obs', 'lo', 'ssc']
).copy()

# Model minus observation
sample_df['lo_error'] = (
    sample_df['lo'] - sample_df['obs']
)

sample_df['ssc_error'] = (
    sample_df['ssc'] - sample_df['obs']
)

# Difference in models
sample_df['model_difference'] = (
    sample_df['ssc'] - sample_df['lo']
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

# ============================================================
# AVERAGE ERROR BY STATION
# ============================================================

station_error = (
    sample_df
    .groupby('name', as_index=False)
    .agg(
        lon=('lon', 'mean'),
        lat=('lat', 'mean'),
        n=('obs', 'size'),
        mean_obs=('obs', 'mean'),
        lo_error=('lo_error', 'mean'),
        ssc_error=('ssc_error', 'mean'),
        model_difference=('model_difference', 'mean'),
    )
)

station_error = station_error.loc[
    station_error['n'] >= min_samples
].copy()

# print(station_error.sort_values('name').to_string(index=False))

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
# SHARED COLOR LIMIT
# ============================================================

# Use one symmetric color range for all three panels.
all_values = np.concatenate([
    station_error['lo_error'].to_numpy(),
    station_error['ssc_error'].to_numpy(),
    station_error['model_difference'].to_numpy(),
])

all_values = all_values[np.isfinite(all_values)]

if len(all_values) == 0:
    raise ValueError('No finite station errors are available.')

vmax = np.nanmax(np.abs(all_values))

# Avoid an invalid normalization if every error is zero
if vmax == 0:
    vmax = 1

norm = TwoSlopeNorm(
    vmin=-vmax,
    vcenter=0,
    vmax=vmax
)

# ============================================================
# PLOT
# ============================================================

plt.close('all')

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14, 6),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

panel_info = [
    (
        'lo_error',
        'LiveOcean error',
        'LiveOcean − observation'
    ),
    (
        'ssc_error',
        'SalishSeaCast error',
        'SalishSeaCast − observation'
    ),
    (
        'model_difference',
        'Difference in models',
        'SSC − LO'
    ),
]

scatter = None

for ax, (column, title, subtitle) in zip(
    axes,
    panel_info
):

    # Bathymetry background
    ax.pcolormesh(
        lon_rho,
        lat_rho,
        h_plot,
        cmap='Greys',
        shading='auto',
        alpha=0.25,
        zorder=0
    )

    # Coastline
    pfun.add_coast(ax)

    scatter = ax.scatter(
        station_error['lon'],
        station_error['lat'],
        c=station_error[column],
        cmap='RdBu_r',
        norm=norm,
        s=75,
        edgecolor='black',
        linewidth=0.6,
        zorder=5
    )

    ax.set_xlim(lon_low, lon_high)
    ax.set_ylim(lat_low, lat_high)

    ax.set_title(
        f'{title}\n{subtitle}',
        fontsize=11
    )

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=8
    )

    ax.set_aspect(
        1 / np.cos(
            np.deg2rad(
                np.mean([lat_low, lat_high])
            )
        )
    )

    # Station labels
    # for _, row in station_error.iterrows():
    #     ax.annotate(
    #         row['name'],
    #         (row['lon'], row['lat']),
    #         xytext=(3, 3),
    #         textcoords='offset points',
    #         fontsize=6,
    #         zorder=6
    #     )

axes[0].set_ylabel('Latitude', fontsize=10)

for ax in axes:
    ax.set_xlabel('Longitude', fontsize=10)

cbar = fig.colorbar(
    scatter,
    ax=axes,
    orientation='horizontal',
    fraction=0.06,
    pad=0.08,
    extend='both'
)

cbar.set_label(
    f'Mean error in {var_label} ({units})',
    fontsize=10
)

cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    f'{var_label} station-mean error, {year}',
    fontsize=14
)

out_fn = out_dir / (
    f'{otype}_{year}_{vn}_station_mean_error_map.png'
)

fig.savefig(
    out_fn,
    dpi=300,
    bbox_inches='tight'
)
print(f'\nSaved figure to:\n{out_fn}')

plt.show()

grid_ds.close()