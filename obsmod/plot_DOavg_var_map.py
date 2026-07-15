"""
Plot station-average dissolved oxygen concentrations for:
1. Observations
2. LiveOcean
3. SalishSeaCast

Each point represents the mean of all matched bottle samples and
depths available at that station.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial import cKDTree

from lo_tools import plotting_functions as pfun
from lo_tools import Lfun


# ============================================================
# USER SETTINGS
# ============================================================

# Set the LiveOcean grid explicitly so Ldir['grid'] does not
# point to LO_data/grids/BLANK.
Ldir = Lfun.Lstart(gridname='cas7')

year = '2014'
otype = 'bottle'

# Variable name in the combined DataFrames
vn = 'DO'

# DO is commonly stored as mmol m-3 in these files.
# mmol O2 m-3 * 0.032 = mg O2 L-1
conversion_factor = 0.032

var_label = 'Dissolved oxygen'
units = r'mg L$^{-1}$'

# Require at least this many matched samples per station
min_samples = 3

# Puget Sound map extent
lon_low = -123.3
lon_high = -122.1
lat_low = 46.95
lat_high = 48.35

# Dataset dictionary keys
obs_key = 'obs'
lo_key = 'cas7_t1_x11ab'
ssc_key = 'ssc'

# Plot settings
station_size = 75
station_edge_width = 0.6
cmap = 'viridis'


# ============================================================
# FILE PATHS
# ============================================================

in_dir = Ldir['parent'] / 'LO_output' / 'obsmod'
out_dir = Ldir['parent'] / 'LO_output' / 'obsmod_plots'
Lfun.make_dir(out_dir)

in_fn = (
    in_dir
    / f'combined_{otype}_{year}_cas7_t1_x11ab_ssc.pkl'
)

grid_fn = Ldir['grid'] / 'grid.nc'

if not in_fn.is_file():
    raise FileNotFoundError(
        f'Combined observation-model file not found:\n{in_fn}'
    )

if not grid_fn.is_file():
    raise FileNotFoundError(
        f'LiveOcean grid file not found:\n{grid_fn}'
    )


# ============================================================
# LOAD COMBINED OBSERVATION-MODEL DATA
# ============================================================

# load the basin masks
mask_ds = xr.open_dataset('basin_masks_from_pugetsoundDObox.nc')

with open(in_fn, 'rb') as file:
    df0_dict = pickle.load(file)

# Remove non-DataFrame entries such as metadata
df0_dict = {
    key: value
    for key, value in df0_dict.items()
    if isinstance(value, pd.DataFrame)
}

required_keys = [obs_key, lo_key, ssc_key]

missing_keys = [
    key for key in required_keys
    if key not in df0_dict
]

if missing_keys:
    raise KeyError(
        f'Missing required DataFrame keys: {missing_keys}\n'
        f'Available keys: {list(df0_dict.keys())}'
    )

obs_df = df0_dict[obs_key].copy()
lo_df = df0_dict[lo_key].copy()
ssc_df = df0_dict[ssc_key].copy()

if not (
    len(obs_df) == len(lo_df) == len(ssc_df)
):
    raise ValueError(
        'The observation, LiveOcean, and SalishSeaCast '
        'DataFrames do not have the same number of rows:\n'
        f'obs={len(obs_df)}, '
        f'LO={len(lo_df)}, '
        f'SSC={len(ssc_df)}'
    )

for key, dataframe in [
    (obs_key, obs_df),
    (lo_key, lo_df),
    (ssc_key, ssc_df),
]:
    if vn not in dataframe.columns:
        raise KeyError(
            f'Variable {vn!r} was not found in {key!r}.'
        )


# ============================================================
# BUILD MATCHED SAMPLE DATAFRAME
# ============================================================

# Use to_numpy() to preserve row-by-row alignment regardless
# of the original DataFrame indices.
sample_df = pd.DataFrame({
    'name': obs_df['name'].to_numpy(),
    'lon': obs_df['lon'].to_numpy(),
    'lat': obs_df['lat'].to_numpy(),
    'obs_do': (
        obs_df[vn].to_numpy(dtype=float)
        * conversion_factor
    ),
    'lo_do': (
        lo_df[vn].to_numpy(dtype=float)
        * conversion_factor
    ),
    'ssc_do': (
        ssc_df[vn].to_numpy(dtype=float)
        * conversion_factor
    ),
})

# Prevent mixed string/integer station identifiers
sample_df['name'] = sample_df['name'].astype(str)

sample_df = sample_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Keep identical samples for all three panels so the averages
# are directly comparable.
sample_df = sample_df.dropna(
    subset=[
        'name',
        'lon',
        'lat',
        'obs_do',
        'lo_do',
        'ssc_do',
    ]
).copy()

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
# CALCULATE STATION MEAN AND VARIANCE 
# ============================================================

station_stats = (
    sample_df
    .groupby('name', as_index=False)
    .agg(
        lon=('lon', 'mean'),
        lat=('lat', 'mean'),
        n=('obs_do', 'size'),

        obs_mean=('obs_do', 'mean'),
        lo_mean=('lo_do', 'mean'),
        ssc_mean=('ssc_do', 'mean'),

        obs_variance=('obs_do', 'var'),
        lo_variance=('lo_do', 'var'),
        ssc_variance=('ssc_do', 'var'),
    )
)

station_stats = station_stats.loc[
    station_stats['n'] >= min_samples
].copy()

station_stats = (
    station_stats
    .sort_values('name')
    .reset_index(drop=True)
)

if station_stats.empty:
    raise ValueError(
        'No stations remain after applying the finite-value, '
        'Puget Sound mask, and minimum-sample filters.'
    )

# print('\nStation-average DO and variance:')
# print(
#     station_stats.to_string(
#         index=False,
#         formatters={
#             'lon': '{:.4f}'.format,
#             'lat': '{:.4f}'.format,
#             'obs_mean': '{:.2f}'.format,
#             'lo_mean': '{:.2f}'.format,
#             'ssc_mean': '{:.2f}'.format,
#             'obs_variance': '{:.2f}'.format,
#             'lo_variance': '{:.2f}'.format,
#             'ssc_variance': '{:.2f}'.format,
#         },
#     )
# )

# ============================================================
# LOAD GRID FOR MAP BACKGROUND
# ============================================================

with xr.open_dataset(grid_fn) as grid_ds:
    lon_rho = grid_ds['lon_rho'].values
    lat_rho = grid_ds['lat_rho'].values
    h = grid_ds['h'].values
    mask_rho = grid_ds['mask_rho'].values

# Retain water depths and mask land
h_plot = np.where(mask_rho == 1, h, np.nan)


# ============================================================
# SHARED CONCENTRATION COLOR SCALE
# ============================================================

all_concentrations = np.concatenate([
    station_stats['obs_mean'].to_numpy(),
    station_stats['lo_mean'].to_numpy(),
    station_stats['ssc_mean'].to_numpy(),
])

all_concentrations = all_concentrations[
    np.isfinite(all_concentrations)
]

# Use one scale across all panels for direct comparison.
vmin = np.nanmin(all_concentrations)
vmax = np.nanmax(all_concentrations)

if np.isclose(vmin, vmax):
    vmin -= 0.5
    vmax += 0.5


# ============================================================
# CREATE THREE-PANEL DO AVG MAP
# ============================================================

plt.close('all')

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(14, 6),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

panel_info = [
    ('obs_mean', 'Observations'),
    ('lo_mean', 'LiveOcean'),
    ('ssc_mean', 'SalishSeaCast'),
]

scatter = None

for ax, (column, title) in zip(axes, panel_info):

    # Plot map background first so it stays behind stations
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
        station_stats['lon'],
        station_stats['lat'],
        c=station_stats[column],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=station_size,
        edgecolor='black',
        linewidth=station_edge_width,
        zorder=5,
    )

    ax.set_xlim(lon_low, lon_high)
    ax.set_ylim(lat_low, lat_high)

    pfun.dar(ax)

    ax.set_title(title, fontsize=12)

    ax.set_xlabel(
        'Longitude',
        fontsize=10,
    )

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=8,
    )

axes[0].set_ylabel(
    'Latitude',
    fontsize=10,
)

# One shared colorbar emphasizes that colors represent the same
# concentrations in all three panels.
cbar = fig.colorbar(
    scatter,
    ax=axes,
    orientation='horizontal',
    fraction=0.06,
    pad=0.08,
    extend='neither',
)

cbar.set_label(
    f'Average {var_label.lower()} ({units})',
    fontsize=10,
)

cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    f'Station-average dissolved oxygen concentrations\n'
    f'{year} {otype} samples',
    fontsize=14,
)

out_fn = (
    out_dir
    / f'{otype}_{year}_{vn}_station_DOavg_map.png'
)

fig.savefig(
    out_fn,
    dpi=300,
    bbox_inches='tight',
)

print(f'\nSaved DO mean figure to:\n{out_fn}')

plt.show()

# ============================================================
# SHARED DO VARIANCE COLOR SCALE
# ============================================================

all_variances = np.concatenate([
    station_stats['obs_variance'].to_numpy(),
    station_stats['lo_variance'].to_numpy(),
    station_stats['ssc_variance'].to_numpy(),
])

all_variances = all_variances[
    np.isfinite(all_variances)
]

if len(all_variances) == 0:
    raise ValueError(
        'No finite station DO variances are available.'
    )

variance_vmin = 0
variance_vmax = np.nanmax(all_variances)

if np.isclose(variance_vmax, 0):
    variance_vmax = 1


# ============================================================
# CREATE THREE-PANEL DO VARIANCE MAP
# ============================================================

fig_var, axes_var = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(14, 6),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

variance_panel_info = [
    ('obs_variance', 'Observations'),
    ('lo_variance', 'LiveOcean'),
    ('ssc_variance', 'SalishSeaCast'),
]

variance_scatter = None

for ax, (column, title) in zip(
    axes_var,
    variance_panel_info,
):

    # Bathymetry background
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

    variance_scatter = ax.scatter(
        station_stats['lon'],
        station_stats['lat'],
        c=station_stats[column],
        cmap='viridis',
        vmin=variance_vmin,
        vmax=variance_vmax,
        s=station_size,
        edgecolor='black',
        linewidth=station_edge_width,
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

    ax.set_title(
        title,
        fontsize=12,
    )

    ax.set_xlabel(
        'Longitude',
        fontsize=10,
    )

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=8,
    )

axes_var[0].set_ylabel(
    'Latitude',
    fontsize=10,
)

variance_cbar = fig_var.colorbar(
    variance_scatter,
    ax=axes_var,
    orientation='horizontal',
    fraction=0.06,
    pad=0.08,
)

variance_cbar.set_label(
    r'Dissolved oxygen variance '
    r'[(mg L$^{-1}$)$^2$]',
    fontsize=10,
)

variance_cbar.ax.tick_params(
    labelsize=8,
)

fig_var.suptitle(
    f'Station dissolved oxygen variance\n'
    f'{year} {otype} samples',
    fontsize=14,
)

variance_out_fn = (
    out_dir
    / f'{otype}_{year}_{vn}_station_variance_map.png'
)

fig_var.savefig(
    variance_out_fn,
    dpi=300,
    bbox_inches='tight',
)

print(
    f'\nSaved DO variance figure to:\n'
    f'{variance_out_fn}'
)

plt.show()