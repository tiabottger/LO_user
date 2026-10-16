"""
Plot station-mean model errors for SSM hourly vs daily average output.

Panels:
1. Daily average RMSE
2. Hourly average RMSE
3. RMSE difference (hourly - daily)
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun

Ldir = Lfun.Lstart()

# ============================================================
# USER SETTINGS
# ============================================================

# Standard deviation threshold (mg/L)
std_threshold = 2.5

year = '2014'
otype = 'bottle'

# Variable as stored in df0_dict
# vn = 'DO'
# vn = 'SA'
vn = 'CT'

# Label and optional conversion
# var_label = 'Dissolved oxygen'
# units = r'mg L$^{-1}$'
# var_label = 'Salinity'
# units = r'g kg$^{-1}$'
var_label = 'Temperature'
units = r'°C'

# Minimum number of matched samples required at a station
min_samples = 3

# Map extent
lat_low = 46.95
lat_high = 48.35
lon_low = -123.3
lon_high = -122.1

# Dataset keys
obs_key = 'obs'
ssm_key = 'ssm'

daily_filename = (
    f'combined_{otype}_{year}_'
    'cas7_t1_x11ab_ssc_ssmpnnl_dailyavg.pkl'
)

hourly_filename = (
    f'combined_{otype}_{year}_'
    'cas7_t1_x11ab_ssc_ssmpnnl.pkl'
)

# ============================================================
# LOAD DATA
# ============================================================

# Load the basin masks
mask_ds = xr.open_dataset(
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

def load_matched_data(filename):
    """Load observations and SSM predictions from a pickle."""

    in_fn = in_dir / filename

    with open(in_fn, 'rb') as f:
        data_dict = pickle.load(f)

    data_dict = {
        key: df
        for key, df in data_dict.items()
        if isinstance(df, pd.DataFrame)
    }

    for key in [obs_key, ssm_key]:
        if key not in data_dict:
            raise KeyError(
                f'{key!r} not found in {filename}. '
                f'Available keys: {list(data_dict.keys())}'
            )

    obs = data_dict[obs_key].copy()
    ssm = data_dict[ssm_key].copy()

    if len(obs) != len(ssm):
        raise ValueError(
            f'{filename}: observations ({len(obs)}) and '
            f'SSM predictions ({len(ssm)}) have different lengths.'
        )
        
    # Confirm row-by-row alignment within each file.
    for col in ['cid', 'time', 'z']:
        if col in obs.columns and col in ssm.columns:
            if col == 'time':
                a = pd.to_datetime(obs[col]).reset_index(drop=True)
                b = pd.to_datetime(ssm[col]).reset_index(drop=True)
                aligned = a.equals(b)
            else:
                a = obs[col].reset_index(drop=True)
                b = ssm[col].reset_index(drop=True)
                aligned = np.allclose(
                    pd.to_numeric(a, errors='coerce'),
                    pd.to_numeric(b, errors='coerce'),
                    equal_nan=True,
                )

            if not aligned:
                raise ValueError(
                    f'{filename}: observation and SSM '
                    f'rows do not align for {col!r}.'
                )

    result = pd.DataFrame({
        'cid': obs['cid'].to_numpy(),
        'time': pd.to_datetime(obs['time']).to_numpy(),
        'z': pd.to_numeric(obs['z']).to_numpy(),
        'name': obs['name'].astype(str).to_numpy(),
        'lon': pd.to_numeric(obs['lon']).to_numpy(),
        'lat': pd.to_numeric(obs['lat']).to_numpy(),
        'obs': pd.to_numeric(obs[vn]).to_numpy() * 0.032,
        'ssm': pd.to_numeric(ssm[vn]).to_numpy() * 0.032,
    })

    return result

daily_df = load_matched_data(daily_filename)
hourly_df = load_matched_data(hourly_filename)

print(f'Daily observations:  {len(daily_df)}')
print(f'Hourly observations: {len(hourly_df)}')


# ============================================================
# ALIGN THE SAME OBSERVATIONS ACROSS BOTH FILES
# ============================================================

# Build observation keys from the original observation identity.
# The occurrence index distinguishes repeated keys while
# preserving every observation row.

match_cols = ['cid', 'time', 'z_key']

for df in [daily_df, hourly_df]:
    df['time'] = pd.to_datetime(df['time'])
    df['z_key'] = df['z'].round(3)

    # Assign 0, 1, 2, ... to repeated observation keys
    df['duplicate_idx'] = (
        df.groupby(match_cols, dropna=False)
        .cumcount()
    )

match_cols = match_cols + ['duplicate_idx']

print(
    'Duplicate rows in daily file:',
    daily_df.duplicated(
        ['cid', 'time', 'z_key'],
        keep=False
    ).sum()
)

print(
    'Duplicate rows in hourly file:',
    hourly_df.duplicated(
        ['cid', 'time', 'z_key'],
        keep=False
    ).sum()
)

# Pair records using observation identity plus occurrence.
paired = daily_df[
    match_cols + ['name', 'lon', 'lat', 'obs', 'ssm']
].merge(
    hourly_df[
        match_cols + ['name', 'lon', 'lat', 'obs', 'ssm']
    ],
    on=match_cols,
    how='inner',
    suffixes=('_daily', '_hourly'),
    validate='one_to_one',
)

print(f'Daily rows: {len(daily_df)}')
print(f'Hourly rows: {len(hourly_df)}')
print(f'Paired rows: {len(paired)}')

# Verify that observations and station names agree.
if not np.allclose(
    paired['obs_daily'],
    paired['obs_hourly'],
    equal_nan=True,
):
    raise ValueError(
        'Observed DO differs between matched records.'
    )

if not paired['name_daily'].equals(
    paired['name_hourly']
):
    raise ValueError(
        'Station names differ between matched records.'
    )
    
# ----------------------------------------------------
# CREATE SAMPLE DATAFRAME FOR PLOTTING
# ----------------------------------------------------

sample_df = pd.DataFrame({
    'cid': paired['cid'],
    'time': paired['time'],
    'z': paired['z_key'],
    'name': paired['name_daily'],
    'lon': paired['lon_daily'],
    'lat': paired['lat_daily'],
    'obs': paired['obs_daily'],
    'ssm_daily': paired['ssm_daily'],
    'ssm_hourly': paired['ssm_hourly'],
})

# Confirm the daily and hourly files use the same observations
np.testing.assert_allclose(
    paired['obs_daily'].to_numpy(),
    paired['obs_hourly'].to_numpy(),
    equal_nan=True,
)

np.testing.assert_allclose(
    paired['lon_daily'].to_numpy(),
    paired['lon_hourly'].to_numpy(),
    equal_nan=True,
)

np.testing.assert_allclose(
    paired['lat_daily'].to_numpy(),
    paired['lat_hourly'].to_numpy(),
    equal_nan=True,
)

print('Sample dataframe rows:', len(sample_df))

# ============================================================
# FILTER TO PUGET SOUND BASIN MASK
# ============================================================

mask_lon = mask_ds['lon_rho'].values
mask_lat = mask_ds['lat_rho'].values
puget_sound_mask = mask_ds['mask_pugetsound'].values

grid_points = np.column_stack((
    mask_lon.ravel(),
    mask_lat.ravel(),
))

valid_grid = (
    np.isfinite(grid_points[:, 0])
    & np.isfinite(grid_points[:, 1])
)

tree = cKDTree(grid_points[valid_grid])

sample_points = np.column_stack((
    sample_df['lon'].to_numpy(),
    sample_df['lat'].to_numpy(),
))

_, nearest_valid_index = tree.query(sample_points)

valid_flat_indices = np.flatnonzero(valid_grid)
nearest_flat_index = valid_flat_indices[nearest_valid_index]

eta_index, xi_index = np.unravel_index(
    nearest_flat_index,
    mask_lon.shape,
)

inside_puget_sound = (
    puget_sound_mask[eta_index, xi_index] == 1
)

sample_df = sample_df.loc[inside_puget_sound].copy()

print(f'Common valid observations inside Puget Sound: {len(sample_df)}')

# ============================================================
# SQUARED ERRORS
# ============================================================

sample_df['daily_sq_error'] = (
    sample_df['ssm_daily'] - sample_df['obs']
) ** 2

sample_df['hourly_sq_error'] = (
    sample_df['ssm_hourly'] - sample_df['obs']
) ** 2


# ============================================================
# STATION RMSE AND OBSERVATIONAL VARIABILITY
# ============================================================

station_rmse = (
    sample_df
    .groupby('name', as_index=False)
    .agg(
        lon=('lon', 'mean'),
        lat=('lat', 'mean'),
        n=('obs', 'size'),
        obs_std=('obs', 'std'),
        daily_mse=('daily_sq_error', 'mean'),
        hourly_mse=('hourly_sq_error', 'mean'),
    )
)

station_rmse['daily_rmse'] = np.sqrt(
    station_rmse['daily_mse']
)

station_rmse['hourly_rmse'] = np.sqrt(
    station_rmse['hourly_mse']
)

# Positive means daily matching has greater RMSE.
station_rmse['rmse_diff'] = (
    station_rmse['daily_rmse']
    - station_rmse['hourly_rmse']
)

# Retain stations with enough common observations.
station_rmse = station_rmse.loc[
    station_rmse['n'] >= min_samples
].copy()

station_rmse['high_variability'] = (
    station_rmse['obs_std'] >= std_threshold
)

print('\nStation RMSE summary:')
print(
    station_rmse[
        [
            'name', 'n', 'obs_std',
            'daily_rmse', 'hourly_rmse', 'rmse_diff'
        ]
    ].to_string(index=False)
)

# ============================================================
# OVERALL RMSE ACROSS ALL STATIONS
# ============================================================

# Daily: use only rows with valid daily model and observation values
daily_valid = sample_df[
    sample_df['ssm_daily'].notna()
    & sample_df['obs'].notna()
]

# Hourly: use only rows with valid hourly model and observation values
hourly_valid = sample_df[
    sample_df['ssm_hourly'].notna()
    & sample_df['obs'].notna()
]

overall_rmse = {
    'SSM Daily': np.sqrt(
        np.mean(
            (daily_valid['ssm_daily'] - daily_valid['obs']) ** 2
        )
    ),

    'SSM Hourly': np.sqrt(
        np.mean(
            (hourly_valid['ssm_hourly'] - hourly_valid['obs']) ** 2
        )
    )
}

print()
print("=" * 60)
print("OVERALL RMSE — ALL STATIONS")
print("=" * 60)

for model, rmse in overall_rmse.items():
    print(f"{model:20s}: {rmse:.3f} mg/L")

print()
print(f"Daily valid observations:  {len(daily_valid)}")
print(f"Hourly valid observations: {len(hourly_valid)}")

rmse_difference = (
    overall_rmse['SSM Daily']
    - overall_rmse['SSM Hourly']
)

print()
print(f"Daily RMSE - Hourly RMSE: {rmse_difference:+.3f} mg/L")

# ============================================================
# OBSERVATIONAL VARIABILITY
# ============================================================

station_rmse['high_variability'] = (
    station_rmse['obs_std']
    >= std_threshold
)


# ============================================================
# LOAD GRID FOR MAP BACKGROUND
# ============================================================

Ldir = Lfun.Lstart(
    gridname='cas7'
)

grid_fn = Ldir['grid'] / 'grid.nc'

grid_ds = xr.open_dataset(
    grid_fn
)

lon_rho = grid_ds['lon_rho'].values
lat_rho = grid_ds['lat_rho'].values
h = grid_ds['h'].values
mask_rho = grid_ds['mask_rho'].values

# Mask land
h_plot = np.where(
    mask_rho == 1,
    h,
    np.nan
)

# ============================================================
# COLOR SCALES
# ============================================================

# Identical scale for daily and hourly RMSE.
# rmse_max = 2.0
# rmse_max = 0.12 # Salinity
rmse_max = 0.06 # Temp

# Symmetric scale for the daily-minus-hourly difference.
diff_max = np.nanmax(
    np.abs(station_rmse['rmse_diff'].to_numpy())
)

# if not np.isfinite(diff_max) or diff_max == 0:
#     diff_max = 0.1
# else:
#     diff_max = np.ceil(diff_max * 10) / 10
    
diff_max = 0.05 # Temp

diff_norm = TwoSlopeNorm(
    vmin=-diff_max,
    vcenter=0,
    vmax=diff_max,
)

orange_yellow_seafoam = LinearSegmentedColormap.from_list(
    'orange_yellow_seafoam',
    [
        '#C4511B',
        '#E69F00',
        '#F7F7F7',
        '#66BFA3',
        '#238B70',
    ],
)

# ============================================================
# PLOT: THREE SSM PANELS
# ============================================================

plt.close('all')

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 7),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)

low = ~station_rmse['high_variability']
high = station_rmse['high_variability']

# ------------------------------------------------------------
# PANEL 1: DAILY RMSE
# ------------------------------------------------------------

sc_daily = axes[0].scatter(
    station_rmse.loc[low, 'lon'],
    station_rmse.loc[low, 'lat'],
    c=station_rmse.loc[low, 'daily_rmse'],
    cmap='viridis',
    vmin=0,
    vmax=rmse_max,
    s=90,
    marker='o',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)

axes[0].scatter(
    station_rmse.loc[high, 'lon'],
    station_rmse.loc[high, 'lat'],
    c=station_rmse.loc[high, 'daily_rmse'],
    cmap='viridis',
    vmin=0,
    vmax=rmse_max,
    s=110,
    marker='^',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)


# ------------------------------------------------------------
# PANEL 2: HOURLY RMSE
# ------------------------------------------------------------

sc_hourly = axes[1].scatter(
    station_rmse.loc[low, 'lon'],
    station_rmse.loc[low, 'lat'],
    c=station_rmse.loc[low, 'hourly_rmse'],
    cmap='viridis',
    vmin=0,
    vmax=rmse_max,
    s=90,
    marker='o',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)

axes[1].scatter(
    station_rmse.loc[high, 'lon'],
    station_rmse.loc[high, 'lat'],
    c=station_rmse.loc[high, 'hourly_rmse'],
    cmap='viridis',
    vmin=0,
    vmax=rmse_max,
    s=110,
    marker='^',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)


# ------------------------------------------------------------
# PANEL 3: DAILY RMSE - HOURLY RMSE
# ------------------------------------------------------------

sc_diff = axes[2].scatter(
    station_rmse.loc[low, 'lon'],
    station_rmse.loc[low, 'lat'],
    c=station_rmse.loc[low, 'rmse_diff'],
    cmap=orange_yellow_seafoam,
    norm=diff_norm,
    s=90,
    marker='o',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)

axes[2].scatter(
    station_rmse.loc[high, 'lon'],
    station_rmse.loc[high, 'lat'],
    c=station_rmse.loc[high, 'rmse_diff'],
    cmap=orange_yellow_seafoam,
    norm=diff_norm,
    s=110,
    marker='^',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)
        
    
# ============================================================
# MAP BACKGROUND
# ============================================================

titles = [
    'SSM Daily-Matched RMSE',
    'SSM Hourly-Matched RMSE',
    'Daily RMSE - Hourly RMSE',
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

    ax.set_xlim(
        lon_low,
        lon_high
    )

    ax.set_ylim(
        lat_low,
        lat_high
    )

    pfun.dar(ax)

    ax.set_title(
        title,
        fontsize=12
    )

    ax.tick_params(
        labelsize=8
    )
    
# ============================================================
# LEGENDS AND COLORBARS
# ============================================================

# Observation variability legend
legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor='white',
        markeredgecolor='black',
        markersize=8,
        label=f'Obs. SD < {std_threshold:.1f} {units}'
    ),

    Line2D(
        [0], [0],
        marker='^',
        color='none',
        markerfacecolor='white',
        markeredgecolor='black',
        markersize=8,
        label=f'Obs. SD ≥ {std_threshold:.1f} {units}'
    )
]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
    frameon=True
)


# RMSE colorbar for panels 1 and 2.
cb_rmse = fig.colorbar(
    sc_daily,
    ax=axes[:2],
    orientation='horizontal',
    fraction=0.05,
    pad=0.08
)

cb_rmse.set_label(
    f'RMSE ({units})'
)

# Difference scale for panel 3.
cb_diff = fig.colorbar(
    sc_diff,
    ax=axes[2],
    orientation='horizontal',
    fraction=0.05,
    pad=0.08,
)

cb_diff.set_label(
    f'Daily RMSE - Hourly RMSE ({units})'
)

fig.subplots_adjust(
    bottom=0.20,
    wspace=0.02,
)

fig.suptitle(
    f'{var_label}: SSM Daily vs Hourly Matching ({year})',
    fontsize=15,
)

# ============================================================
# SAVE
# ============================================================

out_fn = out_dir / (
    f'{otype}_{year}_{vn}_'
    f'station_rmse_daily_vs_hourly.png'
)

fig.savefig(
    out_fn,
    dpi=300,
    bbox_inches='tight'
)

print(
    f'\nSaved figure to:\n{out_fn}'
)

plt.show()

# ============================================================
# CLOSE DATASETS
# ============================================================

grid_ds.close()
mask_ds.close()