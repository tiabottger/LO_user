"""
Plot station-mean model errors for LiveOcean, SalishSeaCast, and SSM.

Panels:
1. LiveOcean RMSE
2. SalishSeaCast RMSE
3. SSM RMSE
4. Best model at each station

The fourth panel shows which model has the lowest station RMSE.
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
vn = 'DO'

# Label and optional conversion
var_label = 'Dissolved oxygen'
units = r'mg L$^{-1}$'

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
ssm_key = 'ssm'

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

in_fn = in_dir / (
    f'combined_{otype}_{year}_cas7_t1_x11ab_ssc_ssmpnnl.pkl'
)

df0_dict = pickle.load(
    open(in_fn, 'rb')
)

# Remove non-DataFrame entries such as metadata
df0_dict = {
    key: df
    for key, df in df0_dict.items()
    if isinstance(df, pd.DataFrame)
}

# Check required keys
required_keys = [
    obs_key,
    lo_key,
    ssc_key,
    ssm_key
]

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
ssm_df = df0_dict[ssm_key].copy()

# ============================================================
# CHECK ALIGNMENT
# ============================================================

lengths = {
    'obs': len(obs_df),
    'lo': len(lo_df),
    'ssc': len(ssc_df),
    'ssm': len(ssm_df)
}

print('\nDataFrame lengths:')
for key, value in lengths.items():
    print(f'  {key}: {value}')

if len(set(lengths.values())) != 1:
    raise ValueError(
        'Observation, LiveOcean, SalishSeaCast, and SSM '
        'DataFrames have different lengths.'
    )

# ============================================================
# CREATE SAMPLE-LEVEL DATAFRAME
# ============================================================

sample_df = pd.DataFrame({
    'name': obs_df['name'],
    'lon': obs_df['lon'],
    'lat': obs_df['lat'],

    'obs': (
        obs_df[vn]
        * 0.032
    ),

    'lo': (
        lo_df[vn]
        * 0.032
    ),

    'ssc': (
        ssc_df[vn]
        * 0.032
    ),

    'ssm': (
        ssm_df[vn]
        * 0.032
    ),
})

sample_df['name'] = (
    sample_df['name']
    .astype(str)
)

# Remove invalid values
sample_df = (
    sample_df
    .replace([np.inf, -np.inf], np.nan)
    .dropna(
        subset=[
            'obs',
            'lo',
            'ssc',
            'ssm'
        ]
    )
)

print(
    f'\nNumber of valid samples before '
    f'Puget Sound mask: {len(sample_df)}'
)

# ============================================================
# FILTER TO PUGET SOUND BASIN MASK
# ============================================================

mask_lon = mask_ds['lon_rho'].values
mask_lat = mask_ds['lat_rho'].values
puget_sound_mask = (
    mask_ds['mask_pugetsound'].values
)

# Build nearest-neighbor tree from valid grid coordinates
grid_points = np.column_stack((
    mask_lon.ravel(),
    mask_lat.ravel(),
))

valid_grid = (
    np.isfinite(grid_points[:, 0])
    & np.isfinite(grid_points[:, 1])
)

tree = cKDTree(
    grid_points[valid_grid]
)

# Observation/station sample coordinates
sample_points = np.column_stack((
    sample_df['lon'].to_numpy(),
    sample_df['lat'].to_numpy(),
))

# Find nearest valid grid point for each sample
_, nearest_valid_index = tree.query(
    sample_points
)

# Convert indices from valid subset back
# to flattened grid indices
valid_flat_indices = np.flatnonzero(
    valid_grid
)

nearest_flat_index = (
    valid_flat_indices[
        nearest_valid_index
    ]
)

eta_index, xi_index = np.unravel_index(
    nearest_flat_index,
    mask_lon.shape,
)

# Determine whether each sample is inside
# Puget Sound
inside_puget_sound = (
    puget_sound_mask[
        eta_index,
        xi_index
    ] == 1
)

sample_df = sample_df.loc[
    inside_puget_sound
].copy()

print(
    f'Number of valid samples inside '
    f'Puget Sound: {len(sample_df)}'
)

# ============================================================
# SQUARED ERRORS
# ============================================================

sample_df['lo_sq_error'] = (
    sample_df['lo']
    - sample_df['obs']
) ** 2

sample_df['ssc_sq_error'] = (
    sample_df['ssc']
    - sample_df['obs']
) ** 2

sample_df['ssm_sq_error'] = (
    sample_df['ssm']
    - sample_df['obs']
) ** 2

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

        obs_std=('obs', 'std'),

        lo_mse=('lo_sq_error', 'mean'),
        ssc_mse=('ssc_sq_error', 'mean'),
        ssm_mse=('ssm_sq_error', 'mean'),
    )
)

# Minimum sample requirement
station_rmse = station_rmse[
    station_rmse['n'] >= min_samples
].copy()

# ============================================================
# CALCULATE RMSE
# ============================================================

station_rmse['lo_rmse'] = np.sqrt(
    station_rmse['lo_mse']
)

station_rmse['ssc_rmse'] = np.sqrt(
    station_rmse['ssc_mse']
)

station_rmse['ssm_rmse'] = np.sqrt(
    station_rmse['ssm_mse']
)

# ============================================================
# OBSERVATIONAL VARIABILITY
# ============================================================

station_rmse['high_variability'] = (
    station_rmse['obs_std']
    >= std_threshold
)

# ============================================================
# DETERMINE BEST MODEL
# ============================================================

rmse_columns = {
    'LiveOcean': 'lo_rmse',
    'SalishSeaCast': 'ssc_rmse',
    'SSM': 'ssm_rmse'
}

rmse_array = station_rmse[
    [
        'lo_rmse',
        'ssc_rmse',
        'ssm_rmse'
    ]
].to_numpy()

model_names = np.array([
    'LiveOcean',
    'SalishSeaCast',
    'SSM'
])

best_model_index = np.argmin(
    rmse_array,
    axis=1
)

station_rmse['best_model'] = (
    model_names[best_model_index]
)

# ============================================================
# PRINT STATION RESULTS
# ============================================================

print('\nStation RMSE:')
print(
    station_rmse[
        [
            'name',
            'n',
            'obs_std',
            'lo_rmse',
            'ssc_rmse',
            'ssm_rmse',
            'best_model'
        ]
    ]
    .sort_values('name')
    .to_string(index=False)
)

print('\nNumber of stations by best model:')
print(
    station_rmse['best_model']
    .value_counts()
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
# COLOR LIMITS
# ============================================================

# Keep the same RMSE scale across all three models
rmse_max = 2.0

# ============================================================
# BEST MODEL COLORS
# ============================================================

best_model_colors = {
    'LiveOcean': 'tab:red',
    'SalishSeaCast': 'tab:blue',
    'SSM': 'tab:green'
}

best_model_order = [
    'LiveOcean',
    'SalishSeaCast',
    'SSM'
]

# ============================================================
# PLOT
# ============================================================

plt.close('all')

fig, axes = plt.subplots(
    1,
    4,
    figsize=(22, 7),
    constrained_layout=True,
    sharex=True,
    sharey=True
)

# ------------------------------------------------------------
# Common station groups
# ------------------------------------------------------------

low = (
    ~station_rmse['high_variability']
)

high = (
    station_rmse['high_variability']
)

# ============================================================
# PANEL 1: LIVE OCEAN RMSE
# ============================================================

sc_lo = axes[0].scatter(
    station_rmse.loc[low, 'lon'],
    station_rmse.loc[low, 'lat'],
    c=station_rmse.loc[low, 'lo_rmse'],
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
    c=station_rmse.loc[high, 'lo_rmse'],
    cmap='viridis',
    vmin=0,
    vmax=rmse_max,
    s=110,
    marker='^',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)

# ============================================================
# PANEL 2: SALISHSEACAST RMSE
# ============================================================

axes[1].scatter(
    station_rmse.loc[low, 'lon'],
    station_rmse.loc[low, 'lat'],
    c=station_rmse.loc[low, 'ssc_rmse'],
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
    c=station_rmse.loc[high, 'ssc_rmse'],
    cmap='viridis',
    vmin=0,
    vmax=rmse_max,
    s=110,
    marker='^',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)

# ============================================================
# PANEL 3: SSM RMSE
# ============================================================

axes[2].scatter(
    station_rmse.loc[low, 'lon'],
    station_rmse.loc[low, 'lat'],
    c=station_rmse.loc[low, 'ssm_rmse'],
    cmap='viridis',
    vmin=0,
    vmax=rmse_max,
    s=90,
    marker='o',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)

axes[2].scatter(
    station_rmse.loc[high, 'lon'],
    station_rmse.loc[high, 'lat'],
    c=station_rmse.loc[high, 'ssm_rmse'],
    cmap='viridis',
    vmin=0,
    vmax=rmse_max,
    s=110,
    marker='^',
    edgecolor='k',
    linewidth=0.6,
    zorder=10,
)

# ============================================================
# PANEL 4: BEST MODEL
# COLOR = MODEL IDENTITY + RMSE MAGNITUDE
# ============================================================

from matplotlib.colors import to_rgb


def rmse_color(
    model,
    rmse,
    rmse_max
):
    """
    Return a model-specific color whose
    saturation/darkness increases with RMSE.
    """

    base_colors = {
        'LiveOcean': 'tab:red',
        'SalishSeaCast': 'tab:blue',
        'SSM': 'tab:green'
    }

    base = np.array(
        to_rgb(base_colors[model])
    )

    # Normalize RMSE to 0-1
    frac = np.clip(
        rmse / rmse_max,
        0,
        1
    )

    # Blend with white.
    # Low RMSE -> light color
    # High RMSE -> base color
    color = (
        (1 - frac) * np.ones(3)
        + frac * base
    )

    return color

for model in best_model_order:

    model_mask = (
        station_rmse['best_model']
        == model
    )

    rmse_column = rmse_columns[model]

    # --------------------------------------------------------
    # Low variability
    # --------------------------------------------------------

    model_low = (
        model_mask
        & ~station_rmse['high_variability']
    )

    for idx in station_rmse.index[model_low]:

        rmse_value = (
            station_rmse.loc[
                idx,
                rmse_column
            ]
        )

        axes[3].scatter(
            station_rmse.loc[idx, 'lon'],
            station_rmse.loc[idx, 'lat'],
            color=rmse_color(
                model,
                rmse_value,
                rmse_max
            ),
            s=90,
            marker='o',
            edgecolor='k',
            linewidth=0.6,
            zorder=10
        )

    # --------------------------------------------------------
    # High variability
    # --------------------------------------------------------

    model_high = (
        model_mask
        & station_rmse['high_variability']
    )

    for idx in station_rmse.index[model_high]:

        rmse_value = (
            station_rmse.loc[
                idx,
                rmse_column
            ]
        )

        axes[3].scatter(
            station_rmse.loc[idx, 'lon'],
            station_rmse.loc[idx, 'lat'],
            color=rmse_color(
                model,
                rmse_value,
                rmse_max
            ),
            s=110,
            marker='^',
            edgecolor='k',
            linewidth=0.6,
            zorder=10
        )
        
    
# ============================================================
# MAP BACKGROUND
# ============================================================

titles = [
    'LiveOcean RMSE',
    'SalishSeaCast RMSE',
    'SSM RMSE',
    'Best Model'
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
# Model colors
# ============================================================

model_colors = {
    'LiveOcean': 'tab:red',
    'SalishSeaCast': 'tab:blue',
    'SSM': 'tab:green'
}

model_order = ['LiveOcean', 'SalishSeaCast', 'SSM']

# ============================================================
# LEGENDS AND COLORBARS
# ============================================================

from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb


# ------------------------------------------------------------
# 1. Observation variability legend
# ------------------------------------------------------------

legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor='white',
        markeredgecolor='k',
        markersize=8,
        label=f'Obs. SD < {std_threshold:.1f} {units}'
    ),
    Line2D(
        [0], [0],
        marker='^',
        color='none',
        markerfacecolor='white',
        markeredgecolor='k',
        markersize=8,
        label=f'Obs. SD ≥ {std_threshold:.1f} {units}'
    )
]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.015),
    ncol=2,
    frameon=True,
    fontsize=9
)


# ------------------------------------------------------------
# 2. ORIGINAL VIRIDIS RMSE COLORBAR
#    Keep this for panels 1–3
# ------------------------------------------------------------

cb = fig.colorbar(
    sc_lo,
    ax=axes[:3],
    orientation='horizontal',
    fraction=0.05,
    pad=0.08
)

cb.set_label(f'RMSE ({units})')


# ------------------------------------------------------------
# 3. Make room underneath panel 4
# ------------------------------------------------------------

fig.subplots_adjust(
    bottom=0.25,
    wspace=0.08
)


# ------------------------------------------------------------
# 4. MODEL-SPECIFIC GRADIENTS
#    Only underneath panel 4
# ------------------------------------------------------------

rmse_values = np.linspace(0, rmse_max, 100)

# Get the position of panel 4 in figure coordinates
pos4 = axes[3].get_position()

# Width and height of each gradient
gradient_width = pos4.width * 0.80
gradient_height = 0.025

# Center the gradients underneath panel 4
gradient_left = (
    pos4.x0
    + (pos4.width - gradient_width) / 2
)

# Vertical positions
gradient_y = [
    0.110,   # LiveOcean
    0.065,   # SalishSeaCast
    0.020    # SSM
]

for model, y in zip(model_order, gradient_y):

    gradient_ax = fig.add_axes([
        gradient_left,
        y,
        gradient_width,
        gradient_height
    ])

    colors = np.array([
        rmse_color(model, rmse, rmse_max)
        for rmse in rmse_values
    ])

    gradient = colors.reshape(
        1,
        len(rmse_values),
        3
    )

    gradient_ax.imshow(
        gradient,
        aspect='auto',
        extent=[0, rmse_max, 0, 1]
    )

    gradient_ax.set_xlim(0, rmse_max)
    gradient_ax.set_ylim(0, 1)

    # Model name on the left
    gradient_ax.text(
        -0.02 * rmse_max,
        0.5,
        model,
        ha='right',
        va='center',
        fontsize=8,
        transform=gradient_ax.transData
    )

    gradient_ax.set_yticks([])

    gradient_ax.set_xticks(
        np.linspace(0, rmse_max, 5)
    )

    gradient_ax.set_xticklabels(
        [f'{x:.1f}' for x in np.linspace(0, rmse_max, 5)],
        fontsize=7
    )

    gradient_ax.tick_params(
        axis='x',
        length=2,
        pad=1
    )

    # Only show x-axis label on bottom gradient
    if model == 'SSM':
        gradient_ax.set_xlabel(
            f'RMSE of lowest-error model ({units})',
            fontsize=8,
            labelpad=1
        )

    # Border
    for spine in gradient_ax.spines.values():
        spine.set_visible(True)

# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    f'{var_label} Station RMSE ({year})',
    fontsize=15
)

# ============================================================
# SAVE
# ============================================================

out_fn = out_dir / (
    f'{otype}_{year}_{vn}_'
    f'station_rmse_LO_SSC_SSM.png'
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