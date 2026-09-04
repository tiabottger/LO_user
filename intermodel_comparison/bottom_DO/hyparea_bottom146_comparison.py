# import things
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pylab as plt
from scipy.spatial import cKDTree

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

import sys
from pathlib import Path
pth = Path(__file__).absolute().parent.parent.parent / 'LO' / 'pgrid'
if str(pth) not in sys.path:
    sys.path.append(str(pth))

Ldir = Lfun.Lstart()

##############################################################
##                       USER INPUTS                        ##
##############################################################

years = ['2014'] # add more years as needed
gtagexes = ['cas7_t1_x11ab']
region = 'pugetsoundDO'

# input/output directories
in_dir = Ldir['LOo'] / 'intermodel_comparison' / 'bottom_DO'
out_dir = Ldir['LOo'] / 'intermodel_comparison' / 'bottom_DO'
Lfun.make_dir(out_dir)

##############################################################
##                      PROCESS DATA                        ##
##############################################################

plt.close('all')
print('Running....')

basin_mask_ds = xr.open_dataset(Ldir['LOo'] / 'hypvol_for_intrmdl_cmprsn' / 'basin_masks_from_pugetsoundDObox.nc')
mask_ps = basin_mask_ds.mask_pugetsound.values

# get grid cell area from one of the original box files
fp_grid = Ldir['LOo'] / 'extract' / gtagexes[0] / 'box' / (region + '_2014.01.01_2014.12.31.nc')
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
# MAP Puget Sound MASK ONTO SSC GRID
# ============================================================
ds_LO = xr.open_dataset(
    in_dir / 'cas7_t1_x11ab_pugetsound_2014_bottom_DO_info.nc'
)

ds_SSC = xr.open_dataset(
    in_dir / 'SSC_2014_pugetsound_bottom_DO_info.nc'
)

lon_SSC = ds_SSC['nav_lon'].values
lat_SSC = ds_SSC['nav_lat'].values

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

# dictionaries
hyp_area_bot = {}
hyp_area_bot146 = {}
time_dict = {}
thick_bot_dict = {}
thick_bot146_dict = {}

for year in years:
    for gtagex in gtagexes:

        print(f'Processing {gtagex}, {region}, {year}')

        # fp = in_dir / (gtagex + '_' + region + '_' + year + '_bottom_DO_info.nc')
        fp = in_dir / 'SSC_2014_pugetsound_bottom_DO_info.nc'
        ds = xr.open_dataset(fp)

        DO_bot = ds['DO_bot'].values
        DO_bot146 = ds['DO_bot146'].values
        thick_bot = ds['thick_bot'].values
        thick_bot146 = ds['thick_bot146'].values

        # True where hypoxic
        hyp_bot = DO_bot <= 2
        hyp_bot146 = DO_bot146 <= 2

        # hypoxic area through time [km2]
        # area_bot = np.nansum(hyp_bot * DA_ps[None, :, :], axis=(1, 2))
        # area_bot146 = np.nansum(hyp_bot146 * DA_ps[None, :, :], axis=(1, 2))
        area_bot = np.nansum(hyp_bot * DA_SSC[None, :, :], axis=(1, 2))
        area_bot146 = np.nansum(hyp_bot146 * DA_SSC[None, :, :], axis=(1, 2))

        key = gtagex + '_' + region + '_' + year

        hyp_area_bot[key] = area_bot
        hyp_area_bot146[key] = area_bot146
        # time_dict[key] = pd.to_datetime(ds['ocean_time'].values)
        time_dict[key] = pd.to_datetime(ds_LO['ocean_time'].values)
        thick_bot_dict[key] = thick_bot
        thick_bot146_dict[key] = thick_bot146
##############################################################
##              PLOT HYPOXIC AREA TIME SERIES               ##
##############################################################
print('Plotting hypoxic area time series')
fig, ax = plt.subplots(figsize=(11, 4.5))

gtagex = gtagexes[0]

for year in years:

    key = gtagex + '_' + region + '_' + year
    dates = time_dict[key]

    ax.plot(
        dates,
        hyp_area_bot[key],
        linewidth=2,
        label='Bottom LiveOcean cell' if year == years[0] else None
    )

    ax.plot(
        dates,
        hyp_area_bot146[key],
        linewidth=2,
        linestyle='--',
        label='Bottom 14.6% layer' if year == years[0] else None
    )

ax.set_ylabel('Hypoxic Area (km$^2$)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)

ax.grid(visible=True, axis='both', color='silver', linestyle='--')
ax.legend(loc='upper right')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='both', labelsize=11)

ax.set_title('SSC Hypoxic Area Comparison: Bottom Cell vs Bottom 14.6%', fontsize=13)

plt.tight_layout()

out_fn = out_dir / 'ssc_hypoxic_area_bottom_vs_bottom146.png'
plt.savefig(out_fn, dpi=300)

##############################################################
##          PLOT MEAN THICKNESS DIFFERENCE MAP             ##
##############################################################
print('Plotting thickness difference map...')

# collect all years together
diff_all = []

gtagex = gtagexes[0]

# longitude and latitude for plotting
lon = box_ds.lon_rho.values
lat = box_ds.lat_rho.values

plon, plat = pfun.get_plon_plat(lon, lat)

for year in years:

    key = gtagex + '_' + region + '_' + year

    diff = (
        thick_bot146_dict[key]
        - thick_bot_dict[key]
    )

    diff_all.append(diff)

# concatenate all years in time
diff_all = np.concatenate(diff_all, axis=0)

# average over time
mean_diff = np.nanmean(diff_all, axis=0)

# remove land
mean_diff = np.where(mask_rho == 1, mean_diff, np.nan)

# make figure
fig, ax = plt.subplots(figsize=(6,8))

vmin = np.nanmin(mean_diff)
vmax = np.nanmax(mean_diff)

pcm = ax.pcolormesh(
    plon,
    plat,
    mean_diff,
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)

plt.colorbar(
    pcm,
    ax=ax,
    label='Thickness difference (m)'
)

pfun.dar(ax)

ax.set_xlim([-123.29,-122.1])
ax.set_ylim([46.95,48.6])

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

ax.set_title(
    'SSC Mean Bottom Layer Thickness Difference\n'
    '(Bottom 14.6% − Bottom Cell)'
)

plt.tight_layout()

plt.savefig(
    out_dir / 'ssc_bottom_layer_thickness_difference.png',
    dpi=300
)

print(f'Saved to {out_fn}')
print('Done')