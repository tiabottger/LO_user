# import things
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pylab as plt

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
in_dir = Ldir['LOu'] / 'intermodel_comparison' / 'bottom_DO'
out_dir = Ldir['LOu'] / 'intermodel_comparison' / 'bottom_DO'
Lfun.make_dir(out_dir)

##############################################################
##                      PROCESS DATA                        ##
##############################################################

plt.close('all')
print('Running....')

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

# dictionaries
hyp_area_bot = {}
hyp_area_bot146 = {}
time_dict = {}

for year in years:
    for gtagex in gtagexes:

        print(f'Processing {gtagex}, {region}, {year}')

        fp = in_dir / (gtagex + '_' + region + '_' + year + '_DO_info.nc')
        ds = xr.open_dataset(fp)

        DO_bot = ds['DO_bot'].values
        DO_bot146 = ds['DO_bot146'].values

        # True where hypoxic
        hyp_bot = DO_bot <= 2
        hyp_bot146 = DO_bot146 <= 2

        # hypoxic area through time [km2]
        area_bot = np.nansum(hyp_bot * DA[None, :, :], axis=(1, 2))
        area_bot146 = np.nansum(hyp_bot146 * DA[None, :, :], axis=(1, 2))

        key = gtagex + '_' + region + '_' + year

        hyp_area_bot[key] = area_bot
        hyp_area_bot146[key] = area_bot146
        time_dict[key] = pd.to_datetime(ds['ocean_time'].values)
        
##############################################################
##              PLOT HYPOXIC AREA TIME SERIES               ##
##############################################################

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

ax.set_title('Hypoxic Area Comparison: Bottom Cell vs Bottom 14.6%', fontsize=13)

plt.tight_layout()

out_fn = out_dir / 'hypoxic_area_bottom_vs_bottom146.png'
plt.savefig(out_fn, dpi=300)

print(f'Saved to {out_fn}')
print('Done')