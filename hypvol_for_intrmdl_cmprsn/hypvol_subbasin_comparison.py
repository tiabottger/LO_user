"""
Compare average bottom DO between multiple years
(Set up to run for 6 years)

"""

# import things
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.cm as cm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

import sys
from pathlib import Path
pth = Path(__file__).absolute().parent.parent.parent / 'LO' / 'pgrid'
if str(pth) not in sys.path:
    sys.path.append(str(pth))
import gfun

Gr = gfun.gstart()

Ldir = Lfun.Lstart()

##############################################################
##                       USER INPUTS                        ##
##############################################################

vn = 'oxygen'

years =  ['2015','2016','2017','2018','2019','2020']

# which  model run to look at?
gtagexes = ['cas7_t1_x11ab','cas7_t1noDIN_x11ab']
# gtagex cas7_t1noDIN_x11ab is no-loading, cas7_t1_x11ab is loading

# # where to put output figures
# out_dir = Ldir['LOo'] / 'chapter_2' / 'figures'
# Lfun.make_dir(out_dir)

region = 'pugetsoundDO'


##############################################################
##                      PROCESS DATA                        ##
##############################################################

plt.close('all')

print('Running....')

# initialize empty dictionaries
DO_bot_dict          = {} # bottom DO concentration [mg/L]
hyp_thick_dict       = {} # hypoxic thickness [m]
water_depth_dict     = {} # time-variable water depth [m]
one_mgL_thick_dict   = {} # thickness of water column that is <= 1 mg/L DO [m]
three_mgL_thick_dict = {} # thickness of water column that is <= 3 mg/L DO [m]

for year in years:
    for gtagex in gtagexes:
        # open dataset
        ds = xr.open_dataset(Ldir['LOo'] / 'hypvol_for_intrmdl_cmprsn' / (gtagex + '_' + region + '_' + year + '_DO_info.nc'))
        # get data from dataset
        DO_bot = ds['DO_bot'].values
        hyp_thick = ds['hyp_thick'].values
        water_depth_m = ds['depth_bot'].values * -1 # don't want negative depths
        one_mgL_thick = ds['one_mgL_thick'].values
        three_mgL_thick = ds['three_mgL_thick'].values
        # save values to dictionary
        DO_bot_dict[gtagex+region+year] = DO_bot
        hyp_thick_dict[gtagex+region+year] = hyp_thick
        water_depth_dict[gtagex+region+year] = water_depth_m
        one_mgL_thick_dict[gtagex+region+year] = one_mgL_thick
        three_mgL_thick_dict[gtagex+region+year] = three_mgL_thick

# get grid cell area
fp = Ldir['LOo'] / 'hypvol_for_intrmdl_cmprsn' / (region + '_2014.01.01_2014.12.31.nc')
box_ds = xr.open_dataset(fp)
DX = (box_ds.pm.values)**-1
DY = (box_ds.pn.values)**-1
DA = DX*DY*(1/1000)*(1/1000) # get area, but convert from m^2 to km^2

# read in masks
basin_mask_ds = xr.open_dataset(Ldir['LOo'] / 'hypvol_for_intrmdl_cmprsn' / 'basin_masks_from_pugetsoundDObox.nc')
mask_rho = basin_mask_ds.mask_rho.values
mask_ps = basin_mask_ds.mask_pugetsound.values
mask_hc = basin_mask_ds.mask_hoodcanal.values
mask_ss = basin_mask_ds.mask_southsound.values
mask_mb = basin_mask_ds.mask_mainbasin.values
mask_wb = basin_mask_ds.mask_whidbeybasin.values

# subbasin masks dictionary
subbasin_masks = {
    'PugetSound': mask_ps,
    'HoodCanal': mask_hc,
    'SouthSound': mask_ss,
    'MainBasin': mask_mb,
    'WhidbeyBasin': mask_wb
}

# initialize dictionary for hypoxic volume [km3]
hyp_vol = {}
water_volume = {}
onemgL_vol = {}
threemgL_vol = {}

for year in years:
    for gtagex in gtagexes:

        # get thicknesses (km)
        hyp_thick = hyp_thick_dict[gtagex + region + year] / 1000
        water_depth_km = water_depth_dict[gtagex + region + year] / 1000
        one_mgL_thick = one_mgL_thick_dict[gtagex + region + year] / 1000
        three_mgL_thick = three_mgL_thick_dict[gtagex + region + year] / 1000

        for subbasin, mask in subbasin_masks.items():

            # hypoxic volume
            hyp_thick_masked = hyp_thick * mask
            hyp_vol_timeseries = np.nansum(hyp_thick_masked * DA, axis=(1, 2))  # km^3
            hyp_vol[gtagex + region + subbasin + year] = hyp_vol_timeseries

            # total water volume
            water_depth_masked = water_depth_km * mask
            water_vol_timeseries = np.nansum(water_depth_masked * DA, axis=(1, 2))  # km^3
            water_volume[gtagex + region + subbasin + year] = water_vol_timeseries
            
            # 1 mg/L volume 
            one_mgL_masked = one_mgL_thick * mask
            onemgL_vol_timeseries = np.nansum(one_mgL_masked * DA, axis=(1, 2))  # km^3
            onemgL_vol[gtagex + region + subbasin + year] = onemgL_vol_timeseries

            # 3 mg/L volume
            three_mgL_masked = three_mgL_thick * mask
            threemgL_vol_timeseries = np.nansum(three_mgL_masked * DA, axis=(1, 2))  # km^3
            threemgL_vol[gtagex + region + subbasin + year] = threemgL_vol_timeseries


# get lon and lat for making pcolormesh plot
lon = basin_mask_ds.lon_rho.values
lat = basin_mask_ds.lat_rho.values
plon, plat = pfun.get_plon_plat(lon,lat)

##############################################################
##              PLOT HYPOXIC VOLUME TIME SERIES             ##
##############################################################

# Puget Sound bounds
xmin = -123.29
xmax = -122.1
ymin = 46.95
ymax = 48.6

# initialize figure
fig, (ax0, ax1) = plt.subplots(1,2,figsize = (12,5),gridspec_kw={'width_ratios': [1, 3.5]})
# fig, (ax0, ax1) = plt.subplots(1,2,figsize = (6,3),gridspec_kw={'width_ratios': [1, 2.8]})

# format figure
ax0.set_xlim([xmin,xmax])
ax0.set_ylim([ymin,ymax])
ax0.set_ylabel('Latitude', fontsize=12)
ax0.set_xlabel('Longitude', fontsize=12)
ax0.tick_params(axis='both', labelsize=12)
# plot map of Puget Sound
ax0.pcolormesh(plon, plat, np.where(mask_rho == 0, np.nan, mask_rho),
                vmin=0, vmax=10, cmap='Blues')
# Hood Canal
ax0.pcolormesh(plon, plat, np.where(mask_hc == 0, np.nan, mask_hc),
            vmin=0, vmax=2, cmap='RdPu' )
# South Sound
ax0.pcolormesh(plon, plat, np.where(mask_ss == 0, np.nan, mask_ss),
              vmin=0, vmax=3, cmap='magma' )
# Main Basin
ax0.pcolormesh(plon, plat, np.where(mask_mb == 0, np.nan, mask_mb),
              vmin=0, vmax=1.5, cmap='summer' )
# # Whidbey Basin
ax0.pcolormesh(plon, plat, np.where(mask_wb == 0, np.nan, mask_wb),
              vmin=0, vmax=3, cmap='cool' )

pfun.dar(ax0)
ax0.set_title('(a)', fontsize = 14, loc='left', fontweight='bold')

# Get nominal Puget Sound volume (non time-varying)
PS_vol = np.nansum(basin_mask_ds['h'].values/1000 * DA * mask_ps) # [km^3]
print('Puget Sound volume: {} km3'.format(round(PS_vol,1)))


subbasin_colors = {
    'PugetSound': cm.Blues(0.9),
    'HoodCanal': cm.RdPu(0.6),
    'SouthSound': cm.magma(0.3),
    'MainBasin': cm.summer(0.5),
    'WhidbeyBasin': cm.cool(0.3),
}

subbasins = list(subbasin_colors.keys())

gtagex = gtagexes[0]  # loading run only

# plot hypoxic volume timeseries
for year in years:

    startdate = year + '.01.01'
    enddate = year + '.12.31'
    dates = pd.date_range(start=startdate, end=enddate, freq='1d')
    dates_local = [pfun.get_dt_local(x) for x in dates]

    for subbasin in subbasin_colors.keys():

        key = gtagex + region + subbasin + year
        
        if year == years[0]:
            ax1.plot(dates_local, hyp_vol[key],
                    color=subbasin_colors[subbasin],
                    linewidth=2,
                    label=subbasin)
        else:
            ax1.plot(dates_local, hyp_vol[key],
                    color=subbasin_colors[subbasin],
                    linewidth=2)

# axis labels
ax1.set_ylabel('Hypoxic Volume (km$^3$)', fontsize=12)

# combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='upper right')

# format figure
ax1.grid(visible=True, axis='both', color='silver', linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.tick_params(axis='x', rotation=30, labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.set_ylim(0,5.5)

plt.title('(b)', fontsize = 14, loc='left', fontweight='bold')
# create time vector
startdate = years[0]+'.01.01'
enddate = years[-1]+'.12.31'
dates = pd.date_range(start= startdate, end= enddate, freq= '1d')
dates_local = [pfun.get_dt_local(x) for x in dates]
ax1.set_xlim([dates_local[0],dates_local[-1]])

plt.tight_layout()
# plt.show()
plt.savefig('hypvol_subbasin_comparison_fig.png')

print('Done')
