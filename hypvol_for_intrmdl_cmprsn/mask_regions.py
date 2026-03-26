"""
One-time script used to generate .nc files
with masks of the different basins in Puget Sound
based on a box extraction

"""

# imports
from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

import sys 
import xarray as xr
from time import time
import numpy as np
from xarray import Dataset

import matplotlib.pyplot as plt

import sys
from pathlib import Path
pth = Path(__file__).absolute().parent.parent.parent.parent / 'LO' / 'pgrid'
if str(pth) not in sys.path:
    sys.path.append(str(pth))
import gfun

Gr = gfun.gstart()

Ldir = Lfun.Lstart()

#########################################

# load box extraction of Puget Sound
fp = Ldir['LOo'] / 'hypvol_for_intrmdl_cmprsn' / 'pugetsoundDO_2014.01.01_2014.12.31.nc'
box_ds = xr.open_dataset(fp)
mask_rho = box_ds.mask_rho.values              # 0 = land 1 = water
xrho = box_ds.coords['lon_rho'].values
yrho = box_ds.coords['lat_rho'].values
h = box_ds['h'].values

lon = xrho
lat = yrho
plon, plat = pfun.get_plon_plat(lon,lat)

########################################
# create Hood Canal mask

# make a mask for Hood Canal basin
mask_hc = np.ones(np.shape(mask_rho)) # copy shape of original land mask
mask_hc[mask_rho==0] = 0 # copy values of original land mask
mask_hc[lat>47.83] = 0
mask_hc[lat<47.3] = 0
mask_hc[lon>-122.6] = 0

cond = (lat >= 47.53) & (lat <= 47.725) & (lon > -122.71)
mask_hc[cond] = 0

cond = (lat >= 47.28) & (lat <= 47.38) & (lon > -122.88)
mask_hc[cond] = 0

########################################
# create South Sound mask

# make a mask for South Sound basin
mask_ss = np.ones(np.shape(mask_rho)) # copy shape of original land mask
mask_ss[mask_rho==0] = 0 # copy values of original land mask
mask_ss[mask_hc==1] = 0 # copy values of Hood Canal mask
mask_ss[lat>47.4] = 0
mask_ss[lon>-122.54] = 0


cond = (lon >= -122.62) & (lon <= -122.52) & (lat > 47.275)
mask_ss[cond] = 0

########################################
# create Main Basin mask

# make a mask for Main basin
mask_mb = np.ones(np.shape(mask_rho)) # copy shape of original land mask
mask_mb[mask_rho==0] = 0 # copy values of original land mask
mask_mb[mask_hc==1] = 0 # copy values of Hood Canal mask
mask_mb[mask_ss==1] = 0 # copy values of South Sound mask
mask_mb[lon<-122.82] = 0

below = lat > (1 * lon + 170.895)
mask_mb[below] = 0

below = lat > (-1 * lon - 74.42)
mask_mb[below] = 0

cond = (lat >= 48.02) & (lat <= 48.14) & (lon > -122.58)
mask_mb[cond] = 0

cond = (lon >= -122.68) & (lon <= -122.64) & (lat > 48.22)
mask_mb[cond] = 0

########################################
# create Whibdey Basin mask

# make a mask for Whidbey basin
mask_wb = np.ones(np.shape(mask_rho)) # copy shape of original land mask
mask_wb[mask_rho==0] = 0 # copy values of original land mask
mask_wb[mask_hc==1] = 0 # copy values of Hood Canal mask
mask_wb[mask_ss==1] = 0 # copy values of South Sound mask
mask_wb[mask_mb==1] = 0 # copy values of Main Basin mask
mask_wb[lon<-122.75] = 0

cond = (lat >= 48.14) & (lat <= 48.21) & (lon < -122.7)
mask_wb[cond] = 0

cond = (lat >= 48.25) & (lat <= 48.50) & (lon < -122.66)
mask_wb[cond] = 0

cond = (lon >= -122.8) & (lon <= -122.38) & (lat > 48.46)
mask_wb[cond] = 0

########################################
# create Puget Sound mask

# make a mask for Puget Sound (just sum of all other basin masks)
mask_ps = mask_hc + mask_ss + mask_wb + mask_mb

# #########################################
# # plot mask

plt.close('all')
# Puget Sound bounds
xmin = -123.29
xmax = -122.1
ymin = 46.95
ymax = 48.50#48.93


# initialize figure
fig, ax = plt.subplots(1,1,figsize = (8,8))

# add basins
# full region
# ax.pcolormesh(plon, plat, np.where(mask_rho == 0, np.nan, mask_rho), cmap='Paired' )
# Hood Canal
ax.pcolormesh(plon, plat, np.where(mask_hc == 0, np.nan, mask_hc),
            vmin=0, vmax=2, cmap='RdPu' )
# South Sound
ax.pcolormesh(plon, plat, np.where(mask_ss == 0, np.nan, mask_ss),
              vmin=0, vmax=3, cmap='magma' )
# Main Basin
ax.pcolormesh(plon, plat, np.where(mask_mb == 0, np.nan, mask_mb),
              vmin=0, vmax=1.5, cmap='summer' )
# # Whidbey Basin
ax.pcolormesh(plon, plat, np.where(mask_wb == 0, np.nan, mask_wb),
              vmin=0, vmax=3, cmap='cool' )

# format figure
# ax.set_xlim([xmin,xmax])
# ax.set_ylim([ymin,ymax])
ax.set_ylabel('Latitude', fontsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.tick_params(axis='both', labelsize=12)
pfun.dar(ax)

######################################
# save masks

new_ds = Dataset()

new_ds['lat_rho'] = (('eta_rho', 'xi_rho'),yrho,{'units':'degree_north'})
new_ds['lat_rho'].attrs['standard_name'] = 'grid_latitude_at_cell_center'
new_ds['lat_rho'].attrs['long_name'] = 'latitude of RHO-points'
new_ds['lat_rho'].attrs['field'] = 'lat_rho'

new_ds['lon_rho'] = (('eta_rho', 'xi_rho'),xrho,{'units': 'degree_east'})
new_ds['lon_rho'].attrs['standard_name'] = 'grid_longitude_at_cell_center'
new_ds['lon_rho'].attrs['long_name'] = 'longitude of RHO-points'
new_ds['lon_rho'].attrs['field'] = 'lon_rho'
 
new_ds['h'] = (('eta_rho', 'xi_rho'),h,{'units': 'm'})
new_ds['h'].attrs['standard_name'] = 'sea_floor_depth'
new_ds['h'].attrs['long_name'] = 'time_independent bathymetry'
new_ds['h'].attrs['field'] = 'bathymetry'
new_ds['h'].attrs['grid'] =  'cas7'

new_ds['mask_rho'] = (('eta_rho', 'xi_rho'),mask_rho,{'units': 'm'})
new_ds['mask_rho'].attrs['standard_name'] = 'land_sea_mask_at_cell_center'
new_ds['mask_rho'].attrs['long_name'] = 'mask on RHO-points'
new_ds['mask_rho'].attrs['flag_values'] = np.array([0.,1.])
new_ds['mask_rho'].attrs['flag_meanings'] = 'land water'
new_ds['mask_rho'].attrs['grid'] =  'cas7'

new_ds['mask_hoodcanal'] = (('eta_rho', 'xi_rho'),mask_hc,{'units': 'm'})
new_ds['mask_hoodcanal'].attrs['standard_name'] = 'basin_mask_at_cell_center'
new_ds['mask_hoodcanal'].attrs['long_name'] = 'mask on RHO-points'
new_ds['mask_hoodcanal'].attrs['flag_values'] = np.array([0.,1.])
new_ds['mask_hoodcanal'].attrs['flag_meanings'] = 'notbasin basin'
new_ds['mask_hoodcanal'].attrs['grid'] =  'cas7'

new_ds['mask_southsound'] = (('eta_rho', 'xi_rho'),mask_ss,{'units': 'm'})
new_ds['mask_southsound'].attrs['standard_name'] = 'basin_mask_at_cell_center'
new_ds['mask_southsound'].attrs['long_name'] = 'mask on RHO-points'
new_ds['mask_southsound'].attrs['flag_values'] = np.array([0.,1.])
new_ds['mask_southsound'].attrs['flag_meanings'] = 'notbasin basin'
new_ds['mask_southsound'].attrs['grid'] =  'cas7'

new_ds['mask_mainbasin'] = (('eta_rho', 'xi_rho'),mask_mb,{'units': 'm'})
new_ds['mask_mainbasin'].attrs['standard_name'] = 'basin_mask_at_cell_center'
new_ds['mask_mainbasin'].attrs['long_name'] = 'mask on RHO-points'
new_ds['mask_mainbasin'].attrs['flag_values'] = np.array([0.,1.])
new_ds['mask_mainbasin'].attrs['flag_meanings'] = 'notbasin basin'
new_ds['mask_mainbasin'].attrs['grid'] =  'cas7'

new_ds['mask_whidbeybasin'] = (('eta_rho', 'xi_rho'),mask_wb,{'units': 'm'})
new_ds['mask_whidbeybasin'].attrs['standard_name'] = 'basin_mask_at_cell_center'
new_ds['mask_whidbeybasin'].attrs['long_name'] = 'mask on RHO-points'
new_ds['mask_whidbeybasin'].attrs['flag_values'] = np.array([0.,1.])
new_ds['mask_whidbeybasin'].attrs['flag_meanings'] = 'notbasin basin'
new_ds['mask_whidbeybasin'].attrs['grid'] =  'cas7'

new_ds['mask_pugetsound'] = (('eta_rho', 'xi_rho'),mask_ps,{'units': 'm'})
new_ds['mask_pugetsound'].attrs['standard_name'] = 'basin_mask_at_cell_center'
new_ds['mask_pugetsound'].attrs['long_name'] = 'mask on RHO-points'
new_ds['mask_pugetsound'].attrs['flag_values'] = np.array([0.,1.])
new_ds['mask_pugetsound'].attrs['flag_meanings'] = 'notbasin basin'
new_ds['mask_pugetsound'].attrs['grid'] =  'cas7'

fn_f = Ldir['LOo'] / 'hypvol_for_intrmdl_cmprsn' / 'basin_masks_from_pugetsoundDObox.nc'
new_ds.to_netcdf(fn_f)