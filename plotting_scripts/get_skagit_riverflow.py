"""
Get Skagit River flow
"""

from subprocess import Popen as Po
from subprocess import PIPE as Pi
from matplotlib.markers import MarkerStyle
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import math
import csv
import cmocean
import matplotlib.pylab as plt
import gsw
import pickle

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

Ldir = Lfun.Lstart()

##########################################################
##                    Define inputs                     ##
##########################################################

gtagex = 'cas7_t0_x4b'
jobname = 'twentyoneinlets'
# startdate = '2017.01.01'
# enddate = '2017.01.02'

river = 'skagit'

dsf = Ldir['ds_fmt']

# ##########################################################
# ##              Get stations and gtagexes               ##
# ##########################################################

# # parse gtagex
# gridname, tag, ex_name = gtagex.split('_')
# Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

# # where to put output figures
# out_dir = Ldir['LOo'] / 'skagit_riverflow'
# Lfun.make_dir(out_dir)

# ##########################################################
# ##     calculate river transports for 2014 - 2015       ##
# ##########################################################

# frc = 'traps00'

# startdate = '2014.01.01'
# enddate = '2015.12.31'

# # list of variables to extract
# vn_list = ['transport']

# dt0 = datetime.strptime(startdate, Lfun.ds_fmt)
# dt1 = datetime.strptime(enddate, Lfun.ds_fmt)
# ndays = (dt1-dt0).days + 1

# # make mds_list: list of datestrings (e.g. 2021.01.01) to loop over
# mds_list = []
# mdt = dt0
# while mdt <= dt1:
#     mds_list.append(datetime.strftime(mdt, Lfun.ds_fmt))
#     mdt = mdt + timedelta(days=1)

# # get list of river names
# # (this is a bit titchy because of NetCDF 3 limitations on strings, forcing them
# # to be arrays of characters)
# mds = mds_list[0]
# # fn = Ldir['LOo'] / 'forcing' / gridname / ('f' + mds) / frc / 'rivers.nc'
# fn = '/dat1/parker/LO_output/forcing/' + gridname + '/f' + mds + '/' +  frc + '/rivers.nc'
# ds = xr.open_dataset(fn)
# rn = ds['river_name'].values
# NR = rn.shape[0]
# riv_name_list = list(rn)

# NT = len(mds_list)

# # get state variable values
# nanmat = np.nan * np.ones((NT, NR))
# v_dict = dict()
# for vn in vn_list:
#     v_dict[vn] = nanmat.copy()
# for tt,mds in enumerate(mds_list):
#     this_dt = datetime.strptime(mds, Lfun.ds_fmt)
#     if this_dt.day == 1 and this_dt.month == 1:
#         print(' Year = %d' % (this_dt.year))
#     # fn = Ldir['LOo'] / 'forcing' / gridname / ('f' + mds) / frc / 'rivers.nc'
#     fn = '/dat1/parker/LO_output/forcing/' + gridname + '/f' + mds + '/' +  frc + '/rivers.nc'
#     ds = xr.open_dataset(fn)
#     # The river transport is given at noon of a number of days surrounding the forcing date.
#     # Here we find the index of the time for the day "mds".
#     RT = pd.to_datetime(ds['river_time'].values)
#     mdt = this_dt + timedelta(hours=12)
#     mask = RT == mdt
#     for vn in vn_list:
#         if vn == 'transport':
#             v_dict[vn][tt,:] = ds['river_' + vn][mask,:]
#         else:
#             # the rest of the variables allow for depth variation, but we
#             # don't use this, so, just use the bottom value
#             v_dict[vn][tt,:] = ds['river_' + vn][mask,0,:]
#     ds.close()

# # make transport positive
# v_dict['transport'] = np.abs(v_dict['transport'])

# # store output in an xarray Dataset
# mdt_list = [(datetime.strptime(item, Lfun.ds_fmt) + timedelta(hours=12)) for item in mds_list]
# times = pd.Index(mdt_list)

# # create dataset
# x = xr.Dataset(coords={'time': times,'riv': riv_name_list})

# # add state variables
# for vn in vn_list:
#     v = v_dict[vn]
#     x[vn] = (('time','riv'), v)

# x.to_netcdf('2014_2015_rivflow.nc')

# ##########################################################
# ##     calculate river transports for 2016 - 2019       ##
# ##########################################################

# frc = 'trapsF00'

# startdate = '2016.01.01'
# enddate = '2019.12.31'

# # list of variables to extract
# vn_list = ['transport']

# dt0 = datetime.strptime(startdate, Lfun.ds_fmt)
# dt1 = datetime.strptime(enddate, Lfun.ds_fmt)
# ndays = (dt1-dt0).days + 1

# # make mds_list: list of datestrings (e.g. 2021.01.01) to loop over
# mds_list = []
# mdt = dt0
# while mdt <= dt1:
#     mds_list.append(datetime.strftime(mdt, Lfun.ds_fmt))
#     mdt = mdt + timedelta(days=1)

# # get list of river names
# # (this is a bit titchy because of NetCDF 3 limitations on strings, forcing them
# # to be arrays of characters)
# mds = mds_list[0]
# # fn = Ldir['LOo'] / 'forcing' / gridname / ('f' + mds) / frc / 'rivers.nc'
# fn = '/dat1/parker/LO_output/forcing/' + gridname + '/f' + mds + '/' +  frc + '/rivers.nc'
# ds = xr.open_dataset(fn)
# rn = ds['river_name'].values
# NR = rn.shape[0]
# riv_name_list = list(rn)

# NT = len(mds_list)

# # get state variable values
# nanmat = np.nan * np.ones((NT, NR))
# v_dict = dict()
# for vn in vn_list:
#     v_dict[vn] = nanmat.copy()
# for tt,mds in enumerate(mds_list):
#     this_dt = datetime.strptime(mds, Lfun.ds_fmt)
#     if this_dt.day == 1 and this_dt.month == 1:
#         print(' Year = %d' % (this_dt.year))
#     # fn = Ldir['LOo'] / 'forcing' / gridname / ('f' + mds) / frc / 'rivers.nc'
#     fn = '/dat1/parker/LO_output/forcing/' + gridname + '/f' + mds + '/' +  frc + '/rivers.nc'
#     ds = xr.open_dataset(fn)
#     # The river transport is given at noon of a number of days surrounding the forcing date.
#     # Here we find the index of the time for the day "mds".
#     RT = pd.to_datetime(ds['river_time'].values)
#     mdt = this_dt + timedelta(hours=12)
#     mask = RT == mdt
#     for vn in vn_list:
#         if vn == 'transport':
#             v_dict[vn][tt,:] = ds['river_' + vn][mask,:]
#         else:
#             # the rest of the variables allow for depth variation, but we
#             # don't use this, so, just use the bottom value
#             v_dict[vn][tt,:] = ds['river_' + vn][mask,0,:]
#     ds.close()

# # make transport positive
# v_dict['transport'] = np.abs(v_dict['transport'])

# # store output in an xarray Dataset
# mdt_list = [(datetime.strptime(item, Lfun.ds_fmt) + timedelta(hours=12)) for item in mds_list]
# times = pd.Index(mdt_list)

# # create dataset
# x = xr.Dataset(coords={'time': times,'riv': riv_name_list})

# # add state variables
# for vn in vn_list:
#     v = v_dict[vn]
#     x[vn] = (('time','riv'), v)

# x.to_netcdf('2016_2019_rivflow.nc')

##########################################################
##                       Plot data                      ##
##########################################################

plt.close('all')

fig, ax = plt.subplots(1,1,figsize = (5,2))

ds_1415 = xr.open_dataset(Ldir['LOo'] / 'skagit_riverflow' / '2014_2015_rivflow.nc')
ds_1619 = xr.open_dataset(Ldir['LOo'] / 'skagit_riverflow' / '2016_2019_rivflow.nc')

# get Skagit river flow only
skagit_2014 = ds_1415.loc[dict(riv='skagit')].loc[dict(time='2014')]['transport'].values
skagit_2015 = ds_1415.loc[dict(riv='skagit')].loc[dict(time='2015')]['transport'].values
skagit_2016 = ds_1619.loc[dict(riv='skagit')].loc[dict(time='2016')]['transport'].values
skagit_2017 = ds_1619.loc[dict(riv='skagit')].loc[dict(time='2017')]['transport'].values
skagit_2018 = ds_1619.loc[dict(riv='skagit')].loc[dict(time='2018')]['transport'].values
skagit_2019 = ds_1619.loc[dict(riv='skagit')].loc[dict(time='2019')]['transport'].values

# create arbitrary time vector
time = pd.date_range(start ='1/1/2014', end ='12/31/2014', freq ='D')

# plot (just a few years)
plt.plot(time,skagit_2014,linewidth=1.5, label='2014', color='deepskyblue')
# plt.plot(skagit_2015,linewidth=2, label='2015')
# plt.plot(skagit_2016,linewidth=2, label='2016')
plt.plot(time,skagit_2017,linewidth=1.5, label='2017', color='navy')
# plt.plot(skagit_2018,linewidth=2, label='2018')
plt.plot(time,skagit_2019,linewidth=1.5, label='2019', color='deeppink')

# format
plt.legend(loc='best',frameon=False)
date_form = mdates.DateFormatter("%b")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlim([time[0],time[-1]])
ax.set_ylim([0,2500])
ax.tick_params(axis='x', labelrotation=30)

plt.tight_layout()