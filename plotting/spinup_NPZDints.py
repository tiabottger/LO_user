"""
Compare average bottom DO between multiple years
(Set up to run for 6 years)

"""

# import things
from subprocess import Popen as Po
from subprocess import PIPE as Pi
from matplotlib.markers import MarkerStyle
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from matplotlib.dates import MonthLocator
import matplotlib.patches as patches
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as image
import pandas as pd
import cmocean
import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as PathEffects
import pinfo

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

import sys
from pathlib import Path
pth = Path(__file__).absolute().parent.parent.parent.parent / 'LO' / 'pgrid'
if str(pth) not in sys.path:
    sys.path.append(str(pth))
import gfun

Gr = gfun.gstart()

Ldir = Lfun.Lstart()

##############################################################
##                       USER INPUTS                        ##
##############################################################

WWTP_loc = False

# Hanning window length
nwin = 20

# years =  ['2015']
years =  ['2013','2014','2015','2016','2017','2018','2019','2020']
# years =  ['2014','2015','2016','2017','2018','2019','2020']

# which  model run to look at?
gtagexes = ['cas7_t1_x11ab'] 

# where to put output figures
out_dir = Ldir['LOo'] / 'chapter_2' / 'figures'
Lfun.make_dir(out_dir)

regions = ['Hood Canal', 'South Sound', 'Whidbey Basin', 'Main Basin', 'All Puget Sound']
colors = ['hotpink','mediumpurple','dodgerblue','yellowgreen','black']

plt.close('all')

##############################################################
##                    HELPER FUNCTIONS                      ##
##############################################################

# helper function to convert Ecology name to LO name
def SSM2LO_name(rname):
    """
    Given a river name in LiveOcean, find corresponding river name in SSM
    """
    repeatrivs_fn = '../../../LO_data/trapsD00/LiveOcean_SSM_rivers.xlsx'
    repeatrivs_df = pd.read_excel(repeatrivs_fn)
    rname_LO = repeatrivs_df.loc[repeatrivs_df['SSM_rname'] == rname, 'LO_rname'].values[0]
    return rname_LO

def LO2SSM_name(rname):
    """
    Given a river name in LiveOcean, find corresponding river name in SSM
    """
    repeatrivs_fn = Ldir['data'] / 'trapsD00' / 'LiveOcean_SSM_rivers.xlsx'
    repeatrivs_df = pd.read_excel(repeatrivs_fn)
    rname_SSM = repeatrivs_df.loc[repeatrivs_df['LO_rname'] == rname, 'SSM_rname'].values[0]
    return rname_SSM


if WWTP_loc == True:
    # set up the time index for the record
    Ldir = Lfun.Lstart()
    dsf = Ldir['ds_fmt']
    dt0 = datetime.strptime('2020.01.01',dsf)
    dt1 = datetime.strptime('2020.12.31',dsf)
    days = (dt0, dt1)
        
    # pandas Index objects
    dt_ind = pd.date_range(start=dt0, end=dt1)
    yd_ind = pd.Index(dt_ind.dayofyear)

    # Get LiveOcean grid info --------------------------------------------------

    # get the grid data
    ds = xr.open_dataset('../../../LO_data/grids/cas7/grid.nc')
    z = -ds.h.values
    mask_rho = np.transpose(ds.mask_rho.values)
    lon = ds.lon_rho.values
    lat = ds.lat_rho.values
    X = lon[0,:] # grid cell X values
    Y = lat[:,0] # grid cell Y values
    plon, plat = pfun.get_plon_plat(lon,lat)
    # make a version of z with nans where masked
    zm = z.copy()
    zm[np.transpose(mask_rho) == 0] = np.nan
    zm[np.transpose(mask_rho) != 0] = -1

    # get flow, nitrate, and ammonium values
    fp_wwtps = '../../../LO_output/pre/trapsP01/moh20_wwtps/lo_base/Data_historical/'
    moh20_flowdf_wwtps = pd.read_pickle(fp_wwtps+'CLIM_flow.p')    # m3/s
    moh20_no3df_wwtps = pd.read_pickle(fp_wwtps+'CLIM_NO3.p')      # mmol/m3
    moh20_nh4df_wwtps = pd.read_pickle(fp_wwtps+'CLIM_NH4.p')      # mmol/m3

    fp_wwtps = '../../../LO_output/pre/trapsP01/was24_wwtps/lo_base/Data_historical/'
    was24_flowdf_wwtps = pd.read_pickle(fp_wwtps+'CLIM_flow.p')    # m3/s
    was24_no3df_wwtps = pd.read_pickle(fp_wwtps+'CLIM_NO3.p')      # mmol/m3
    was24_nh4df_wwtps = pd.read_pickle(fp_wwtps+'CLIM_NH4.p')      # mmol/m3

    # calculate total DIN concentration in mg/L
    moh20_dindf_wwtps = (moh20_no3df_wwtps + moh20_nh4df_wwtps)/71.4    # mg/L
    was24_dindf_wwtps = (was24_no3df_wwtps + was24_nh4df_wwtps)/71.4    # mg/L

    # calculate daily loading timeseries in kg/d
    moh20_dailyloaddf_wwtps = 86.4*moh20_dindf_wwtps*moh20_flowdf_wwtps # kg/d = 86.4 * mg/L * m3/s
    was24_dailyloaddf_wwtps = 86.4*was24_dindf_wwtps*was24_flowdf_wwtps # kg/d = 86.4 * mg/L * m3/s

    # calculate average daily load over the year (kg/d)
    moh20_avgload_wwtps = moh20_dailyloaddf_wwtps.mean(axis=0).to_frame(name='avg-daily-load(kg/d)')
    was24_avgload_wwtps = was24_dailyloaddf_wwtps.mean(axis=0).to_frame(name='avg-daily-load(kg/d)')

    # add row and col index for plotting on LiveOcean grid
    griddf0_wwtps = pd.read_csv('../../../LO_data/grids/cas7/moh20_wwtp_info.csv')
    griddf_wwtps = griddf0_wwtps.set_index('rname') # use point source name as index
    moh20_avgload_wwtps = moh20_avgload_wwtps.join(griddf_wwtps['row_py']) # add row to avg load df (uses rname to index)
    moh20_avgload_wwtps = moh20_avgload_wwtps.join(griddf_wwtps['col_py']) # do the same for cols

    griddf0_wwtps = pd.read_csv('../../../LO_data/grids/cas7/was24_wwtp_info.csv')
    griddf_wwtps = griddf0_wwtps.set_index('rname') # use point source name as index
    was24_avgload_wwtps = was24_avgload_wwtps.join(griddf_wwtps['row_py']) # add row to avg load df (uses rname to index)
    was24_avgload_wwtps = was24_avgload_wwtps.join(griddf_wwtps['col_py']) # do the same for cols

    # get point source lat and lon
    moh20_lon_wwtps = [X[int(col)] for col in moh20_avgload_wwtps['col_py']]
    moh20_lat_wwtps = [Y[int(row)] for row in moh20_avgload_wwtps['row_py']]
    was24_lon_wwtps = [X[int(col)] for col in was24_avgload_wwtps['col_py']]
    was24_lat_wwtps = [Y[int(row)] for row in was24_avgload_wwtps['row_py']]
    
    # define marker sizes (minimum size is 10 so dots don't get too small)
    moh20_sizes_wwtps = [max(0.05*load,5) for load in moh20_avgload_wwtps['avg-daily-load(kg/d)']]
    was24_sizes_wwtps = [max(0.05*load,5) for load in was24_avgload_wwtps['avg-daily-load(kg/d)']]


##############################################################
##                      PROCESS DATA                        ##
##############################################################

# read in masks
basin_mask_ds = xr.open_dataset('../../../LO_output/chapter_2/data/basin_masks_from_pugetsoundDObox.nc')
mask_rho = basin_mask_ds.mask_rho.values
mask_hc = basin_mask_ds.mask_hoodcanal.values
mask_ss = basin_mask_ds.mask_southsound.values
mask_wb = basin_mask_ds.mask_whidbeybasin.values
mask_mb = basin_mask_ds.mask_mainbasin.values
mask_ps = basin_mask_ds.mask_pugetsound.values
lon = basin_mask_ds['lon_rho'].values
lat = basin_mask_ds['lat_rho'].values
h = basin_mask_ds['h'].values
plon, plat = pfun.get_plon_plat(lon,lat)

##############################################################
# get average concentration per basin

# initialize empty dictionaries and fill with vertical integrals
NO3_vert_dict = {}
DO_vert_dict = {}

for year in years:
    for gtagex in gtagexes:
        ds = xr.open_dataset(Ldir['LOo'] / 'chapter_2' / 'data' / (gtagex + '_pugetsoundDO_' + year + '_NPZD_vert_ints.nc'))
        NO3_vert_int = ds['NO3_vert_int'].values
        phyto_vert_int = ds['phyto_vert_int'].values
        zoop_vert_int = ds['zoop_vert_int'].values
        NH4_vert_int = ds['NH4_vert_int'].values
        LdetritusN_vert_int = ds['LdetritusN_vert_int'].values
        SdetritusN_vert_int = ds['SdetritusN_vert_int'].values
        DO_vert_int = ds['DO_vert_int'].values
        # add data to dictionaries
        NO3_vert_dict[gtagex+year] = NO3_vert_int
        DO_vert_dict[gtagex+year] = DO_vert_int

# grid cell areas
fp = Ldir['LOo'] / 'extract' / 'cas7_t1_x11ab' / 'box' / ('pugetsoundDO_2014.01.01_2014.12.31.nc')
box_ds = xr.open_dataset(fp)
DX = (box_ds.pm.values)**-1
DY = (box_ds.pn.values)**-1
DA = DX*DY # get area in m2


# initialize dictionary for average concentration (volume integrals [mol], normalized by volume)
NO3_vol_norm = {}
DO_vol_norm = {}

for year in years:
    for region in regions:

        # get mask for the region
        if region == 'Hood Canal':
            mask = mask_hc
        elif region == 'South Sound':
            mask = mask_ss
        elif region == 'Whidbey Basin':
            mask = mask_wb
        elif region == 'Main Basin':
            mask = mask_mb
        elif region == 'All Puget Sound':
            mask = mask_ps

        # basin volume
        h_masked = h * mask
        basin_vol = np.sum(h_masked * DA) # [m3]

        for gtagex in gtagexes:

            NO3_vert_int = NO3_vert_dict[gtagex+year]
            NO3_vert_int_masked = NO3_vert_int * mask
            NO3_vol_timeseries = np.sum(NO3_vert_int_masked * DA, axis=(1, 2)) # [mol]
            NO3_vol_norm[gtagex+region+year] = NO3_vol_timeseries / basin_vol *1000 # [mmol/m3]

            DO_vert_int = DO_vert_dict[gtagex+year]
            DO_vert_int_masked = DO_vert_int * mask
            DO_vol_timeseries = np.sum(DO_vert_int_masked * DA, axis=(1, 2)) # [mol]
            DO_vol_norm[gtagex+region+year] = DO_vol_timeseries / basin_vol *1000 # [mmol/m3]

##############################################################
##                    Plot basin map                        ##
##############################################################

# Puget Sound bounds
xmin = -123.29
xmax = -122.1
ymin = 46.95
ymax = 48.50#48.93

# variables
vars = ['NO3','Dissolved Oxygen']
for j,var_vol_norm in enumerate([NO3_vol_norm,DO_vol_norm]):

    # initialize figure
    fig, (ax0, ax1) = plt.subplots(1,2,figsize = (11,5),gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle(vars[j], fontsize = 16)

    # All Puget Sound
    ax0.pcolormesh(plon, plat, np.where(mask_rho == 0, np.nan, mask_rho),
                vmin=0, vmax=1.1, cmap='bone' )
    # Hood Canal
    ax0.pcolormesh(plon, plat, np.where(mask_hc == 0, np.nan, mask_hc),
                vmin=0, vmax=2.5, cmap='RdPu' )
    # South Sound
    ax0.pcolormesh(plon, plat, np.where(mask_ss == 0, np.nan, mask_ss),
                vmin=0, vmax=2, cmap='Purples' )
    # Whidbey Basin
    ax0.pcolormesh(plon, plat, np.where(mask_wb == 0, np.nan, mask_wb),
                vmin=0, vmax=3, cmap='cool' )
    # Main Basin
    ax0.pcolormesh(plon, plat, np.where(mask_mb == 0, np.nan, mask_mb),
                vmin=0, vmax=1.5, cmap='summer' )


    # format figure
    ax0.set_xlim([xmin,xmax])
    ax0.set_ylim([ymin,ymax])
    ax0.set_ylabel('Latitude', fontsize=12)
    ax0.set_xlabel('Longitude', fontsize=12)
    ax0.tick_params(axis='both', labelsize=12)
    ax0.set_title('(a) Basins', loc='left', fontsize=14, fontweight='bold')
    pfun.dar(ax0)

    # add wwtp locations
    if WWTP_loc == True:
        edgecolor = 'black'
        facecolor = 'none'
        alpha = 1
        ax0.scatter(moh20_lon_wwtps,moh20_lat_wwtps,color=facecolor, edgecolors=edgecolor, alpha=alpha,
                     linewidth=1, s=moh20_sizes_wwtps, label='WWTPs')
        ax0.scatter(was24_lon_wwtps,was24_lat_wwtps,color=facecolor, edgecolors=edgecolor, alpha=alpha,
                     linewidth=1, s=was24_sizes_wwtps)
        leg_szs = [100, 1000, 10000]
        szs = [0.05*(leg_sz) for leg_sz in leg_szs]
        l0 = plt.scatter([],[], s=szs[0], color=facecolor, alpha=alpha, edgecolors=edgecolor, linewidth=1)
        l1 = plt.scatter([],[], s=szs[1], color=facecolor, alpha=alpha, edgecolors=edgecolor, linewidth=1)
        l2 = plt.scatter([],[], s=szs[2], color=facecolor, alpha=alpha, edgecolors=edgecolor, linewidth=1)
        labels = ['< 100', '1,000', '10,000']
        legend = ax0.legend([l0, l1, l2], labels, fontsize = 10, markerfirst=False,
            title='WWTP loading \n'+r' (kg N d$^{-1}$)',loc='upper left', labelspacing=1, borderpad=0.8)
        plt.setp(legend.get_title(),fontsize=9)

##############################################################
##    Sub-basins and multiple years and percent change      ##
##############################################################

# # plot timeseries
#     for year in years:
#         # create time vector
#         startdate = year+'.01.01'
#         enddate = year+'.12.31'
#         dates = pd.date_range(start= startdate, end= enddate, freq= '1d')
#         dates_local = [pfun.get_dt_local(x) for x in dates]
#         # loop through model runs
#         for k,region in enumerate(regions):
#             # get data for the basin and gtagex and year
#             gtagex = 'cas7_t1_x11ab'
#             avg_concentration = var_vol_norm[gtagex+region+year]
#             # pass through hanning window
#             avg_concentration_filtered = zfun.lowpass(avg_concentration,n=nwin)
#             # plot data
#             if region == 'All Puget Sound':
#                 ax1.plot(dates_local,avg_concentration_filtered,linewidth=2,
#                         color=colors[k], alpha=1,linestyle='--')
#             else:
#                 ax1.plot(dates_local,avg_concentration_filtered,linewidth=3,
#                     color=colors[k], alpha=0.8)

    # plot timeseries
    gtagex = 'cas7_t1_x11ab'

    for k, region in enumerate(regions):
        all_dates_local = []
        all_avg_concentration = []

        # concatenate all years first
        for year in years:
            startdate = year + '.01.01'
            enddate = year + '.12.31'
            dates = pd.date_range(start=startdate, end=enddate, freq='1D')
            dates_local = [pfun.get_dt_local(x) for x in dates]

            avg_concentration = var_vol_norm[gtagex + region + year]

            all_dates_local.extend(dates_local)
            all_avg_concentration.extend(avg_concentration)

        # convert to array
        all_avg_concentration = np.array(all_avg_concentration)

        # apply lowpass to the full concatenated series
        all_avg_concentration_filtered = zfun.lowpass(all_avg_concentration, n=nwin)

        # plot the full filtered series
        if region == 'All Puget Sound':
            ax1.plot(all_dates_local, all_avg_concentration_filtered,
                    linewidth=1, color=colors[k], alpha=1, linestyle='--')
        else:
            ax1.plot(all_dates_local, all_avg_concentration_filtered,
                    linewidth=2, color=colors[k], alpha=0.8)
                
    # format figure
    ax1.grid(visible=True, axis='both', color='silver', linestyle='--')
    ax1.tick_params(axis='both', labelsize=12, rotation=30)
    ax1.set_ylabel(r'mmol/m3', fontsize=12)
    # create time vector
    startdate = years[0]+'.01.01'
    enddate = years[-1]+'.12.31'
    dates = pd.date_range(start= startdate, end= enddate, freq= '1d')
    dates_local = [pfun.get_dt_local(x) for x in dates]
    ax1.set_xlim([dates_local[0],dates_local[-1]])
    ax1.hlines(y=0, xmin=dates_local[0], xmax=dates_local[-1],
               color='silver', linestyle='--', linewidth=0.75)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.set_title('(b) Avg. Conc. ({}-day Hanning Window)'.format(nwin), loc='left', fontsize=14, fontweight='bold')
    

    # set y-lims
    if vars[j] == 'NO3':
        ymax_conc = 40
    elif vars[j] == 'NH4':
        ymax_conc = 1.75
    elif vars[j] == 'Phytoplankton':
        ymax_conc = 1.75
    elif vars[j] == 'Zooplankton':
        ymax_conc = 0.175
    elif vars[j] == 'Large Detritus':
        ymax_conc = 0.04
    elif vars[j] == 'Small Detritus':
        ymax_conc = 1.2
    elif vars[j] == 'Dissolved Oxygen':
        ymax_conc = 300
    ax1.set_ylim([0,ymax_conc])

    plt.tight_layout()
    plt.show()