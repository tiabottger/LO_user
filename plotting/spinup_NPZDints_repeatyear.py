"""
Compare average bottom DO between multiple years
(Set up to run for 6 years)

Tia modified to compare model reruns of 2013 

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
pth = Path(__file__).absolute().parent.parent.parent / 'LO' / 'pgrid'
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
years =  ['2013'] #,'2014','2015','2016','2017','2018','2019','2020']
# years =  ['2014','2015','2016','2017','2018','2019','2020']

# which  model run to look at?
gtagexes = ['cas7_t1_x11ab','cas7_t1y13v2_x11ab'] 

# where to put output figures
out_dir = Ldir['LOo'] / 'intermodel_comparison'
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
basin_mask_ds = xr.open_dataset('../../LO_output/hypvol_for_intrmdl_cmprsn/basin_masks_from_pugetsoundDObox.nc')
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
        ds = xr.open_dataset(Ldir['LOo'] / 'intermodel_comparison'/(gtagex + '_pugetsoundDO_' + year + '_NPZD_vert_ints.nc'))
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

# Variables to plot
plot_variables = [
    {
        'name': 'Nitrate',
        'short_name': 'NO3',
        'data': NO3_vol_norm,
        'ylim': (0, 40),
        'ylabel': r'NO$_3$ (mmol m$^{-3}$)'
    },
    {
        'name': 'Dissolved Oxygen',
        'short_name': 'DO',
        'data': DO_vol_norm,
        'ylim': (0, 300),
        'ylabel': r'DO (mmol m$^{-3}$)'
    }
]
##############################################################
##                    Plotting config                       ##
##############################################################

plot_year = '2013'
synthetic_start_year = 2013

# Puget Sound bounds
xmin = -123.29
xmax = -122.10
ymin = 46.95
ymax = 48.50

for variable_info in plot_variables:

    var_name = variable_info['name']
    var_short_name = variable_info['short_name']
    var_vol_norm = variable_info['data']

    print(f'Plotting {var_name}')

    # Create exactly one figure for this variable
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        gridspec_kw={'width_ratios': [1, 2]}
    )

    fig.suptitle(var_name, fontsize=16)

    ##########################################################
    ##                    Plot basin map                     ##
    ##########################################################

    ax0.pcolormesh(
        plon,
        plat,
        np.where(mask_rho == 0, np.nan, mask_rho),
        vmin=0,
        vmax=1.1,
        cmap='bone'
    )

    ax0.pcolormesh(
        plon,
        plat,
        np.where(mask_hc == 0, np.nan, mask_hc),
        vmin=0,
        vmax=2.5,
        cmap='RdPu'
    )

    ax0.pcolormesh(
        plon,
        plat,
        np.where(mask_ss == 0, np.nan, mask_ss),
        vmin=0,
        vmax=2,
        cmap='Purples'
    )

    ax0.pcolormesh(
        plon,
        plat,
        np.where(mask_wb == 0, np.nan, mask_wb),
        vmin=0,
        vmax=3,
        cmap='cool'
    )

    ax0.pcolormesh(
        plon,
        plat,
        np.where(mask_mb == 0, np.nan, mask_mb),
        vmin=0,
        vmax=1.5,
        cmap='summer'
    )

    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    ax0.set_ylabel('Latitude', fontsize=12)
    ax0.set_xlabel('Longitude', fontsize=12)
    ax0.tick_params(axis='both', labelsize=12)

    ax0.set_title(
        '(a) Basins',
        loc='left',
        fontsize=14,
        fontweight='bold'
    )

    pfun.dar(ax0)

    ##########################################################
    ##       Consecutive repeated runs for each gtagex       ##
    ##########################################################

    run_date_ranges = []

    for k, region in enumerate(regions):

        for g, gtagex in enumerate(gtagexes):

            synthetic_year = synthetic_start_year + g

            data_key = gtagex + region + plot_year

            if data_key not in var_vol_norm:
                print(
                    f'Missing {var_name} data for '
                    f'gtagex={gtagex}, region={region}, year={plot_year}'
                )
                continue

            avg_concentration = np.asarray(
                var_vol_norm[data_key]
            )

            synthetic_dates = pd.date_range(
                start=f'{synthetic_year}-01-01',
                periods=len(avg_concentration),
                freq='1D'
            )

            avg_concentration_filtered = zfun.lowpass(
                avg_concentration,
                n=nwin
            )

            line_label = region if g == 0 else '_nolegend_'

            if region == 'All Puget Sound':
                ax1.plot(
                    synthetic_dates,
                    avg_concentration_filtered,
                    linewidth=1,
                    color=colors[k],
                    alpha=1,
                    linestyle='--',
                    label=line_label
                )
            else:
                ax1.plot(
                    synthetic_dates,
                    avg_concentration_filtered,
                    linewidth=2,
                    color=colors[k],
                    alpha=0.8,
                    label=line_label
                )

            # Save each gtagex range only once
            if k == 0:
                run_date_ranges.append(
                    (
                        synthetic_dates[0],
                        synthetic_dates[-1],
                        gtagex
                    )
                )

    ##########################################################
    ##                  Format timeseries                    ##
    ##########################################################

    if len(run_date_ranges) == 0:
        print(f'No data were plotted for {var_name}')
        plt.close(fig)
        continue

    ax1.grid(
        visible=True,
        axis='both',
        color='silver',
        linestyle='--'
    )

    ax1.tick_params(
        axis='both',
        labelsize=12,
        rotation=30
    )

    ax1.set_ylabel(variable_info['ylabel'], fontsize=12)
    ax1.set_ylim(variable_info['ylim'])

    plot_start = run_date_ranges[0][0]
    plot_end = run_date_ranges[-1][1]

    ax1.set_xlim(plot_start, plot_end)

    ax1.axhline(
        y=0,
        color='silver',
        linestyle='--',
        linewidth=0.75
    )

    ##########################################################
    ##                Repeated year labels                   ##
    ##########################################################

    year_tick_locations = []

    for run_start, run_end, gtagex in run_date_ranges:
        midpoint = run_start + (run_end - run_start) / 2
        year_tick_locations.append(midpoint)

    ax1.set_xticks(year_tick_locations)
    ax1.set_xticklabels(['2013'] * len(run_date_ranges))

    ##########################################################
    ##             Separate the individual runs              ##
    ##########################################################

    for g in range(1, len(run_date_ranges)):
        ax1.axvline(
            run_date_ranges[g][0],
            color='0.4',
            linestyle=':',
            linewidth=1
        )

    ##########################################################
    ##                    gtagex labels                      ##
    ##########################################################

    for run_start, run_end, gtagex in run_date_ranges:

        midpoint = run_start + (run_end - run_start) / 2

        ax1.text(
            midpoint,
            0.94,
            gtagex,
            transform=ax1.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=9,
            bbox=dict(
                facecolor='white',
                alpha=0.7,
                edgecolor='none',
                pad=1
            )
        )

    ax1.set_xlabel('Repeated 2013 model runs', fontsize=12)

    ax1.set_title(
        f'(b) Avg. Conc. ({nwin}-day Hanning Window)',
        loc='left',
        fontsize=14,
        fontweight='bold'
    )

    ax1.legend(
        loc='lower right',
        fontsize=8
    )

    ##########################################################
    ##                    Save figure                        ##
    ##########################################################

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_filename = (
        f'2013_rerun_{var_short_name}_timeseries.png'
    )

    fig.savefig(
        out_dir / output_filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)