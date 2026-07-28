"""
This focuses on property-property plots and obs-mod plots.

It specializes on model-model-obs comparisons, then with a bunch of
choices for filtering the data based on source, season, and depth.

Hence it is primarily a tool for model development: is one version
different of better than another?

Tia edited this version for model intercomparison between SalishSeaCast, 
Salish Sea Model, and LiveOcean
- filtered to include only data within Puget Sound mask, with ability to select sub-basins
- converted DO to mg/L 
- added lines of best fit 
"""
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial import cKDTree
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun, zrfun
Ldir = Lfun.Lstart()

testing = False

year = '2014'
in_dir = Ldir['parent'] / 'LO_output' / 'obsmod'

# --- Puget Sound basin mask settings---
mask_ds = xr.open_dataset('basin_masks_from_pugetsoundDObox.nc')

# Model-grid coordinates used by the basin masks
lon_rho = mask_ds['lon_rho'].values
lat_rho = mask_ds['lat_rho'].values

# Build nearest-neighbor search tree
xy_grid = np.column_stack((
    lon_rho.ravel(),
    lat_rho.ravel()
))
tree = cKDTree(xy_grid)

# Short basin names mapped to variables in the mask file
basin_var = {
    'ps': 'mask_pugetsound', # the Puget Sound domain
    'hc': 'mask_hoodcanal',
    'ss': 'mask_southsound',
    'mb': 'mask_mainbasin',
    'wb': 'mask_whidbeybasin',
}

selected_basin = 'ps'

basin_name = {
    'ps': 'Puget Sound',
    'hc': 'Hood Canal',
    'ss': 'South Sound',
    'mb': 'Main Basin',
    'wb': 'Whidbey Basin',
    'all': 'All Locations',
}[selected_basin]

plt.close('all')

# specify input (created by process_multi_bottle.py and process_multi_ctd.py)
for otype in ['bottle']:#, 'ctd']:
    in_fn = in_dir / ('combined_' + otype + '_' + year + '_cas7_t1_x11ab_ssc_ssm.pkl')
    df0_dict = pickle.load(open(in_fn, 'rb'))
    
    # remove non-DataFrame entries (like meta)
    df0_dict = {
        k: v for k, v in df0_dict.items()
        if isinstance(v, pd.DataFrame)
    }

    # where to put output figures
    out_dir = Ldir['parent'] / 'LO_output' / 'obsmod_plots'
    Lfun.make_dir(out_dir)

    if otype == 'bottle':
        # add DIN field
        for gtx in df0_dict.keys():
            if gtx == 'cas6_v0_live':
                df0_dict[gtx]['DIN'] = df0_dict[gtx]['NO3']
            else:
                df0_dict[gtx]['DIN'] = df0_dict[gtx]['NO3'] + df0_dict[gtx]['NH4']

    # loop over a variety of choices

    if otype == 'bottle':
        if True:
            source_list = ['all']
        else:
            source_list = ['nceiCoastal', 'nceiSalish', 'dfo1', 'ecology']
            #source_list = ['nceiSalish']
        
    elif otype == 'ctd':
        if True:
            source_list = ['all']
        else:
            source_list = ['dfo1', 'ecology']
            
    if True:
        time_range_list = ['all']
    else:
        time_range_list = ['spring','summer']
        
    if True:
        depth_range_list = ['all']
    else:
        depth_range_list = ['shallow','deep']
        
        
    for source in source_list:
        for depth_range in depth_range_list:
            for time_range in time_range_list:
            
                df_dict = df0_dict.copy()

                # ===== FILTERS ======================================================
                f_str = otype + ' ' + year + '\n\n' # a string to put for info on the map
                ff_str = otype + '_' + year + '_' + selected_basin # a string for the output .png file name

                # --- Basin filter ---
                lon = df_dict['obs']['lon'].to_numpy()
                lat = df_dict['obs']['lat'].to_numpy()

                if selected_basin == 'all':
                    basin_mask = np.ones(len(df_dict['obs']), dtype=bool)
                else:
                    obs_xy = np.column_stack((lon, lat))
                    _, idx = tree.query(obs_xy)

                    eta_idx, xi_idx = np.unravel_index(
                        idx,
                        lon_rho.shape
                    )

                    basin_grid = mask_ds[
                        basin_var[selected_basin]
                    ].values

                    basin_mask = basin_grid[eta_idx, xi_idx] == 1

                for gtx in df_dict.keys():

                    if len(df_dict[gtx]) != len(basin_mask):
                        raise ValueError(
                            f'Length mismatch before basin filtering: '
                            f'obs={len(basin_mask)}, '
                            f'{gtx}={len(df_dict[gtx])}'
                        )

                    df_dict[gtx] = (
                        df_dict[gtx]
                        .iloc[basin_mask]
                        .copy()
                        .reset_index(drop=True)
                    )


                # --- Source filter ---            
                # limit which sources to use
                if source == 'all':
                    # use df_dict as-is
                    f_str += 'Source = all\n'
                    ff_str += '_all'
                else:
                    # use just one source
                    f_str += 'Source = ' + source + '\n'
                    ff_str += '_' + source
                    for gtx in df_dict.keys():
                        df_dict[gtx] = df_dict[gtx].loc[df_dict[gtx].source==source,:]

                
                # --- Depth range filter ---
                if depth_range == 'all':
                    pass
                elif depth_range == 'shallow':
                    # shallow water
                    zz = -30
                    f_str += 'Z above ' + str(zz) + ' [m]\n'
                    ff_str += '_shallow'
                    for gtx in df_dict.keys():
                        df_dict[gtx] = df_dict[gtx].loc[df_dict[gtx].z >= zz,:]
                elif depth_range == 'deep':
                    # deep water
                    zz = -30
                    f_str += 'Z below ' + str(zz) + ' [m]\n'
                    ff_str += '_deep'
                    for gtx in df_dict.keys():
                        df_dict[gtx] = df_dict[gtx].loc[df_dict[gtx].z <= zz,:]
        
                # --- Time range filter ---
                if time_range == 'all':
                    pass
                elif time_range == 'spring':
                    # specific months
                    f_str += 'Months = [4,5,6]\n'
                    ff_str += '_spring'
                    for gtx in df_dict.keys():
                        dti = pd.DatetimeIndex(df_dict[gtx].time)
                        mask = (dti.month==4) | (dti.month==5) | (dti.month==6)
                        df_dict[gtx] = df_dict[gtx].loc[mask,:]
                elif time_range == 'summer':
                    # specific months
                    f_str += 'Months = [7,8,9]\n'
                    ff_str += '_summer'
                    for gtx in df_dict.keys():
                        dti = pd.DatetimeIndex(df_dict[gtx].time)
                        mask = (dti.month==7) | (dti.month==8) | (dti.month==9)
                        df_dict[gtx] = df_dict[gtx].loc[mask,:]
                # ====================================================================

                # Plotting

                fs = 12
                pfun.start_plot(figsize=(20,12), fs=fs)

                gtx_list = ['cas7_t1_x11ab', 'ssc', 'ssm']
                #gtx_list = ['cas7_t1_x11ab', 'ssc']
                c_dict = dict(zip(gtx_list,['r','b','g']))
                t_dict = dict(zip(gtx_list,[.05,.15,0.25])) # vertical position of stats text

                alpha = 0.3
                fig = plt.figure()

                if otype == 'bottle':
                    vn_list = ['SA','CT','DO','NO3','NH4','DIN',
                        'DIC', 'TA', 'Chl']
                    jj_list = [1,2,3,5,6,7,9,10,11] # indices for the data plots
                elif otype == 'ctd':
                    vn_list = ['SA','CT','DO','Chl']
                    jj_list = [1,2,4,5] # indices for the data plots

                lim_dict = {'SA':(14,36),'CT':(0,20),'DO':(0,20),
                    'NO3':(0,50),'NH4':(0,10),'DIN':(0,50),
                    'DIC':(1500,2500),'TA':(1500,2500),'Chl':(0,20)}

                # convert ssc diatoms + flagellates to chlorophyll 
                df_dict['ssc']['Chl'] = (df_dict['ssc']['DIAT'] + df_dict['ssc']['FLAG']) * 2

                for ii in range(len(vn_list)):
                    jj = jj_list[ii]
                    if otype == 'bottle':
                        ax = fig.add_subplot(3,4,jj)
                    elif otype == 'ctd':
                        ax = fig.add_subplot(2,3,jj)
                    vn = vn_list[ii]
                    x_raw = df_dict['obs'][vn].to_numpy()
                    for gtx in gtx_list:
                        
                        # skip variable if missing in model
                        if vn not in df_dict[gtx].columns:
                            print(f"Skipping {vn} for {gtx} (missing)")
                            continue
                        
                        y_raw = df_dict[gtx][vn].to_numpy()
                        
                        # Convert DO from µM to mg/L
                        if vn == 'DO':
                            x = x_raw * 0.032
                            y = y_raw * 0.032
                        else:
                            x = x_raw.copy()
                            y = y_raw.copy()
                            
                        # Keep only paired finite observation-model values
                        valid = np.isfinite(x) & np.isfinite(y)
                        x_valid = x[valid]
                        y_valid = y[valid]
                        
                        ax.plot(x_valid,y_valid,marker='.',ls='',color=c_dict[gtx], alpha=alpha)
        
                        # Calculate bias and RMSE
                        bias = np.nanmean(y_valid-x_valid)
                        rmse = np.sqrt(np.nanmean((y_valid-x_valid)**2))
                        
                        # Add a linear trend line
                        if len(x_valid) >= 2 and np.unique(x_valid).size >= 2: 
                            slope, intercept = np.polyfit( x_valid, y_valid, deg=1 )
                        
                            # Draw the trend line over the displayed variable range 
                            x_trend = np.linspace( lim_dict[vn][0], lim_dict[vn][1], 200 ) 
                            y_trend = slope * x_trend + intercept
                            
                            ax.plot(
                                x_trend, y_trend, 
                                color=c_dict[gtx], linewidth=1, linestyle='-')
                        
                        ax.text(.95,t_dict[gtx],'bias=%0.1f, rmse=%0.1f' % (bias,rmse),c=c_dict[gtx],
                                transform=ax.transAxes, ha='right', fontweight='bold', bbox=pfun.bbox,
                                fontsize=15,style='italic')

                    if otype == 'bottle':
                        if jj in [9,10,11]:
                            ax.set_xlabel('Observed', fontsize=16)
                        if jj in [1,5,9]:
                            ax.set_ylabel('Modeled', fontsize=16)
                    elif otype == 'ctd':
                        if jj in [4,5]:
                            ax.set_xlabel('Observed')
                        if jj in [1,4]:
                            ax.set_ylabel('Modeled')
                            
                
                           
                    # add labels to identify the model runs with the colors
                    # if jj == 1:
                    #     yy = 0
                    #     for gtx in c_dict.keys():
                    #         ax.text(.05, .7 + 0.1*yy, gtx, c=c_dict[gtx], transform=ax.transAxes,
                    #             fontweight='bold', ha='left')
                    #         yy += 1
            
                    vn_label=['Salinity','Temperature','DO','NO3','NH4','DIN',
                        'DIC', 'TA', 'Chl']
                    vn_lab = vn_label[ii]
                    
                    # ax.text(.05,.9,vn,transform=ax.transAxes, fontweight='bold')
                    ax.text(.05,.9,vn_lab,transform=ax.transAxes, fontweight='bold', fontsize=16)
                    ax.axis([lim_dict[vn][0], lim_dict[vn][1], lim_dict[vn][0], lim_dict[vn][1]])
                    ax.plot([lim_dict[vn][0], lim_dict[vn][1]], [lim_dict[vn][0], lim_dict[vn][1]],'-k')
                    ax.grid(True)
                    ax.tick_params(axis='both', labelsize=14)
    
                # station map
                if otype == 'bottle':
                    ax = fig.add_subplot(1,4,4)
                elif otype == 'ctd':
                    ax = fig.add_subplot(1,3,3)
                df_dict['obs'].plot(x='lon',y='lat',style='.g',legend=False, ax=ax)
                pfun.add_coast(ax)
    
                # ax.axis([-130,-122,42,52]) # zoomed out
                ax.axis([-123.2, -122.25, 47.0, 48.35]) # Puget Sound
                
                pfun.dar(ax)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.text(.05,0,f_str,va='bottom',transform=ax.transAxes,fontweight='bold')

                fig.tight_layout()
                
                print('Plotting ' + ff_str)
                sys.stdout.flush()
                
                if testing:
                    plt.show()
                else:
                    #plt.savefig(out_dir / (ff_str + '.png'))
                    plt.savefig(out_dir / (ff_str + '_withssm_poster.png'))
                    plt.close('all')

    
