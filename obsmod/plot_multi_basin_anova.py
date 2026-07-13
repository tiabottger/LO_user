"""
This focuses on property-property plots and obs-mod plots.

It specializes on model-model-obs comparisons, then with a bunch of
choices for filtering the data based on source, season, and depth.

Hence it is primarily a tool for model development: is one version
different of better than another?

Here Tia added a function to highlight where different basins lay within
the property-property plots using bounding boxes.
A one-way ANOVA is performed to examine whether model error
(error= model - obs) are explained by within vs between basin variance.
"""
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial import cKDTree
from scipy.stats import f_oneway
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun, zrfun
Ldir = Lfun.Lstart()


testing = False

year = '2014'
in_dir = Ldir['parent'] / 'LO_output' / 'obsmod'

plt.close('all')

# load the basin masks
mask_ds = xr.open_dataset('basin_masks_from_pugetsoundDObox.nc')

# grid coordinates
lon_rho = mask_ds['lon_rho'].values
lat_rho = mask_ds['lat_rho'].values

# build nearest neighbor search tree
xy_grid = np.column_stack((lon_rho.ravel(), lat_rho.ravel()))
tree = cKDTree(xy_grid)

basin_var = {
    'hc': 'mask_hoodcanal',
    'ss': 'mask_southsound',
    'mb': 'mask_mainbasin',
    'wb': 'mask_whidbeybasin',
}

selected_basin = 'hc'  # 'hc','ss','mb','wb','all'

basin_name = {
    'hc': 'Hood Canal',
    'ss': 'South Sound',
    'mb': 'Main Basin',
    'wb': 'Whidbey Basin',
    'all': 'All Basins'
}[selected_basin]

# specify input (created by process_multi_bottle.py and process_multi_ctd.py)
for otype in ['bottle']:#, 'ctd']:
    in_fn = in_dir / ('combined_' + otype + '_' + year + '_cas7_t1_x11ab_ssc.pkl')
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
                ff_str = otype + '_' + year + '_' + selected_basin  # a string for the output .png file name

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

                # depth range
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
        
                # time range
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

                gtx_list = ['cas7_t1_x11ab', 'ssc']
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
                
                lon = df_dict['obs']['lon'].to_numpy()
                lat = df_dict['obs']['lat'].to_numpy()
                
                if selected_basin == 'all':
                    basin_mask = np.ones_like(lon, dtype=bool)
                else:
                    # find nearest model grid point for each observation
                    obs_xy = np.column_stack((lon, lat))
                    _, idx = tree.query(obs_xy)

                    jj, ii = np.unravel_index(idx, lon_rho.shape)

                    # extract basin mask at those grid points
                    basin_grid = mask_ds[basin_var[selected_basin]].values

                    # True where observation falls inside basin
                    basin_mask = basin_grid[jj, ii] == 1

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
                            x = x_raw
                            y = y_raw
                        
                        # background (all obs)
                        ax.plot(x, y, '.', color=c_dict[gtx], alpha=0.03)

                        # highlighted basin
                        ax.plot(x[basin_mask], y[basin_mask],
                                '.', color=c_dict[gtx], alpha=0.95)
        
                        # calculate bias and rmse for only highlighted basin
                        x_basin = x[basin_mask]
                        y_basin = y[basin_mask]
                        
                        # remove NaN values
                        valid = np.isfinite(x_basin) & np.isfinite(y_basin)
                        
                        if np.any(valid):
                            diff = y_basin[valid] - x_basin[valid]
                            bias = np.mean(diff)
                            rmse = np.sqrt(np.mean(diff**2))
                            
                        # ==========================================================
                        # ANOVA: Obs vs current model across all basin masks
                        # ==========================================================

                        # full-domain dataframe (not selected basin only)
                        df_anova = pd.DataFrame({
                            'lon': lon,
                            'lat': lat,
                            'obs': x,
                            'model': y
                        })

                        # remove NaNs
                        df_anova = df_anova[
                            np.isfinite(df_anova['obs']) &
                            np.isfinite(df_anova['model'])
                        ].copy()

                        # model error
                        df_anova['error'] = df_anova['model'] - df_anova['obs']

                        # map valid obs points to model grid
                        obs_xy_valid = np.column_stack((
                            df_anova['lon'].to_numpy(),
                            df_anova['lat'].to_numpy()
                        ))

                        _, idx_valid = tree.query(obs_xy_valid)
                        jj_valid, ii_valid = np.unravel_index(idx_valid, lon_rho.shape)

                        # assign basin labels
                        df_anova['basin'] = 'other'

                        for basin, varname in basin_var.items():
                            basin_grid = mask_ds[varname].values
                            inside = basin_grid[jj_valid, ii_valid] == 1
                            df_anova.loc[inside, 'basin'] = basin

                        # collect basin groups
                        groups = []
                        valid_basins = []

                        for basin in basin_var.keys():
                            group = df_anova.loc[
                                df_anova['basin'] == basin,
                                'error'
                            ].dropna()

                            if len(group) > 1:
                                groups.append(group)
                                valid_basins.append(basin)

                        # run ANOVA
                        if len(groups) > 1:

                            F, p = f_oneway(*groups)

                            overall_mean = df_anova['error'].mean()

                            ss_between = 0
                            ss_within = 0

                            for basin in valid_basins:
                                group = df_anova.loc[
                                    df_anova['basin'] == basin,
                                    'error'
                                ].dropna()

                                n = len(group)
                                mean_b = group.mean()

                                ss_between += n * (mean_b - overall_mean)**2
                                ss_within += ((group - mean_b)**2).sum()

                            eta2 = ss_between / (ss_between + ss_within)

                            print(
                                f'{gtx} | {vn}: '
                                f'F={F:.2f}, '
                                f'p={p:.3e}, '
                                f'eta²={eta2:.3f}'
                            )
                        
                            ax.text(.95,t_dict[gtx],'bias=%0.1f, rmse=%0.1f' % (bias,rmse),c=c_dict[gtx],
                                transform=ax.transAxes, ha='right', fontweight='bold', bbox=pfun.bbox,
                                fontsize=fs-1,style='italic')

                    if otype == 'bottle':
                        if jj in [9,10,11]:
                            ax.set_xlabel('Observed')
                        if jj in [1,5,9]:
                            ax.set_ylabel('Modeled')
                    elif otype == 'ctd':
                        if jj in [4,5]:
                            ax.set_xlabel('Observed')
                        if jj in [1,4]:
                            ax.set_ylabel('Modeled')
        
                    # add labels to identify the model runs with the colors
                    if jj == 1:
                        yy = 0
                        for gtx in c_dict.keys():
                            ax.text(.05, .7 + 0.1*yy, gtx, c=c_dict[gtx], transform=ax.transAxes,
                                fontweight='bold', ha='left')
                            yy += 1
            
                    ax.text(.05,.9,f"{vn} — {basin_name}",transform=ax.transAxes, fontweight='bold')
                    ax.axis([lim_dict[vn][0], lim_dict[vn][1], lim_dict[vn][0], lim_dict[vn][1]])
                    ax.plot([lim_dict[vn][0], lim_dict[vn][1]], [lim_dict[vn][0], lim_dict[vn][1]],'-g')
                    ax.grid(True)
    
                # station map
                if otype == 'bottle':
                    ax = fig.add_subplot(1,4,4)
                elif otype == 'ctd':
                    ax = fig.add_subplot(1,3,3)
                    
                pfun.add_coast(ax)
                
                # all stations (faded)
                ax.plot(
                    lon,
                    lat,
                    '.',
                    color='grey',
                    alpha=0.95,
                    markersize=7, zorder=2
                )

                # highlighted basin stations
                ax.plot(
                    lon[basin_mask],
                    lat[basin_mask],
                    '.',
                    color='k',
                    alpha=0.95,
                    markersize=7, zorder=3
                )

                #ax.axis([-130,-122,42,52])
                ax.axis([-123.2, -122.25, 47.0, 48.35])
                
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
                    plt.savefig(out_dir / (ff_str + '.png'))
                    plt.close('all')

    
