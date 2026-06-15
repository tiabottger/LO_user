"""
This focuses on property-property plots and obs-mod plots.

It specializes on model-model-obs comparisons, then with a bunch of
choices for filtering the data based on source, season, and depth.

Hence it is primarily a tool for model development: is one version
different of better than another?
"""
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun, zrfun
Ldir = Lfun.Lstart()

testing = False

years = ['2014', '2015', '2016', '2017']

in_dir = Ldir['parent'] / 'LO_output' / 'obsmod'

plt.close('all')

# specify input (created by process_multi_bottle.py and process_multi_ctd.py)

df0_dict_all = {}

for year in years:

    in_fn = (
        in_dir /
        f'combined_bottle_{year}_cas7_t1_x11ab_ssc.pkl'
    )

    this_dict = pickle.load(open(in_fn, 'rb'))

    this_dict = {
        k:v for k,v in this_dict.items()
        if isinstance(v, pd.DataFrame)
    }

    for k, df in this_dict.items():

        if k not in df0_dict_all:
            df0_dict_all[k] = df.copy()

        else:
            df0_dict_all[k] = pd.concat(
                [df0_dict_all[k], df],
                ignore_index=True
            )

df0_dict = df0_dict_all
otype = 'bottle'

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
            f_str = otype + ' ' + '2014-2017' + '\n\n' # a string to put for info on the map
            ff_str = otype + '_' + '2014-2017' # a string for the output .png file name
            
            f_str += 'Highlighted year = 2014 \n'

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

            # gtx_list = ['cas7_t1_x11ab', 'ssc']
            # c_dict = dict(zip(gtx_list,['r','b','g']))
            # t_dict = dict(zip(gtx_list,[.05,.15,0.25])) # vertical position of stats text

            # gtx_list = ['cas7_t1_x11ab']
            # c_dict = {
            #     'cas7_t1_x11ab': 'r'
            # }
            # t_dict = {
            #     'cas7_t1_x11ab': 0.05
            # }
            
            gtx_list = ['ssc']
            c_dict = {
                'ssc': 'b'
            }
            t_dict = {
                'ssc': 0.05
            }

            alpha = 0.3
            fig = plt.figure()

            if otype == 'bottle':
                vn_list = ['SA','CT','DO','NO3','NH4','DIN',
                    'DIC', 'TA', 'Chl']
                jj_list = [1,2,3,5,6,7,9,10,11] # indices for the data plots
            elif otype == 'ctd':
                vn_list = ['SA','CT','DO','Chl']
                jj_list = [1,2,4,5] # indices for the data plots

            lim_dict = {'SA':(14,36),'CT':(0,20),'DO':(0,600),
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
                
                if vn not in df_dict[gtx].columns:
                    print(f"Skipping {vn} for {gtx} (missing)")
                    continue
                
                x = df_dict['obs'][vn].to_numpy()
                for gtx in gtx_list:
                    
                    # skip variable if missing in model
                    if vn not in df_dict[gtx].columns:
                        print(f"Skipping {vn} for {gtx} (missing)")
                        continue
                    
                    y = df_dict[gtx][vn].to_numpy()

                    year = pd.DatetimeIndex(
                        df_dict[gtx]['time']
                    ).year.to_numpy()

                    valid = (
                        np.isfinite(x)
                        & np.isfinite(y)
                    )

                    x_valid = x[valid]
                    y_valid = y[valid]
                    year_valid = year[valid]

                    mask_2014 = (year_valid == 2014)
                    mask_other = (year_valid != 2014)

                    ax.plot(x_valid[mask_other],y_valid[mask_other],marker='.',ls='',color='k', alpha=0.8)
                    
                    # plot 2014 data on top with higher alpha
                    ax.plot(x_valid[mask_2014],y_valid[mask_2014],marker='.',ls='',color=c_dict[gtx], alpha=0.8)
    
                    if (not np.isnan(x).all()) and (not np.isnan(y).all()) and (len(x) > 0) and (len(y) > 0):
                        bias = np.nanmean(y-x)
                        rmse = np.sqrt(np.nanmean((y-x)**2))
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
        
                ax.text(.05,.9,vn,transform=ax.transAxes, fontweight='bold')
                ax.axis([lim_dict[vn][0], lim_dict[vn][1], lim_dict[vn][0], lim_dict[vn][1]])
                ax.plot([lim_dict[vn][0], lim_dict[vn][1]], [lim_dict[vn][0], lim_dict[vn][1]],'-g')
                ax.grid(True)

            # station map
            if otype == 'bottle':
                ax = fig.add_subplot(1,4,4)
            elif otype == 'ctd':
                ax = fig.add_subplot(1,3,3)
            df_dict['obs'].plot(x='lon',y='lat',style='.g',legend=False, ax=ax)
            pfun.add_coast(ax)
            ax.axis([-130,-122,42,52])
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


