"""
Code to plot obs and mod casts at a given station, typically from ecology because
Those are monthly time series at a named location.

It allows for more than one gtagex.
"""

import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun, zrfun
Ldir = Lfun.Lstart()

year = '2017'
in_dir = Ldir['parent'] / 'LPM_output' / 'obsmod'

# which runs to use (you must have run combine_obs_mod.py for each one)
#gtagex_list = ['cas7_t0_x4b', 'cas7_t1_x10ab', 'cas7_t1_x11ab']
gtagex_list = ['cas7_t0_x4b', 'cas7_t1_x10ab']

# data choices
source = 'ecology_nc'
sta_name = 'HCB003'
"""
HCB003 is around Hoodsport
HCB004 is near Alderbrook
HCB007 is closer to the head of Lynch Cove
HCB010 is near the connection to Dabob Bay
PSB003 is in Main Basin
BUD005 is in Budd Inlet
SAR003 is in Saratoga Passage - Whidbey Basin
SJF001 is in Eastern JdF

NOTE: There is a station map plot in LO_output/obs/ecology.
"""

# property to plot
vn = 'DO (uM)'
#vn = 'NO3 (uM)'
#vn = 'NH4 (uM)'
#vn = 'SA'
#vn = 'CT'
if vn in ['SA','CT','DO (uM)']:
    otype = 'ctd'
else:
    otype = 'bottle'

full_dict = dict()
for gtagex in gtagex_list:
    in_fn = in_dir / ('combined_' + otype + '_' + year + '_' + gtagex + '.p')
    df_dict = pickle.load(open(in_fn, 'rb'))
    # filter for source name
    for gtx in df_dict.keys():
        df_dict[gtx] = df_dict[gtx].loc[df_dict[gtx].source==source,:]
    # filter for station name
    for gtx in df_dict.keys():
        df_dict[gtx] = df_dict[gtx].loc[df_dict[gtx].name==sta_name,:]
    full_dict[gtagex] = df_dict

# dicts to help with plotting
lim_dict = {'SA':(14,36),'CT':(0,25),'DO (uM)':(0,500),
    'NO3 (uM)':(0,50),'NH4 (uM)':(0,10),'DIN (uM)':(0,50),
    'DIC (uM)':(1500,2500),'TA (uM)':(1500,2500),'Chl (mg m-3)':(0,20)}
# line colors
c_list = ['r','b','g','c']
c_dict = {'obs':'k'}
ii = 0
for gtagex in gtagex_list:
        c_dict[gtagex] = c_list[ii]
        ii += 1
    
# plotting
plt.close('all')
pfun.start_plot(figsize=(20,12))
# pfun.start_plot(figsize=(12,8))
fig = plt.figure()

kk = 0
ax_dict = dict()
for gtx in gtagex_list:
    df_dict = full_dict[gtx]
    cid_list = df_dict['obs'].cid.unique()
    cid_list.sort()
    ii = 1
    zbot = 0
    for cid in cid_list:
        mo = df_dict[gtx].loc[df_dict[gtx].cid==cid,'time'].to_list()[0].month
        if kk == 0:
            ax = fig.add_subplot(2,6,ii)
            ax_dict[ii] = ax
        else:
            ax = ax_dict[ii]
        for gtx in df_dict.keys():
            x = df_dict[gtx].loc[df_dict[gtx].cid==cid,vn].to_numpy()
            y = df_dict[gtx].loc[df_dict[gtx].cid==cid,'z'].to_numpy()
            zbot = np.min((zbot,np.min(y)))
            if otype == 'bottle':
                ax.plot(x,y,'-o',c=c_dict[gtx])
            elif otype == 'ctd':
                ax.plot(x,y,'-',c=c_dict[gtx])
            ax.text(.05,.1,'Month=%d' % (mo),transform=ax.transAxes,
                fontweight='bold',bbox=pfun.bbox)
            ax.set_xlim(lim_dict[vn])
        ax.set_xlim(lim_dict[vn])
        if ii not in [1,7]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Z [m]')
        if (ii == 1) and (kk == 0):
            jj = 0
            for gtx_label in ['obs'] + gtagex_list:
                ax.text(.05,.9-jj/10-kk/10,gtx_label,color=c_dict[gtx_label],transform=ax.transAxes,
                    fontweight='bold',bbox=pfun.bbox)
                jj += 1
        ii += 1
    kk += 1
    
for ii in ax_dict.keys():
    ax = ax_dict[ii]
    ax.set_ylim(zbot,0)
    
fig.suptitle('Station: %s, %s, %s' % (sta_name,vn,year))
plt.show()
