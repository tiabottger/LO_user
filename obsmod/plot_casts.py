"""
Code to plot obs and mod casts at a given station, typically from ecology because
Those are monthly time series at a named location.
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
in_dir = Ldir['parent'] / 'LO_output' / 'obsmod'

# choices
sta_name = 'HCB003'
vn = 'DO (uM)'
#vn = 'SA'
"""
HCB003 is around Hoodsport
HCB004 is near Alderbrook
HCB007 is closer to the head of Lynch Cove
"""

# specify input (created by process_multi_bottle.py and process_multi_ctd.py)
otype = 'ctd'
in_fn = in_dir / ('multi_' + otype + '_' + year + '.p')
df_dict = pickle.load(open(in_fn, 'rb'))

source = 'ecology'
for gtx in df_dict.keys():
    df_dict[gtx] = df_dict[gtx].loc[df_dict[gtx].source==source,:]
    
for gtx in df_dict.keys():
    df_dict[gtx] = df_dict[gtx].loc[df_dict[gtx].name==sta_name,:]

# dicts to help with plotting
lim_dict = {'SA':(14,36),'CT':(0,20),'DO (uM)':(0,500),
    'NO3 (uM)':(0,50),'NH4 (uM)':(0,10),'DIN (uM)':(0,50),
    'DIC (uM)':(1500,2500),'TA (uM)':(1500,2500),'Chl (mg m-3)':(0,20)}
    
c_list = ['r','b','y','c']
c_dict = {'obs':'k'}
ii = 0
for gtx in df_dict.keys():
    if gtx == 'obs':
        pass
    else:
        c_dict[gtx] = c_list[ii]
        ii += 1
    
    
# plotting
plt.close('all')
pfun.start_plot(figsize=(20,12))
#pfun.start_plot(figsize=(12,8))
fig = plt.figure()

cid_list = df_dict['obs'].cid.unique()
cid_list.sort()
ii = 1
zbot = 0
ax_dict = dict()
for cid in cid_list:
    mo = df_dict[gtx].loc[df_dict[gtx].cid==cid,'time'].to_list()[0].month
    ax = fig.add_subplot(2,6,ii)
    for gtx in df_dict.keys():
        x = df_dict[gtx].loc[df_dict[gtx].cid==cid,vn].to_numpy()
        y = df_dict[gtx].loc[df_dict[gtx].cid==cid,'z'].to_numpy()
        zbot = np.min((zbot,np.min(y)))
        ax.plot(x,y,'-',c=c_dict[gtx])
        ax.text(.05,.1,'Month=%d' % (mo),transform=ax.transAxes,
            fontweight='bold',bbox=pfun.bbox)
        ax.set_xlim(lim_dict[vn])
    ax_dict[ii] = ax
    ax.set_xlim(lim_dict[vn])
    if ii not in [1,7]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Z [m]')
    if ii == 1:
        jj = 0
        for gtx in df_dict.keys():
            ax.text(.05,.9-jj/10,gtx,color=c_dict[gtx],transform=ax.transAxes,
                fontweight='bold',bbox=pfun.bbox)
            jj += 1
    ii += 1
    
    
for ii in ax_dict.keys():
    ax = ax_dict[ii]
    ax.set_ylim(zbot,0)
    
fig.suptitle('Station: %s, %s, %s' % (sta_name,vn,year))
plt.show()
