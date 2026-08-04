"""
Generic code to plot any mooring extraction
"""
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Ldir = Lfun.Lstart()

verbose = False

# choose first file
in_dir0 = Ldir['LOo'] / 'extract'
print('Choose first gtagex')
gtagex1 = Lfun.choose_item(in_dir0, tag='', exclude_tag='', itext='** Choose gtagex from list **')
in_dir1 = in_dir0 / gtagex1 / 'moor'

moor_name1 = Lfun.choose_item(in_dir1, itext='** Choose mooring extraction or folder from list **')
moor_item1 = in_dir1 / moor_name1
if moor_item1.is_file() and moor_name1[-3:]=='.nc':
    moor_fn1 = moor_item1
elif moor_item1.is_dir():
    moor_name1 = Lfun.choose_item(moor_item1, tag='.nc', itext='** Choose mooring extraction from list **')
    moor_fn1 = moor_item1 / moor_name1
 
# choose second file   
print('Choose second gtagex')
gtagex2 = Lfun.choose_item(in_dir0, tag='', exclude_tag='', itext='** Choose gtagex from list **')
in_dir2 = in_dir0 / gtagex2 / 'moor'

moor_name2 = Lfun.choose_item(in_dir2, itext='** Choose mooring extraction or folder from list **')
moor_item2 = in_dir2 / moor_name2
if moor_item2.is_file() and moor_name2[-3:]=='.nc':
    moor_fn2 = moor_item2
elif moor_item2.is_dir():
    moor_name2 = Lfun.choose_item(moor_item2, tag='.nc', itext='** Choose mooring extraction from list **')
    moor_fn2 = moor_item2 / moor_name2
    
out_dir = Ldir['LOo'] / 'intermodel_comparison' / f'{gtagex1}_vs_{gtagex2}'
out_dir.mkdir(parents=True, exist_ok=True) 

ds1 = xr.open_dataset(moor_fn1)
ds2 = xr.open_dataset(moor_fn2)

ot1 = pd.to_datetime(ds1.ocean_time.values)
ot2 = pd.to_datetime(ds2.ocean_time.values)

t = (ot1 - ot1[0]).total_seconds().to_numpy()

# Variables common to both datasets
VN_list = sorted(set(ds1.data_vars).intersection(ds2.data_vars))

if verbose:
    print('info'.center(60,'-'))
    for vn in VN_list:
        print(f'{vn} {ds1[vn].shape}')

# Variables to plot
vn2_list = ['zeta']

vn3_list = []
if 'salt' in VN_list:
    vn3_list += ['salt', 'temp']
if 'oxygen' in VN_list:
    vn3_list += ['oxygen']
if 'NO3' in VN_list:
    vn3_list += ['NO3']

vn2_list = [vn for vn in vn2_list if vn in VN_list]
vn3_list = [vn for vn in vn3_list if vn in VN_list]

plot_vars = vn2_list + vn3_list

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.close('all')
pfun.start_plot(figsize=(12, 3*len(plot_vars)))

fig, axes = plt.subplots(
    len(plot_vars), 1,
    figsize=(12, 3*len(plot_vars)),
    sharex=True
)

if len(plot_vars) == 1:
    axes = [axes]

for ax, vn in zip(axes, plot_vars):

    if vn in vn2_list:
        y1 = ds1[vn].values
        y2 = ds2[vn].values
    else:
        # bottom layer
        y1 = ds1[vn][:, 0].values
        y2 = ds2[vn][:, 0].values

    ax.plot(ot1, y1, lw=1.5, label=gtagex1)
    ax.plot(ot2, y2, lw=1.5, label=gtagex2)

    ax.set_ylabel(vn)
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel('Time')
fig.suptitle(moor_fn1.stem)

fig.tight_layout()

fig.savefig(
    out_dir / (moor_fn1.stem + "_comparison_timeseries.png"),
    dpi=300,
    bbox_inches="tight"
)

# also make a map with station location
fig2 = plt.figure(figsize=(8,8))
ax2 = fig2.add_subplot()
gfn = Ldir['data'] / 'grids' / 'cas7' / 'grid.nc'
gds = xr.open_dataset(gfn)
x = gds.lon_rho.values
y = gds.lat_rho.values
h = gds.h.values
m = gds.mask_rho.values
px, py = pfun.get_plon_plat(x,y)
m[m==0] = np.nan
ax2.pcolormesh(px,py,h,cmap='Blues', vmin=0, vmax=1)
pfun.add_coast(ax2)
mx = float(ds1.lon_rho.values)
my = float(ds1.lat_rho.values)
ax2.plot(mx,my,
         marker='*', color='r', markersize=22, markeredgecolor='k', markeredgewidth=1.5)
pad = .5
ax2.axis([mx-pad, mx+pad, my-pad, my+pad])
pfun.dar(ax2)

plt.savefig(out_dir / (moor_fn1.stem + '_map.png'), dpi=300, bbox_inches='tight')
pfun.end_plot()