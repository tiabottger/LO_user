import numpy as np
import xarray as xr
import pickle
from datetime import datetime, timedelta
import pandas as pd
import sys

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import pinfo
from importlib import reload
reload(pfun)
reload(pinfo)

Ldir = Lfun.Lstart()
if '_mac' in Ldir['lo_env']: # mac version
    pass
else: # remote linux version
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

from cmocean import cm # have to import after matplotlib to work on remote machine

ds = xr.open_dataset('~/LO_output/extract/ae0_t0_xa0/moor/ae0/boundary_2020.01.01_2020.01.15.nc')

%matplotlib widget
ds.zeta.plot(figsize=(10, 4))
plt.title("Sea Surface Height (zeta)")
plt.xlabel("Time")
plt.ylabel("zeta (m)")
plt.grid(True)
plt.show()