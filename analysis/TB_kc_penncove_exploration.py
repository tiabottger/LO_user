"""
Code for initial exploration of KC CTD cast data. Referencing plot_casts.py and combine_obs_mod.py, 
but written for exploration without comparing to model results for now.

Tia Bottger 
Last modified:
12/1/25
"""

import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun, zrfun
Ldir = Lfun.Lstart()

# set data source and type
source = 'kc_whidbeyBasin'
otype = 'ctd'

# years available 2022, 2023, 2024
year = '2022'

obs_fn = Ldir['LOo'] / 'obs' / source / otype / (year + '.p')

obs_df = pd.read_pickle(obs_fn)
