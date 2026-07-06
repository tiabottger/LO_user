"""
Code to plot obs and mod casts at a given station, typically from ecology because
those are monthly time series at a named location.

Modified to plot multiple models to compare with observational casts data.

Modified to read in full station data.
"""

import sys
import pandas as pd
import numpy as np
import pickle
from matplotlib.patches import Rectangle
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun, zrfun
Ldir = Lfun.Lstart()

# --------------------
# Read model station files
# --------------------
station_names = ["SAR003", "PSS019", "SKG003", "HCB004", "HCB003", "ADM003",
                 "CMB003", "GOR001", "CSE001", "EAP001", "SIN001", "PSB003"]

data_lo = {}
data_ssc = {}

for station in station_names:
    filename_lo = f"{station}_2014.01.01_2014.12.31.nc"
    filename_ssc = f"{station}_2014.01.01_2014.12.31_ssc.nc"

    filepath_lo = "../intermodel_comparison/stations/" + filename_lo
    filepath_ssc = "../intermodel_comparison/stations/" + filename_ssc

    data_lo[station] = xr.open_dataset(filepath_lo)
    data_ssc[station] = xr.open_dataset(filepath_ssc)


year = '2014'
in_dir = Ldir['parent'] / 'LO_output' / 'obsmod'

# choices
sta_name = 'PSB003'
vn = 'DO'
# vn = 'SA'
# vn = 'CT'
# vn = 'Chl (mg m-3)'

# where to put output figures
out_dir = Ldir['LOo'] / 'obsmod_plots'
Lfun.make_dir(out_dir)

# variable names in each model file
var_dict = {
    "DO": {
        "obs": "DO",
        "lo": "oxygen",
        "ssc": "dissolved_oxygen",
    },
    "SA": {
        "obs": "SA",
        "lo": "salt",
        "ssc": "vosaline",
    },
    "CT": {
        "obs": "CT",
        "lo": "temp",
        "ssc": "votemper",
    },
    "Chl": {
        "obs": "Chl",
        "lo": "phytoplankton",
        "ssc": "chlorophyll",
    },
}

# specify input (created by process_multi_bottle.py and process_multi_ctd.py)
# read in for obs
otype = 'bottle'
in_fn = in_dir / ('combined_' + otype + '_' + year + '_cas7_t1_x11ab_ssc.pkl')
df_dict = pickle.load(open(in_fn, 'rb'))

obs = df_dict['obs'].copy()

obs = obs.loc[obs["name"] == sta_name, :].copy()

obs["time"] = pd.to_datetime(obs["time"])

def nearest_model_profile(ds, target_time, varname, model_type):
    """
    Returns x, z for nearest model profile to target_time.
    """

    if model_type == "lo":
        ds_t = ds.sel(ocean_time=target_time, method="nearest")
        
        x = ds_t[varname].values
        z = ds_t["z_rho"].values

    elif model_type == "ssc":
        ds_t = ds.sel(time_counter=target_time, method="nearest")
        
        x = ds_t[varname].values
        z = -ds_t["deptht"].values

    # Convert DO to mg/L
    if vn == "DO":
        x = x * 0.032

    x = np.asarray(x).squeeze()
    z = np.asarray(z).squeeze()

    # Remove fill values (SSC sometimes has zeros below the bottom)
    if model_type == "ssc":
        good = np.isfinite(x) & np.isfinite(z) & (x != 0)
    else:
        good = np.isfinite(x) & np.isfinite(z)
    return x[good], z[good]
      
# plotting
plt.close('all')

# sort cids by time
cid_list = obs.groupby("cid")["time"].first().sort_values().index.to_numpy()


################################################################ 
##                     Plot cast profiles                     ##
################################################################ 

pfun.start_plot(figsize=(15,8))
fig = plt.figure()

plt.subplots_adjust(wspace=0, hspace=0.1)

# colors
c_dict = {
    "obs": "k",
    "lo": "red",
    "ssc": "royalblue",
}

labels = [
    "(a) January", "(b) February", "(c) March", "(d) April",
    "(e) May", "(f) June",
    "(g) July", "(h) August",
    "(i) September", "(j) October",
    "(k) November", "(l) December"
]

lim_dict = {
    "SA": (15, 36),
    "CT": (0, 20),
    "DO": (0, 14),
    "NO3": (0, 50),
    "NH4": (0, 10),
    "DIN": (0, 50),
    "DIC": (1500, 2500),
    "TA": (1500, 2500),
    "Chl": (0, 20),
}

ax_dict = {}
zbot = 0

ds_lo = data_lo[sta_name]
ds_ssc = data_ssc[sta_name]

for i, cid in enumerate(cid_list[:12]):

    ax = fig.add_subplot(2, 6, i + 1)

    obs_cast = obs.loc[obs["cid"] == cid, :].copy()
    obs_cast = obs_cast.sort_values("z")

    target_time = obs_cast["time"].iloc[0]

    # observations
    x_obs = obs_cast[var_dict[vn]["obs"]].to_numpy()
    z_obs = obs_cast["z"].to_numpy()

    if vn == "DO":
        x_obs = x_obs * 0.032

    good_obs = np.isfinite(x_obs) & np.isfinite(z_obs)
    
    ax.plot(
    x_obs[good_obs],
    z_obs[good_obs],
    linestyle="None",     
    marker="o",           
    markersize=6,
    markerfacecolor="k",
    markeredgecolor="k",
    color="k",
    label="Observations",
    zorder=10,
)

    # LiveOcean
    x_lo, z_lo = nearest_model_profile(
        ds_lo,
        target_time,
        var_dict[vn]["lo"],
        model_type="lo",
    )
    ax.plot(
        x_lo,
        z_lo,
        color=c_dict["lo"],
        linewidth=1.5,
        label="LiveOcean",
    )

    # SalishSeaCast
    x_ssc, z_ssc = nearest_model_profile(
        ds_ssc,
        target_time,
        var_dict[vn]["ssc"],
        model_type="ssc",
    )
    ax.plot(
        x_ssc,
        z_ssc,
        color=c_dict["ssc"],
        linewidth=1.5,
        label="SalishSeaCast",
    )

    zbot = min(zbot, np.nanmin(z_obs), np.nanmin(z_lo), np.nanmin(z_ssc))

    ax.text(
        0.1, 0.05, labels[i],
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )

    ax.set_xlim(lim_dict[vn])

    if i not in [0, 6]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Z [m]")

    if i < 6:
        ax.set_xticklabels([])

    ax.tick_params(axis="x", rotation=30)

    ax.set_facecolor("#EEEEEE")
    ax.grid(True, color="w", linewidth=2)

    for border in ["top", "right", "bottom", "left"]:
        ax.spines[border].set_visible(False)

    ax_dict[i + 1] = ax


for ax in ax_dict.values():
    ax.set_ylim(zbot, 0)

handles, leg_labels = ax_dict[1].get_legend_handles_labels()

fig.legend(
    handles,
    leg_labels,
    loc=(0.35, 0.88),
    ncol=3,
    frameon=False,
    labelcolor="linecolor",
    prop=dict(weight="bold"),
)

fig.suptitle(
    f"Station: {sta_name}, {vn}, {year}",
    fontweight="bold",
    fontsize=16,
)

fig.tight_layout()
fig.subplots_adjust(top=0.84, bottom=0.1)

plt.savefig(out_dir / f"{sta_name}_{vn}_{year}_profile_from_station_files.png", dpi=300)
plt.show()
