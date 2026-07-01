import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial import cKDTree
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun
from matplotlib.colors import TwoSlopeNorm

Ldir = Lfun.Lstart()

testing = False
year = '2014'
otype = 'bottle'

in_dir = Ldir['parent'] / 'LO_output' / 'obsmod'
out_dir = Ldir['parent'] / 'LO_output' / 'obsmod_plots'
Lfun.make_dir(out_dir)

plt.close('all')

# -------------------------------------------------------
# Load basin masks
# -------------------------------------------------------

mask_ds = xr.open_dataset('basin_masks_from_pugetsoundDObox.nc')

lon_rho = mask_ds['lon_rho'].values
lat_rho = mask_ds['lat_rho'].values

xy_grid = np.column_stack((lon_rho.ravel(), lat_rho.ravel()))
tree = cKDTree(xy_grid)

basin_var = {
    'hc': 'mask_hoodcanal',
    'ss': 'mask_southsound',
    'mb': 'mask_mainbasin',
    'wb': 'mask_whidbeybasin',
}

basin_order = ['ps', 'hc', 'ss', 'mb', 'wb']

basin_name = {
    'ps': 'Puget Sound',
    'hc': 'Hood Canal',
    'ss': 'South Sound',
    'mb': 'Main Basin',
    'wb': 'Whidbey Basin',
}

gtx_list = ['cas7_t1_x11ab', 'ssc']

model_name = {
    'cas7_t1_x11ab': 'LiveOcean',
    'ssc': 'SalishSeaCast',
}

# -------------------------------------------------------
# Load obs/model dictionary
# -------------------------------------------------------

in_fn = in_dir / f'combined_{otype}_{year}_cas7_t1_x11ab_ssc.pkl'
df0_dict = pickle.load(open(in_fn, 'rb'))

df0_dict = {
    k: v for k, v in df0_dict.items()
    if isinstance(v, pd.DataFrame)
}

if otype == 'bottle':
    for gtx in df0_dict.keys():
        if gtx == 'cas6_v0_live':
            df0_dict[gtx]['DIN'] = df0_dict[gtx]['NO3']
        else:
            df0_dict[gtx]['DIN'] = df0_dict[gtx]['NO3'] + df0_dict[gtx]['NH4']

# SSC chlorophyll from diatoms + flagellates
df0_dict['ssc']['Chl'] = (
    df0_dict['ssc']['DIAT'] +
    df0_dict['ssc']['FLAG']
) * 2

vn_list = ['SA', 'CT', 'DO', 'NO3', 'NH4', 'DIN', 'DIC', 'TA', 'Chl']

# -------------------------------------------------------
# Assign every observation to a basin
# -------------------------------------------------------

lon = df0_dict['obs']['lon'].to_numpy()
lat = df0_dict['obs']['lat'].to_numpy()

obs_xy = np.column_stack((lon, lat))
_, idx = tree.query(obs_xy)

jj, ii = np.unravel_index(idx, lon_rho.shape)

basin_masks = {}

# Overall Puget Sound row
basin_masks['ps'] = np.ones_like(lon, dtype=bool)

# Subbasins
for basin in ['hc', 'ss', 'mb', 'wb']:
    basin_grid = mask_ds[basin_var[basin]].values
    basin_masks[basin] = basin_grid[jj, ii] == 1

# -------------------------------------------------------
# Metric functions
# -------------------------------------------------------

def calc_metrics(obs, mod, obs_std_all):
    valid = np.isfinite(obs) & np.isfinite(mod)

    if valid.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    obs = obs[valid]
    mod = mod[valid]

    diff = mod - obs

    bias = np.mean(diff)
    rmse = np.sqrt(np.mean(diff**2))

    if obs_std_all > 0:
        nrmse = rmse / obs_std_all
    else:
        nrmse = np.nan

    # Nash-Sutcliffe Efficiency
    denom_nse = np.sum((obs - np.mean(obs))**2)
    if denom_nse > 0:
        nse = 1 - np.sum((mod - obs)**2) / denom_nse
    else:
        nse = np.nan

    # Willmott's Index of Agreement
    denom_d = np.sum(
        (np.abs(mod - np.mean(obs)) + np.abs(obs - np.mean(obs)))**2
    )

    if denom_d > 0:
        willmott_d = 1 - np.sum((mod - obs)**2) / denom_d
    else:
        willmott_d = np.nan

    return bias, rmse, nrmse, nse, willmott_d

# -------------------------------------------------------
# Calculate metrics
# -------------------------------------------------------

records = []

for vn in vn_list:

    x_raw = df0_dict['obs'][vn].to_numpy()

    # Convert DO from umol/L to mg/L
    if vn == 'DO':
        obs_all = x_raw * 0.032
    else:
        obs_all = x_raw

    # standard deviation from ALL observations for this variable
    valid_obs_all = np.isfinite(obs_all)
    obs_std_all = np.std(obs_all[valid_obs_all], ddof=1)

    for gtx in gtx_list:

        if vn not in df0_dict[gtx].columns:
            print(f"Skipping {vn} for {gtx} because it is missing")
            continue

        y_raw = df0_dict[gtx][vn].to_numpy()

        if vn == 'DO':
            mod_all = y_raw * 0.032
        else:
            mod_all = y_raw

        for basin in basin_order:

            mask = basin_masks[basin]

            obs_basin = obs_all[mask]
            mod_basin = mod_all[mask]

            bias, rmse, nrmse, nse, willmott_d = calc_metrics(
                obs_basin,
                mod_basin,
                obs_std_all
            )

            records.append({
                'model': gtx,
                'model_name': model_name[gtx],
                'variable': vn,
                'basin': basin,
                'basin_name': basin_name[basin],
                'bias': bias,
                'rmse': rmse,
                'nrmse': nrmse,
                'nse': nse,
                'willmott_d': willmott_d,
                'obs_std_all': obs_std_all,
                'n': np.sum(np.isfinite(obs_basin) & np.isfinite(mod_basin))
            })

metrics_df = pd.DataFrame(records)

# Save metrics table
csv_fn = out_dir / f'model_skill_metrics_{otype}_{year}.csv'
metrics_df.to_csv(csv_fn, index=False)

print(f"Saved metrics to: {csv_fn}")

# -------------------------------------------------------
# Heatmap plotting function
# -------------------------------------------------------

def plot_metric_heatmaps(metrics_df, metric, metric_label, cmap='viridis'):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 5.5),
        constrained_layout=True
    )

    # same color scale for both models
    vals = metrics_df[metric].to_numpy()
    vals = vals[np.isfinite(vals)]

    if metric == "nrmse":
        cmap = "viridis"
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        norm = None

    elif metric == "nse":
        cmap = "RdYlGn"
        vmin = -1
        vmax = 1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    elif metric == "willmott_d":
        cmap = "YlGn"
        vmin = 0
        vmax = 1
        norm = None

    else:
        cmap = "viridis"
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        norm = None

    last_im = None

    for ax, gtx in zip(axes, gtx_list):

        sub = metrics_df[metrics_df['model'] == gtx]

        mat = (
            sub.pivot(
                index='basin_name',
                columns='variable',
                values=metric
            )
            .reindex(index=[basin_name[b] for b in basin_order])
            .reindex(columns=vn_list)
        )

        data = mat.to_numpy(dtype=float)

        last_im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )

        ax.set_title(model_name[gtx], fontsize=13, fontweight='bold')

        ax.set_xticks(np.arange(len(vn_list)))
        ax.set_xticklabels(vn_list, rotation=45, ha='right')

        ax.set_yticks(np.arange(len(basin_order)))
        ax.set_yticklabels([basin_name[b] for b in basin_order])

        # write values in cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]

                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f'{val:.2f}',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white' if val > (vmin + vmax) / 2 else 'black'
                    )

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        shrink=0.85,
        pad=0.02
    )

    cbar.set_label(metric_label)

    fig.suptitle(
        f'{metric_label} by basin and variable, {year}',
        fontsize=15,
        fontweight='bold'
    )

    out_fn = out_dir / f'heatmap_{metric}_{year}.png'

    if testing:
        plt.show()
    else:
        plt.savefig(out_fn, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to: {out_fn}")

# -------------------------------------------------------
# Make heatmaps
# -------------------------------------------------------

plot_metric_heatmaps(
    metrics_df,
    metric='nrmse',
    metric_label='NRMSE'
)

plot_metric_heatmaps(
    metrics_df,
    metric='nse',
    metric_label='Nash-Sutcliffe Efficiency'
)

plot_metric_heatmaps(
    metrics_df,
    metric='willmott_d',
    metric_label="Willmott's Index of Agreement"
)