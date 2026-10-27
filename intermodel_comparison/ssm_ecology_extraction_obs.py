
# ============================================================
# Ecology/WQM extraction at observation locations, times, depths
#
# Adds Ecology/WQM DO values to an existing LO + SSC
# observation pickle.
#
# Strategy:
#   1. Load existing observation targets
#   2. Load horizontal coordinates from the MAT grid
#   3. Build a nearest-node KDTree
#   4. Match observations to the nearest hourly timestamp
#   5. Find the nearest vertical layer using Var_9
#   6. Extract Var_10 (DO, mg/L) at the matching node/layer/time
#   7. Preserve exactly one row per observation and original order
#   8. Save the result under data["ecology"]
#
# The annual NetCDF is accessed locally using memory mapping.
# Only requested node/layer profiles are read.
# ============================================================

import pickle
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.io import loadmat, netcdf_file
from netCDF4 import Dataset
from scipy.spatial import cKDTree

from lo_tools import Lfun


# ============================================================
# PATHS
# ============================================================

Ldir = Lfun.Lstart()

year = 2014

pkl_path = (
    Ldir["LOo"] / "obsmod" /
    f"combined_bottle_{year}_cas7_t1_x11ab_ssc_ssmpnnl.pkl"
)

out_pkl_path = (
    Ldir["LOo"] / "obsmod" /
    f"combined_bottle_{year}_cas7_t1_x11ab_ssc_ssmpnnl_ecology.pkl"
)

nc_path = Path(
    "/dat2/tbottger/SSM_ecology/2014_Exist_W3_WQ_hot10_Base4_nodes.nc"
)

mat_path = Path(
    "/dat2/tbottger/SSM_ecology/2014_Exist_W3_WQ_hot10_Base4_nodes.mat"
)

# Verified grid dimensions
n_nodes = 16012
n_layers = 10
n_flat = n_nodes * n_layers

nc = Dataset(
    str(nc_path),
    mode="r"
)

# ========================================================
# VARIABLES TO EXTRACT
# ========================================================

# Var_10 DO mg/L
# Var_14 NH3 mg/L
# Var_15 NO3 mg/L
# Var_18 Temperature degC
# Var_19 Salinity ppt
# Var_30 DIC mmol/m3
# Var_31 TAlk mmol/m3

ecology_vars = {
    "DO": "Var_10",
    "NH3": "Var_14",
    "NO3": "Var_15",
    "CT": "Var_18",
    "SA": "Var_19",
    "DIC": "Var_30",
    "TA": "Var_31",
}

ecology_var_objects = {
    name: nc.variables[varname]
    for name, varname in ecology_vars.items()
}

depth_varname = "Var_9"
# Var_9 has incomplete depth information, and is zero for all but 230 nodes
# (16012 total). Instead we will use the depth bathymetry from the .mat file to reconstruct 
# layer depths

# ============================================================
# LOAD OBSERVATIONS
# ============================================================

print("\nLoading observation pickle...")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

obs = data["obs"].copy()
obs["time"] = pd.to_datetime(obs["time"])

obs["_obs_index"] = np.arange(len(obs))

print("Number of observations:", len(obs))
print("Observation date range:")
print("  First:", obs["time"].min())
print("  Last: ", obs["time"].max())

if obs.empty:
    raise ValueError("Observation dataframe is empty.")

if obs[["lon", "lat", "time", "z"]].isna().any().any():
    raise ValueError(
        "Observations contain missing lon, lat, time, or z values. "
        "Handle these rows before extraction."
    )

if (
    (obs["time"] < pd.Timestamp(f"{year}-01-01")).any()
    or
    (obs["time"] >= pd.Timestamp(f"{year + 1}-01-01")).any()
):
    raise ValueError(
        f"All observation times must fall within {year}."
    )


# ============================================================
# LOAD MAT HORIZONTAL GRID AND READ DEPTHS
# ============================================================

print("\nLoading Ecology/WQM horizontal grid...")

mat = loadmat(
    mat_path,
    squeeze_me=True,
    struct_as_record=False
)

grd = mat["grd"]

# Node-specific positive-down bottom depths
bottom_depth = np.asarray(
    grd.horiz.elev_bottom,
    dtype=float
)

# Vertical-grid metadata
layer_thickness = np.asarray(
    grd.vert.thickness,
    dtype=float
)

layer_top = np.asarray(
    grd.vert.elev_top,
    dtype=float
)

# Fraction of the total water depth at each layer center.
# Assumes elev_top is negative-down and thickness is positive.
layer_fraction = -(layer_top - layer_thickness / 2)

# Shape: (n_nodes, n_layers)
# Negative-down reconstructed layer-center depths in metres.
reconstructed_depths = -(
    bottom_depth[:, None] * layer_fraction[None, :]
)


horiz = mat["grd"].horiz
cellidx = mat["grd"].cellidx

# Coordinates are in the same order as 1-based horizontal node IDs.
lon_model = np.asarray(horiz.Lon, dtype=float).ravel()
lat_model = np.asarray(horiz.Lat, dtype=float).ravel()

assert len(lon_model) == n_nodes
assert len(lat_model) == n_nodes

# Verify the confirmed node-major, layer-fast mapping.
node_mapping = np.asarray(cellidx.I).ravel()
layer_mapping = np.asarray(cellidx.K).ravel()

expected_nodes = np.repeat(
    np.arange(1, n_nodes + 1),
    n_layers
)

expected_layers = np.tile(
    np.arange(1, n_layers + 1),
    n_nodes
)

assert reconstructed_depths.shape == (n_nodes, n_layers)
assert np.all(np.isfinite(reconstructed_depths))
assert np.all(reconstructed_depths < 0)

assert np.array_equal(node_mapping, expected_nodes)
assert np.array_equal(layer_mapping, expected_layers)

print("Horizontal nodes:", len(lon_model))
print("Layers per node:", n_layers)
print("Flattened entries:", n_flat)
print("MAT node/layer mapping verified.")

print("Reconstructed layer-depth array:", reconstructed_depths.shape)
print("Number of nodes:", n_nodes)
print("Number of layers:", n_layers)
print("Layer-center fractions:", layer_fraction)


# ============================================================
# BUILD SPHERICAL KD-TREE
# ============================================================

def lonlat_to_xyz(lon_deg, lat_deg):
    """Convert longitude/latitude to 3-D unit-sphere coordinates."""
    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)

    return np.column_stack([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])


print("\nBuilding nearest-node KDTree...")

model_xyz = lonlat_to_xyz(lon_model, lat_model)
tree = cKDTree(model_xyz)

obs_lon = obs["lon"].to_numpy(dtype=float)
obs_lat = obs["lat"].to_numpy(dtype=float)

obs_xyz = lonlat_to_xyz(obs_lon, obs_lat)

chord_distance, nearest_node = tree.query(obs_xyz)

# nearest_node is zero-based, suitable for NumPy indexing.
nearest_node = nearest_node.astype(int)

# Convert unit-sphere chord distance to great-circle distance.
central_angle = 2.0 * np.arcsin(
    np.clip(chord_distance / 2.0, 0.0, 1.0)
)

earth_radius_km = 6371.0
horizontal_distance_km = earth_radius_km * central_angle

obs["_model_node"] = nearest_node
obs["_model_lon"] = lon_model[nearest_node]
obs["_model_lat"] = lat_model[nearest_node]
obs["_model_horizontal_distance_km"] = horizontal_distance_km

print(
    "Mean nearest-node distance: "
    f"{np.mean(horizontal_distance_km):.3f} km"
)
print(
    "Maximum nearest-node distance: "
    f"{np.max(horizontal_distance_km):.3f} km"
)


# ============================================================
# CREATE THE VALIDATED HOURLY TIMELINE
# ============================================================

model_times = pd.date_range(
    start=f"{year}-01-01 00:00:00",
    periods=8760,
    freq="h"
)

obs_times = pd.DatetimeIndex(obs["time"])

# Convert timestamps to hours relative to January 1.
hours_from_start = (
    (obs_times - model_times[0])
    / pd.Timedelta(hours=1)
).to_numpy(dtype=float)

# Nearest hourly record.
time_idx = np.rint(hours_from_start).astype(int)

if np.any((time_idx < 0) | (time_idx >= len(model_times))):
    raise ValueError("Some observations fall outside the model timeline.")

matched_model_times = model_times.take(time_idx)

time_difference_min = np.abs(
    (matched_model_times - obs_times)
    / pd.Timedelta(minutes=1)
)

obs["_model_time_idx"] = time_idx
obs["_model_time"] = matched_model_times
obs["_model_time_difference_min"] = time_difference_min

print("\nTime matching:")
print(f"Mean |time difference|: " f"{np.mean(np.abs(time_difference_min)):.2f} min")
print(f"Max |time difference|:  {time_difference_min.max():.2f} min")


# ============================================================
# OPEN ANNUAL NETCDF
# ============================================================

try:
    print("NetCDF dimensions:")
    for name, dim in nc.dimensions.items():
        print(f"  {name}: {dim}")

        n_times, n_flat = nc.variables["Var_10"].shape

    assert n_flat == n_nodes * n_layers, (
        f"Flattened NetCDF dimension {n_flat} does not match "
        f"{n_nodes} nodes x {n_layers} layers"
    )

    print("\nNetCDF time steps:", n_times)
    print("Flattened node-layer dimension:", n_flat)

    # Check that each requested variable has the expected layout.
    for name, var in ecology_var_objects.items():
        assert var.shape == (n_times, n_flat), (
            f"{name} has shape {var.shape}; "
            f"expected {(n_times, n_flat)}"
        )

    # --------------------------------------------------------
    # Cache each requested (time, node) profile.
    #
    # Each profile has 10 layer depths and 10 DO values.
    # Copies ensure the cache does not depend on the NetCDF
    # variable's underlying memory map after the read.
    # --------------------------------------------------------

    @lru_cache(maxsize=20000)
    def get_profile(t_idx, node_row):
        start = node_row * n_layers
        stop = start + n_layers
        
        depths = reconstructed_depths[node_row]

        profiles = {
            name: np.asarray(
                var[t_idx, start:stop],
                dtype=float
            ).copy()
            for name, var in ecology_var_objects.items()
        }

        return depths, profiles

    # ========================================================
    # EXTRACT EACH OBSERVATION
    # ========================================================

    print("\nExtracting Ecology/WQM values...")

    n_obs = len(obs)

    ssm_z = np.full(n_obs, np.nan)
    ssm_layer = np.full(n_obs, -1, dtype=int)
    
    ssm_vars = {
        name: np.full(n_obs, np.nan)
        for name in ecology_vars
    }

    for row in range(n_obs):
        t_idx = int(time_idx[row])
        node_row = int(nearest_node[row])

        depths, profiles = get_profile(t_idx, node_row)

        obs_z = float(obs.iloc[row]["z"])

        valid_depth = np.isfinite(depths)

        if not valid_depth.any():
            continue

        # Choose the layer whose center is closest to the
        # observation depth. Both are assumed positive downward.
        depth_difference = np.where(
            valid_depth,
            np.abs(depths - obs_z),
            np.inf
    )

        layer_idx = int(np.argmin(depth_difference))

        ssm_layer[row] = layer_idx + 1  # 1-based layer number
        ssm_z[row] = depths[layer_idx]

        for name, profile in profiles.items():
            if np.isfinite(profile[layer_idx]):
                ssm_vars[name][row] = profile[layer_idx]

        if (row + 1) % 500 == 0 or row + 1 == n_obs:
            print(f"Processed {row + 1}/{n_obs} observations")


    # ========================================================
    # BUILD OUTPUT DATAFRAME
    # ========================================================

    ecology = pd.DataFrame({
        "_obs_index": obs["_obs_index"].to_numpy(),

        # Original observation identifiers and targets
        "cid": obs["cid"].to_numpy(),
        "lon": obs["lon"].to_numpy(),
        "lat": obs["lat"].to_numpy(),
        "time": obs["time"].to_numpy(),
        "z": obs["z"].to_numpy(),

        # Model match information
        "ssm_time": matched_model_times,
        "ssm_z": ssm_z,
        "ssm_node": nearest_node + 1,  # Convert to 1-based ID
        "ssm_layer": ssm_layer,
        "ssm_lon": lon_model[nearest_node],
        "ssm_lat": lat_model[nearest_node],
        "horizontal_distance_km": horizontal_distance_km,
        "vertical_difference_m": ssm_z - obs["z"].to_numpy(dtype=float),
        "time_difference_min": time_difference_min,

        **ssm_vars
    })

finally:
    nc.close()

    
# ============================================================
# RESTORE AND VERIFY EXACT OBSERVATION ORDER
# ============================================================

ecology = (
    ecology
    .sort_values("_obs_index")
    .reset_index(drop=True)
)

assert len(ecology) == len(obs)
assert ecology["_obs_index"].is_unique
assert np.array_equal(
    ecology["_obs_index"].to_numpy(),
    np.arange(len(obs))
)

assert np.array_equal(
    ecology["cid"].to_numpy(),
    obs["cid"].to_numpy()
)

assert np.array_equal(
    pd.to_datetime(ecology["time"]).to_numpy(),
    pd.to_datetime(obs["time"]).to_numpy()
)

assert np.allclose(
    ecology["z"].to_numpy(dtype=float),
    obs["z"].to_numpy(dtype=float),
    equal_nan=True
)

print("\nObservation alignment checks passed.")


# ============================================================
# EXTRACTION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ECOLOGY/WQM EXTRACTION SUMMARY")
print("=" * 60)

print("Observations:", len(obs))
print("Extracted rows:", len(ecology))
print("Finite DO values:", np.isfinite(ecology["DO"]).sum())
print("Missing DO values:", ecology["DO"].isna().sum())

print(
    "Mean horizontal distance: "
    f"{ecology['horizontal_distance_km'].mean():.3f} km"
)
print(
    "Maximum horizontal distance: "
    f"{ecology['horizontal_distance_km'].max():.3f} km"
)

print(
    "Mean |vertical difference|: "
    f"{np.nanmean(np.abs(ecology['vertical_difference_m'])):.3f} m"
)
print(
    "Maximum |vertical difference|: "
    f"{np.nanmax(np.abs(ecology['vertical_difference_m'])):.3f} m"
)

print(
    "Mean |time difference|: "
    f"{ecology['time_difference_min'].mean():.2f} min"
)
print(
    "Maximum |time difference|: "
    f"{ecology['time_difference_min'].max():.2f} min"
)

print("\nDO range (mg/L):")
print(ecology["DO"].describe())


# ============================================================
# ADD TO DATA DICTIONARY AND SAVE
# ============================================================

# Remove the internal row-index column from the final model table.
ecology = ecology.drop(columns="_obs_index")

# ------------------------------------------------------------
# CONVERT ECOLOGY CONCENTRATIONS FROM mg/L TO uM to match 
# conventions in other models
# ------------------------------------------------------------

# DO: O2 molecular weight = 31.998 g/mol
ecology["DO"] *= 1000 / 31.998

# NO3: nitrate molecular weight = 62.004 g/mol
ecology["NO3"] *= 1000 / 62.004

# NH3: ammonia molecular weight = 17.031 g/mol
ecology["NH3"] *= 1000 / 17.031

# Preserve existing LO, SSC, or SSM entries if present.
data["ecology"] = ecology

# Restore the original observation table without temporary columns.
temporary_columns = [
    "_obs_index",
    "_model_node",
    "_model_lon",
    "_model_lat",
    "_model_horizontal_distance_km",
    "_model_time_idx",
    "_model_time",
    "_model_time_difference_min",
]

data["obs"] = obs.drop(
    columns=temporary_columns,
    errors="ignore"
).reset_index(drop=True)

# Final alignment checks
assert len(data["obs"]) == len(data["ecology"])
assert np.array_equal(
    data["obs"]["cid"].to_numpy(),
    data["ecology"]["cid"].to_numpy()
)
assert np.array_equal(
    pd.to_datetime(data["obs"]["time"]).to_numpy(),
    pd.to_datetime(data["ecology"]["time"]).to_numpy()
)
assert np.allclose(
    data["obs"]["z"].to_numpy(dtype=float),
    data["ecology"]["z"].to_numpy(dtype=float),
    equal_nan=True
)

print("\nSaving combined pickle...")

with open(out_pkl_path, "wb") as f:
    pickle.dump(
        data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print(f"Saved to:\n{out_pkl_path}")
