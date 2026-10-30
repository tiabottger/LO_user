'''
Ecology SSM WQM box extraction
'''

import os
import pickle
import xarray as xr
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

nc_path = Path(
    "/dat2/tbottger/SSM_ecology/2014_Exist_W3_WQ_hot10_Base4_nodes.nc"
)

mat_path = Path(
    "/dat2/tbottger/SSM_ecology/2014_Exist_W3_WQ_hot10_Base4_nodes.mat"
)

output_dir = (
    f"/dat2/tbottger/SSM_ecology/"
    f"ecology_box/"
)

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    f"SSM_ecology_2014_box_subset.nc",
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
# DEFINE BOX
# ========================================================

# box extent
# [-123.29, -122.1, 46.95, 48.93] pugetsoundDO from job_definitions in LO_user/extract/box
lon0, lon1 = -123.29, -122.1
lat0, lat1 = 46.95, 48.93
# note: triangular elements selected by their centroids, then includes all nodes belonging to 
# those elements, so the actual extent of the extracted data is slightly larger than the box defined above.

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
    "zeta": "Var_5",
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
horiz = grd.horiz
vert = grd.vert
cellidx = grd.cellidx

# Coordinates are in the same order as 1-based horizontal node IDs.
lon = np.asarray(horiz.Lon, dtype=float).ravel()
lat = np.asarray(horiz.Lat, dtype=float).ravel()

# Node-specific positive-down bottom depths
bottom_depth = np.asarray(
    grd.horiz.elev_bottom,
    dtype=float
).ravel()

# Vertical-grid metadata
layer_thickness = np.asarray(
    grd.vert.thickness,
    dtype=float
).ravel()

layer_top = np.asarray(
    grd.vert.elev_top,
    dtype=float
).ravel()

# Fraction of the total water depth at each layer center.
# Assumes elev_top is negative-down and thickness is positive.
layer_fraction = -(layer_top - layer_thickness / 2)

assert len(lon) == n_nodes
assert len(lat) == n_nodes
assert len(bottom_depth) == n_nodes
assert len(layer_thickness) == n_layers
assert len(layer_top) == n_layers

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

assert np.array_equal(node_mapping, expected_nodes)
assert np.array_equal(layer_mapping, expected_layers)

print("Horizontal nodes:", len(lon))
print("Layers per node:", n_layers)
print("Flattened entries:", n_flat)
print("MAT node/layer mapping verified.")

print("Number of nodes:", n_nodes)
print("Number of layers:", n_layers)
print("Layer-center fractions:", layer_fraction)

# ============================================================
# VERIFY FLATTENED NODE-LAYER MAPPING
# ============================================================

node_mapping = np.asarray(
    cellidx.I
).ravel()

layer_mapping = np.asarray(
    cellidx.K
).ravel()

expected_nodes = np.repeat(
    np.arange(1, n_nodes + 1),
    n_layers
)

expected_layers = np.tile(
    np.arange(1, n_layers + 1),
    n_nodes
)

assert np.array_equal(node_mapping, expected_nodes)
assert np.array_equal(layer_mapping, expected_layers)

print("MAT node/layer mapping verified.")

# ============================================================
# IDENTIFY NODES INSIDE GEOGRAPHIC BOX
# ============================================================

node_inside = (
    (lon >= lon0) & (lon <= lon1) &
    (lat >= lat0) & (lat <= lat1)
)

node_indices = np.flatnonzero(node_inside)

if len(node_indices) == 0:
    raise ValueError(
        "No Ecology nodes found inside the requested box."
    )

# Flattened IJK indices for selected nodes.
# Original ordering is node-major, layer-fast.

ijk_indices = (
    node_indices[:, None] * n_layers +
    np.arange(n_layers)[None, :]
).ravel()

print(f"Nodes selected: {len(node_indices):,}")
print(f"Layers per node: {n_layers}")
print(f"Flattened entries: {len(ijk_indices):,}")

# ============================================================
# SAVE STATIC GRID INFORMATION
# ============================================================

mesh_ds = xr.Dataset(
    coords={
        "local_node": np.arange(len(node_indices)),
        "layer": np.arange(n_layers),
        "local_ijk": np.arange(len(ijk_indices)),
    },

    data_vars={
        "lon": (
            "local_node",
            lon[node_indices]
        ),

        "lat": (
            "local_node",
            lat[node_indices]
        ),

        "bottom_depth": (
            "local_node",
            bottom_depth[node_indices]
        ),

        "layer_fraction": (
            "layer",
            layer_fraction
        ),

        "original_node_index": (
            "local_node",
            node_indices.astype(np.int32)
        ),

        "original_ijk_index": (
            "local_ijk",
            ijk_indices.astype(np.int32)
        ),

        "original_node_id": (
            "local_ijk",
            node_mapping[ijk_indices].astype(np.int32)
        ),

        "original_layer_id": (
            "local_ijk",
            layer_mapping[ijk_indices].astype(np.int32)
        ),
    },

    attrs={
        "description": "Ecology/WQM regional box extraction",
        "year": 2014,
        "bbox": f"{lon0}, {lon1}, {lat0}, {lat1}",
        "vertical_depth_convention": "Negative downward",
        "flattened_mapping": "Node-major, layer-fast",
    },
)

# ============================================================
# VALIDATE NETCDF DIMENSIONS
# ============================================================

n_times = len(nc.dimensions["Time"])

assert len(nc.dimensions["IJK"]) == n_flat

for name, var in ecology_var_objects.items():

    assert var.shape == (n_times, n_flat), (
        f"{name}: unexpected shape {var.shape}"
    )

print("\nNetCDF dimensions verified.")
print("Time steps:", n_times)
print("Flattened entries:", n_flat)

# ============================================================
# VALIDATE HOURLY TIMELINE
# ============================================================

model_times = pd.date_range(
    start=f"2014-01-01 00:00:00",
    periods=n_times,
    freq="h"
)

print("First model time:", model_times[0])
print("Last model time:", model_times[-1])

# ============================================================
# LOOP THROUGH HOURLY DATA, DAILY AVERAGE
# ============================================================

daily_datasets = []

for day in range(1, 366):

    start_idx = (day - 1) * 24
    stop_idx = day * 24

    if stop_idx > n_times:
        break

    print(f"Processing day {day}/365")

    data_vars = {}

    for name, var in ecology_var_objects.items():

        # Read selected nodes and all their vertical layers.
        # Shape: (24, selected_nodes * n_layers)

        arr = np.ma.filled(
            var[start_idx:stop_idx, :][:, ijk_indices],
            np.nan
        ).astype(float)

        # Daily mean, independently for each node/layer.
        arr_daily = np.nanmean(
            arr,
            axis=0
        )

        # Restore node/layer dimensions.
        arr_daily = arr_daily.reshape(
            len(node_indices),
            n_layers
        )

        data_vars[name] = (
            ("local_node", "layer"),
            arr_daily
        )

    day_timestamp = np.datetime64(
        pd.Timestamp(2014, 1, 1) +
        pd.Timedelta(days=day - 1),
        "ns"
    )

    daily_ds = xr.Dataset(
        data_vars={
            name: (
                ("time", "local_node", "layer"),
                values[np.newaxis, :, :]
            )
            for name, (_, values) in data_vars.items()
        },

        coords={
            "time": [day_timestamp],
            "local_node": np.arange(len(node_indices)),
            "layer": np.arange(n_layers),
        },
    )

    daily_datasets.append(daily_ds)

# ============================================================
# COMBINE DAILY DATA
# ============================================================

annual_ds = xr.concat(
    daily_datasets,
    dim="time"
)

annual_ds = xr.merge(
    [annual_ds, mesh_ds],
    compat="override"
)

annual_ds.attrs.update(mesh_ds.attrs)

# ============================================================
# VALIDATE OUTPUT
# ============================================================

print("\nOutput dimensions:")
print(dict(annual_ds.sizes))

print("\nVariable shapes:")

for name in ecology_vars:
    print(f"{name}: {annual_ds[name].shape}")

print("\nFinite values:")

for name in ecology_vars:
    print(
        f"{name}: "
        f"{np.isfinite(annual_ds[name].values).sum():,}"
    )

# ============================================================
# SAVE
# ============================================================

annual_ds.to_netcdf(
    output_file,
    engine="netcdf4"
)

print("\nExtraction complete.")
print(f"Output: {output_file}")

print(annual_ds)