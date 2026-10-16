'''
SSM WQM box extraction

SSM data:
  https://s3.kopah.uw.edu/ssm/wqm/
  
SSM grid:
    EPSG:26910 (NAD83 / UTM Zone 10N)

fsspec + scipy access the NetCDF-3 files remotely using HTTP range requests.
'''

import os
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from scipy.io import netcdf_file
from scipy.spatial import cKDTree
import pyproj
import gsw
from pyproj import Transformer

from lo_tools import Lfun

import sys
from pathlib import Path

Ldir = Lfun.Lstart()

# ============================================================
# USER SETTINGS
# ============================================================

year = 2014

# box extent
# [-123.29, -122.1, 46.95, 48.93] pugetsoundDO from job_definitions in LO_user/extract/box
lon0, lon1 = -123.29, -122.1
lat0, lat1 = 46.95, 48.93
# note: triangular elements selected by their centroids, then includes all nodes belonging to 
# those elements, so the actual extent of the extracted data is slightly larger than the box defined above.

ssm_base_url = f"https://s3.kopah.uw.edu/ssm/wqm/{year}/"

# ============================================================
# CONFIGURATION
# ============================================================

output_dir = (
    f"/dat2/tbottger/SSM_pnnl/"
    f"ssm_box/{year}"
)
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    f"SSM_{year}_box_subset.nc",
)

ssm_fs = fsspec.filesystem(
    "http",
    block_size=8 * 1024 * 1024,
)

# Use your existing ssm_base_url
grid_url = ssm_base_url + "ssm_FVCOMICM_00001.nc"

# ============================================================
# VARIABLES TO EXTRACT
# ============================================================

variables_to_extract = [
    "salinity",
    "temp",
    "CCHL1",
    "B1",
    "B2",
    "DOXG",
    "NH4",
    "NO3",
    "PO4",
    "TDIC",
    "TALK",
    "pH",
    "pCO2",
    "zeta",
    "depth",
]

# ============================================================
# READ REMOTE GRID METADATA
# ============================================================

with ssm_fs.open(grid_url, "rb") as remote_file:
    nc = netcdf_file(
        remote_file,
        mode="r",
        mmap=False,
    )

    try:
        x = np.asarray(nc.variables["x"][:]).copy()
        y = np.asarray(nc.variables["y"][:]).copy()
        h = np.asarray(nc.variables["h"][:]).copy()  # bathymetry (m)

        # FVCOM connectivity: typically (three, nele),
        # with node indices starting at 1.
        nv = np.asarray(nc.variables["nv"][:]).copy()

        # Identify connectivity orientation.
        if nv.shape[0] == 3:
            nv = nv.T
        elif nv.shape[1] != 3:
            raise ValueError(
                f"Unexpected connectivity shape: {nv.shape}"
            )

        # Convert UTM Zone 10N coordinates to lon/lat.
        transformer = Transformer.from_crs(
            "EPSG:26910",
            "EPSG:4326",
            always_xy=True,
        )

        lon, lat = transformer.transform(x, y)

        # Identify nodes inside the requested box.
        node_inside = (
            (lon >= lon0) & (lon <= lon1) &
            (lat >= lat0) & (lat <= lat1)
        )

        # FVCOM connectivity is normally 1-based.
        nv0 = nv.astype(np.int64) - 1

        # Select elements whose centroids fall in the box.
        elem_lon = lon[nv0].mean(axis=1)
        elem_lat = lat[nv0].mean(axis=1)

        elem_inside = (
            (elem_lon >= lon0) & (elem_lon <= lon1) &
            (elem_lat >= lat0) & (elem_lat <= lat1)
        )

        elem_indices = np.flatnonzero(elem_inside)

        # Include all nodes required by selected elements.
        node_indices = np.unique(nv0[elem_inside].ravel())

        # Remap original node indices to local subset indices.
        old_to_new = np.full(len(x), -1, dtype=np.int64)
        old_to_new[node_indices] = np.arange(len(node_indices))

        nv_subset = old_to_new[nv0[elem_indices]]

        print(f"Nodes inside geographic box: {node_inside.sum():,}")
        print(f"Nodes required by selected elements: {len(node_indices):,}")
        print(f"Selected elements: {len(elem_indices):,}")

    finally:
        nc.close()
        
# ============================================================
# INSPECT REMOTE FILE STRUCTURE
# ============================================================

# with ssm_fs.open(grid_url, "rb") as remote_file:
#     nc = netcdf_file(remote_file, mode="r", mmap=False)

#     try:
#         print("Dimensions:")
#         for name, dim in nc.dimensions.items():
#             print(f"  {name}: {dim}")

#         print("\nVariables:")
#         for name, var in nc.variables.items():
#             print(
#                 f"  {name:20s} "
#                 f"dimensions={var.dimensions}, "
#                 f"shape={var.shape}, "
#                 f"dtype={var.data.dtype}"
#             )

#     finally:
#         nc.close()
        
# ============================================================
# SAVE STATIC MESH INFORMATION
# ============================================================

mesh_ds = xr.Dataset(
    coords={
        "local_node": np.arange(len(node_indices)),
        "elem": np.arange(len(elem_indices)),
        "three": np.arange(3),
    },
    data_vars={
        "lon": ("local_node", lon[node_indices]),
        "lat": ("local_node", lat[node_indices]),
        "h": ("local_node", h[node_indices]),
        "nv": (("elem", "three"), nv_subset.astype(np.int32)),
        "original_node_index": (
            "local_node", node_indices.astype(np.int32)
        ),
        "original_elem_index": (
            "elem", elem_indices.astype(np.int32)
        ),
    },
    attrs={
        "description": "SSM FVCOM regional box extraction",
        "year": year,
        "bbox": f"{lon0}, {lon1}, {lat0}, {lat1}",
    },
)

# ============================================================
# LOOP THROUGH DAILY FILES, DAILY AVERAGE
# ============================================================

daily_datasets = []

# testing for a few days, change to range(1, 366) for full year
for day in range(1, 3):

    filename = f"ssm_FVCOMICM_{day:05d}.nc"
    file_url = ssm_base_url + filename

    print(f"Processing {day}/365: {filename}")

    with ssm_fs.open(file_url, "rb") as remote_file:
        nc = netcdf_file(remote_file, mode="r", mmap=False)

        try:
            time = np.asarray(nc.variables["time"][:]).copy()
            
            time_var = nc.variables["time"]
            time = np.asarray(time_var[:]).copy()

            if day == 1:
                print("Time units:", getattr(time_var, "units", "not specified"))
                print("First day's time values:", time[:5])
                print("First day's last time:", time[-1])

            data_vars = {}

            for name in variables_to_extract:

                if name not in nc.variables:
                    print(f"  Skipping missing variable: {name}")
                    continue

                var = nc.variables[name]
                dims = var.dimensions

                if "node" not in dims:
                    continue

                # Read variable and subset the selected nodes.
                arr = np.asarray(var[:]).copy()

                node_axis = dims.index("node")
                arr = np.take(arr, node_indices, axis=node_axis)

                # Average across the 24 hourly records.
                time_axis = dims.index("time")
                arr_daily = np.nanmean(arr, axis=time_axis)

                # Remove the time dimension after averaging.
                out_dims = tuple(
                    "local_node" if d == "node" else d
                    for d in dims
                    if d != "time"
                )

                data_vars[name] = (out_dims, arr_daily)

            # Use midnight to label each daily mean.
            day_timestamp = np.datetime64(
                pd.Timestamp(year, 1, 1) + pd.Timedelta(days=day - 1),
                "ns",
            )

            coords = {
                "time": [day_timestamp],
                "local_node": np.arange(len(node_indices)),
            }

            if "siglay" in nc.variables:
                coords["siglay"] = np.asarray(
                    nc.variables["siglay"][:]
                ).copy()

            if "siglev" in nc.variables:
                coords["siglev"] = np.asarray(
                    nc.variables["siglev"][:]
                ).copy()

            # Add a single time dimension to every daily-mean variable.
            daily_ds = xr.Dataset(
                data_vars={
                    name: (("time",) + dims, np.expand_dims(values, axis=0))
                    for name, (dims, values) in data_vars.items()
                },
                coords=coords,
            )

            daily_datasets.append(daily_ds)

        finally:
            nc.close()

# ============================================================
# COMBINE DAILY DATA
# ============================================================

annual_ds = xr.concat(
    daily_datasets,
    dim="time",
)

# Add static mesh fields.
annual_ds = xr.merge(
    [annual_ds, mesh_ds],
    compat="override",
)

    print("Output dimensions:", dict(annual_ds.sizes))
    print("Time length:", annual_ds.sizes["time"])
    print("Variables:", list(annual_ds.data_vars))
    print("DOXG shape:", annual_ds["DOXG"].shape)

annual_ds.attrs.update(mesh_ds.attrs)

annual_ds.to_netcdf(output_file)

print("\nExtraction complete.")
print(f"Output: {output_file}")
print(annual_ds)