import numpy as np
import pandas as pd
import xarray as xr
import fsspec

from scipy.io import netcdf_file
from scipy.spatial import cKDTree
from pyproj import Transformer

from lo_tools import Lfun


# ============================================================
# USER SETTINGS
# ============================================================

Ldir = Lfun.Lstart()

year = 2014
station = "SAR003"

target_lon = -122.49000
target_lat = 48.10833

ssm_base_url = f"https://s3.kopah.uw.edu/ssm/wqm/{year}/"

out_fn = (
    Ldir["LOo"]
    / "stations"
    / f"{station}_{year}.01.01_{year}.12.31_ssm.nc"
)

print("=" * 70)
print("SSM EXTRACTION")
print("=" * 70)
print(f"Station:       {station}")
print(f"Target lon:    {target_lon}")
print(f"Target lat:    {target_lat}")
print(f"Output:        {out_fn}")
print()


# ============================================================
# HTTP FILESYSTEM
#
# Reuse ONE filesystem for the entire extraction.
# No local downloads or persistent local files.
# ============================================================

ssm_fs = fsspec.filesystem(
    "http",
    block_size=8 * 1024 * 1024,
    cache_type="readahead",
)


# ============================================================
# FUNCTION: READ ONLY ONE NODE
# ============================================================

def read_variable_node(nc, varname, node_index):
    """
    Read only the requested node from an SSM variable.

    SSM variables are expected to have dimensions such as:

        profile: (time, siglay, node)
        point:   (time, node)
        1-D:     (siglay), (siglev), etc.

    Only the requested node is read for variables containing
    a node dimension.
    """

    var = nc.variables[varname]
    dims = var.dimensions

    if "node" in dims:

        node_axis = dims.index("node")

        # 3-D profile variable:
        # (time, siglay, node)
        if len(dims) == 3:

            if node_axis == 2:
                return np.asarray(
                    var[:, :, node_index]
                )

            elif node_axis == 1:
                return np.asarray(
                    var[:, node_index, :]
                )

            elif node_axis == 0:
                return np.asarray(
                    var[node_index, :, :]
                )

        # 2-D point variable:
        # (time, node)
        elif len(dims) == 2:

            if node_axis == 1:
                return np.asarray(
                    var[:, node_index]
                )

            elif node_axis == 0:
                return np.asarray(
                    var[node_index, :]
                )

        else:
            raise ValueError(
                f"Unexpected node-containing dimensions "
                f"for {varname}: {dims}"
            )

    # Variables without a node dimension
    return np.asarray(var[:])


# ============================================================
# FUNCTION: OPEN DAILY SSM FILE
# ============================================================

def open_ssm_daily_file(day_of_year):
    """
    Open one remote SSM daily NetCDF file.

    Nothing is downloaded to a local file.
    """

    filename = f"ssm_FVCOMICM_{day_of_year:05d}.nc"

    url = ssm_base_url + filename

    # fsspec HTTPFile provides remote byte-range access
    # rather than creating a local copy.
    f = ssm_fs.open(url, mode="rb")

    nc = netcdf_file(
        f,
        mode="r",
        mmap=False,
    )

    return nc, f


# ============================================================
# GET STATIC GRID INFORMATION
#
# Read the grid once to determine the nearest node.
# ============================================================

print("Reading SSM grid information...")

first_file = (
    ssm_base_url
    + "ssm_FVCOMICM_00001.nc"
)

f0 = ssm_fs.open(
    first_file,
    mode="rb"
)

nc0 = netcdf_file(
    f0,
    mode="r",
    mmap=False,
)


# ------------------------------------------------------------
# Node coordinates
# ------------------------------------------------------------

x = np.asarray(
    nc0.variables["x"][:]
)

y = np.asarray(
    nc0.variables["y"][:]
)

print(f"Number of SSM nodes: {len(x):,}")


# ------------------------------------------------------------
# Convert UTM Zone 10N -> longitude/latitude
# ------------------------------------------------------------

transformer = Transformer.from_crs(
    "EPSG:26910",
    "EPSG:4326",
    always_xy=True,
)

lon, lat = transformer.transform(
    x,
    y,
)


# ------------------------------------------------------------
# Find nearest SSM node
# ------------------------------------------------------------

tree = cKDTree(
    np.column_stack(
        [lon, lat]
    )
)

distance_deg, nearest_node = tree.query(
    [target_lon, target_lat]
)

print()
print("Nearest SSM node:")
print(f"    Node index: {nearest_node}")
print(f"    SSM lon:    {lon[nearest_node]:.5f}")
print(f"    SSM lat:    {lat[nearest_node]:.5f}")
print(f"    Distance:   {distance_deg:.5f} degrees")
print()


# ============================================================
# STATIC VERTICAL GRID
# ============================================================

siglay = np.asarray(
    nc0.variables["siglay"][:]
)

siglev = np.asarray(
    nc0.variables["siglev"][:]
)

h = float(
    np.asarray(
        nc0.variables["h"][:]
    )[nearest_node]
)

print(f"Number of sigma layers: {len(siglay)}")
print(f"Bathymetry at node:     {h:.3f} m")
print()

print("Initial sigma-layer depths:")
for ii, sigma in enumerate(siglay):
    depth = -sigma * h
    print(
        f"    Layer {ii:2d}: "
        f"siglay = {sigma: .6f}, "
        f"depth = {depth:8.3f} m"
    )

print()


# Close first file
nc0.close()
f0.close()


# ============================================================
# VARIABLES TO EXTRACT
#
# Only these variables will be accessed.
# ============================================================

profile_variables = [
    "salinity",
    "temp",
    "DOXG",
    "NH4",
    "NO3",
    "PO4",
    "TDIC",
    "TALK",
    "pH",
    "pCO2",
]

point_variables = [
    "zeta",
    "depth",
]


# ============================================================
# STORAGE
# ============================================================

dates = []

daily_data = {
    varname: []
    for varname in profile_variables
}

daily_point_data = {
    varname: []
    for varname in point_variables
}

daily_sigma_depth = []


# ============================================================
# LOOP THROUGH DAYS
# ============================================================

start_date = pd.Timestamp(
    f"{year}-01-01"
)

n_days = (
    366
    if pd.Timestamp(f"{year}-12-31").dayofyear == 366
    else 365
)

print("=" * 70)
print(f"Processing {n_days} daily SSM files")
print("=" * 70)


for day_of_year in range(
    1,
    n_days + 1
):

    date = (
        start_date
        + pd.Timedelta(
            days=day_of_year - 1
        )
    )

    print(
        f"[{day_of_year:3d}/{n_days}] "
        f"{date.strftime('%Y-%m-%d')}",
        end=" ... ",
        flush=True,
    )

    nc = None
    f = None

    try:

        # ----------------------------------------------------
        # Open remote daily file
        # ----------------------------------------------------

        nc, f = open_ssm_daily_file(
            day_of_year
        )


        # ----------------------------------------------------
        # Read zeta ONLY at nearest node
        #
        # zeta dimensions:
        #     (time, node)
        # ----------------------------------------------------

        zeta_hourly = read_variable_node(
            nc,
            "zeta",
            nearest_node,
        )

        zeta_daily = np.nanmean(
            zeta_hourly
        )


        # ----------------------------------------------------
        # Physical depth of sigma layers
        #
        # FVCOM sigma relationship:
        #
        # z = zeta + siglay * (h + zeta)
        #
        # Positive depth below surface:
        #
        # depth = -z
        # ----------------------------------------------------

        sigma_depth_daily = (
            -siglay
            * (h + zeta_daily)
        )

        daily_sigma_depth.append(
            sigma_depth_daily
        )


        # ----------------------------------------------------
        # Read depth ONLY at nearest node
        #
        # This is a point variable with dimensions:
        #     (time, node)
        # ----------------------------------------------------

        depth_hourly = read_variable_node(
            nc,
            "depth",
            nearest_node,
        )

        depth_daily = np.nanmean(
            depth_hourly
        )

        daily_point_data["depth"].append(
            depth_daily
        )


        # ----------------------------------------------------
        # Save daily zeta
        # ----------------------------------------------------

        daily_point_data["zeta"].append(
            zeta_daily
        )


        # ----------------------------------------------------
        # Read profile variables ONLY at nearest node
        #
        # Each returned array is:
        #
        #     (time, siglay)
        #
        # rather than:
        #
        #     (time, siglay, node)
        #
        # ----------------------------------------------------

        for varname in profile_variables:

            values = read_variable_node(
                nc,
                varname,
                nearest_node,
            )

            # ------------------------------------------------
            # Average the 24 hourly values
            # independently for each sigma layer
            # ------------------------------------------------

            daily_mean = np.nanmean(
                values,
                axis=0,
            )

            daily_data[varname].append(
                daily_mean
            )


        dates.append(
            date
        )

        print("done")


    except Exception as e:

        print(
            f"FAILED: {e}"
        )


    finally:

        # ----------------------------------------------------
        # Close remote file
        # ----------------------------------------------------

        if nc is not None:
            nc.close()

        if f is not None:
            f.close()


# ============================================================
# CONVERT LISTS TO ARRAYS
# ============================================================

print()
print("=" * 70)
print("Building output dataset")
print("=" * 70)

dates = pd.DatetimeIndex(
    dates
)

sigma_depth = np.asarray(
    daily_sigma_depth
)

print(
    f"Number of daily records: {len(dates)}"
)

print(
    f"Sigma-depth shape:       {sigma_depth.shape}"
)


# ============================================================
# CREATE XARRAY DATASET
# ============================================================

ds_out = xr.Dataset(
    coords={
        "time": dates,
        "siglay": siglay,
        "siglev": siglev,
    }
)


# ============================================================
# PROFILE VARIABLES
# ============================================================

for varname in profile_variables:

    values = np.asarray(
        daily_data[varname]
    )

    ds_out[varname] = xr.DataArray(
        values,
        dims=("time", "siglay"),
    )


# ============================================================
# POINT VARIABLES
# ============================================================

for varname in point_variables:

    values = np.asarray(
        daily_point_data[varname]
    )

    ds_out[varname] = xr.DataArray(
        values,
        dims=("time",),
    )


# ============================================================
# PHYSICAL SIGMA-LAYER DEPTH
# ============================================================

ds_out["sigma_depth"] = xr.DataArray(
    sigma_depth,
    dims=("time", "siglay"),
)


# ============================================================
# STATIC LOCATION INFORMATION
# ============================================================

ds_out["x"] = xr.DataArray(
    x[nearest_node]
)

ds_out["y"] = xr.DataArray(
    y[nearest_node]
)

ds_out["lon"] = xr.DataArray(
    lon[nearest_node]
)

ds_out["lat"] = xr.DataArray(
    lat[nearest_node]
)

ds_out["h"] = xr.DataArray(
    h
)


# ============================================================
# ATTRIBUTES
# ============================================================

ds_out.attrs[
    "station"
] = station

ds_out.attrs[
    "target_longitude"
] = target_lon

ds_out.attrs[
    "target_latitude"
] = target_lat

ds_out.attrs[
    "ssm_node"
] = int(nearest_node)


ds_out["sigma_depth"].attrs[
    "long_name"
] = (
    "physical depth of sigma layer below surface"
)

ds_out["sigma_depth"].attrs[
    "units"
] = "m"


ds_out["x"].attrs[
    "units"
] = "m"

ds_out["y"].attrs[
    "units"
] = "m"

ds_out["lon"].attrs[
    "units"
] = "degrees_east"

ds_out["lat"].attrs[
    "units"
] = "degrees_north"

ds_out["h"].attrs[
    "units"
] = "m"


# ============================================================
# COPY VARIABLE ATTRIBUTES FROM FIRST FILE
#
# Open day 1 again only to obtain metadata.
# ============================================================

print("Reading variable metadata...")

fmeta = ssm_fs.open(
    ssm_base_url
    + "ssm_FVCOMICM_00001.nc",
    mode="rb"
)

ncmeta = netcdf_file(
    fmeta,
    mode="r",
    mmap=False,
)

for varname in (
    profile_variables
    + point_variables
):

    if varname in ncmeta.variables:

        var = ncmeta.variables[varname]

        # scipy netcdf attributes
        for attr_name in dir(var):

            if attr_name.startswith("_"):
                continue

            if attr_name in [
                "dimensions",
                "shape",
                "data",
                "mask",
                "assignValue",
                "getValue",
                "typecode",
                "isrec",
                "itemsize",
                "size",
                "ndim",
            ]:
                continue

            try:
                attr_value = getattr(
                    var,
                    attr_name,
                )

                if isinstance(
                    attr_value,
                    (
                        str,
                        int,
                        float,
                    )
                ):
                    ds_out[varname].attrs[
                        attr_name
                    ] = attr_value

            except Exception:
                pass


ncmeta.close()
fmeta.close()


# ============================================================
# REMOVE ANY DUPLICATE TIMES
# ============================================================

_, unique_indices = np.unique(
    ds_out.time.values,
    return_index=True,
)

unique_indices = np.sort(
    unique_indices
)

ds_out = ds_out.isel(
    time=unique_indices
)

ds_out = ds_out.sortby(
    "time"
)


# ============================================================
# SAVE
# ============================================================

print()
print("Saving:")
print(out_fn)

ds_out.to_netcdf(
    out_fn,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Output dimensions:"
)
print(
    ds_out.dims
)

print()
print(
    f"Time records:   {ds_out.sizes['time']}"
)

print(
    f"Sigma layers:   {ds_out.sizes['siglay']}"
)

print(
    f"SSM node:        {nearest_node}"
)

print(
    f"Node longitude:  {lon[nearest_node]:.5f}"
)

print(
    f"Node latitude:   {lat[nearest_node]:.5f}"
)

print(
    f"Bathymetry:      {h:.3f} m"
)

print()
print(
    f"Saved to:\n{out_fn}"
)