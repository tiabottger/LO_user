import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import fsspec

from scipy.io import netcdf_file
from scipy.spatial import cKDTree
from pyproj import Transformer

from lo_tools import Lfun

from pathlib import Path



# ============================================================
# USER SETTINGS
# ============================================================

Ldir = Lfun.Lstart()

year = 2014

stations = {
    "SAR003": {
        "lon": -122.49000,
        "lat": 48.10833,
    },

    # "ADM003": {
    #     "lon": -122.48180,
    #     "lat": 47.87917,
    # },
}

ssm_base_url = f"https://s3.kopah.uw.edu/ssm/wqm/{year}/"


# ============================================================
# OUTPUT FILES
# ============================================================

out_dir = Path(Ldir["LOo"]) / "stations"
out_dir.mkdir(parents=True, exist_ok=True)

out_files = {}

for station, info in stations.items():

    out_files[station] = (
        Ldir["LOo"]
        / "stations"
        / f"{station}_{year}.01.01_{year}.12.31_ssm.nc"
    )

# ============================================================
# HTTP FILESYSTEM
# ============================================================

ssm_fs = fsspec.filesystem(
    "http",
    block_size=8 * 1024 * 1024,
)


# ============================================================
# VARIABLES TO EXTRACT
# ============================================================

profile_variables = [
    "salinity",
    "temp",
    "CCHL1",
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
# READ ONLY ONE NODE
# ============================================================

def read_variable_node(nc, varname, node_index):
    """
    Read only the requested node from an SSM variable.

    Profile variables:
        (time, siglay, node)

    Point variables:
        (time, node)
    """

    var = nc.variables[varname]
    dims = var.dimensions

    if "node" in dims:

        node_axis = dims.index("node")

        # ----------------------------------------------------
        # 3-D profile variable
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 2-D point variable
        # ----------------------------------------------------

        elif len(dims) == 2:

            if node_axis == 1:
                return np.asarray(
                    var[:, node_index]
                )

            elif node_axis == 0:
                return np.asarray(
                    var[node_index, :]
                )

        raise ValueError(
            f"Unexpected node-containing dimensions "
            f"for {varname}: {dims}"
        )

    # Variable without node dimension
    return np.asarray(var[:])


# ============================================================
# DOWNLOAD ONE DAILY FILE TEMPORARILY
# ============================================================

def download_daily_file(day_of_year):

    filename = (
        f"ssm_FVCOMICM_{day_of_year:05d}.nc"
    )

    url = ssm_base_url + filename

    tmp = tempfile.NamedTemporaryFile(
        suffix=".nc",
        delete=False,
    )

    tmp_path = tmp.name
    tmp.close()

    print(
        f"Downloading {filename}...",
        end=" ",
        flush=True,
    )

    try:

        with ssm_fs.open(
            url,
            mode="rb",
        ) as src:

            with open(
                tmp_path,
                "wb",
            ) as dst:

                while True:

                    chunk = src.read(
                        16 * 1024 * 1024
                    )

                    if not chunk:
                        break

                    dst.write(chunk)

        print("done")

        return tmp_path

    except Exception:

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        raise


# ============================================================
# FIND NEAREST NODE FOR EACH STATION
# ============================================================

print("=" * 70)
print("SSM GRID / STATION LOCATIONS")
print("=" * 70)

grid_tmp = download_daily_file(1)

try:

    nc0 = netcdf_file(
        grid_tmp,
        mode="r",
        mmap=False,
    )

    # --------------------------------------------------------
    # Grid coordinates
    # --------------------------------------------------------

    x = np.asarray(
        nc0.variables["x"][:]
    )

    y = np.asarray(
        nc0.variables["y"][:]
    )

    print(
        f"Number of SSM nodes: {len(x):,}"
    )

    # --------------------------------------------------------
    # UTM Zone 10N -> longitude/latitude
    # --------------------------------------------------------

    transformer = Transformer.from_crs(
        "EPSG:26910",
        "EPSG:4326",
        always_xy=True,
    )

    lon, lat = transformer.transform(
        x,
        y,
    )

    # --------------------------------------------------------
    # KDTree
    # --------------------------------------------------------

    tree = cKDTree(
        np.column_stack(
            [lon, lat]
        )
    )

    # --------------------------------------------------------
    # Sigma coordinates
    # --------------------------------------------------------

    siglay = np.asarray(
        nc0.variables["siglay"][:]
    )

    siglev = np.asarray(
        nc0.variables["siglev"][:]
    )

    # --------------------------------------------------------
    # Bathymetry
    # --------------------------------------------------------

    h_all = np.asarray(
        nc0.variables["h"][:]
    )

    # --------------------------------------------------------
    # Find nearest node for every station
    # --------------------------------------------------------

    station_info = {}

    for station, info in stations.items():

        distance_deg, nearest_node = tree.query(
            [
                info["lon"],
                info["lat"],
            ]
        )

        nearest_node = int(
            nearest_node
        )

        h = float(
            h_all[nearest_node]
        )

        station_info[station] = {
            "node": nearest_node,
            "lon": float(
                lon[nearest_node]
            ),
            "lat": float(
                lat[nearest_node]
            ),
            "distance_deg": float(
                distance_deg
            ),
            "h": h,
        }

        print()
        print(
            f"{station}:"
        )
        print(
            f"    Target:       "
            f"{info['lon']:.5f}, "
            f"{info['lat']:.5f}"
        )
        print(
            f"    Node:         "
            f"{nearest_node}"
        )
        print(
            f"    Model:        "
            f"{lon[nearest_node]:.5f}, "
            f"{lat[nearest_node]:.5f}"
        )
        print(
            f"    Distance:     "
            f"{distance_deg:.5f} degrees"
        )
        print(
            f"    Bathymetry:   "
            f"{h:.3f} m"
        )

    nc0.close()

finally:

    if os.path.exists(grid_tmp):
        os.remove(grid_tmp)


# ============================================================
# STORAGE FOR EACH STATION
# ============================================================

station_storage = {}

for station in stations:

    station_storage[station] = {
        "dates": [],

        "profile": {
            varname: []
            for varname in profile_variables
        },

        "point": {
            varname: []
            for varname in point_variables
        },

        "sigma_depth": [],
    }


# ============================================================
# LOOP THROUGH DAYS
#
# IMPORTANT:
# One download serves BOTH stations.
# ============================================================

start_date = pd.Timestamp(
    f"{year}-01-01"
)

n_days = (
    366
    if pd.Timestamp(
        f"{year}-12-31"
    ).dayofyear == 366
    else 365
)


print()
print("=" * 70)
print(
    f"Processing {n_days} daily SSM files"
)
print(
    f"Stations: {', '.join(stations.keys())}"
)
print("=" * 70)


for day_of_year in range(
    1,
    n_days + 1,
):

    date = (
        start_date
        + pd.Timedelta(
            days=day_of_year - 1
        )
    )

    print()
    print(
        f"[{day_of_year:3d}/{n_days}] "
        f"{date.strftime('%Y-%m-%d')}"
    )

    tmp_path = None
    nc = None

    try:

        # ====================================================
        # DOWNLOAD ONE DAILY FILE
        # ====================================================

        tmp_path = download_daily_file(
            day_of_year
        )

        # ====================================================
        # OPEN LOCAL TEMPORARY FILE
        # ====================================================

        nc = netcdf_file(
            tmp_path,
            mode="r",
            mmap=False,
        )

        # ====================================================
        # EXTRACT BOTH STATIONS
        # ====================================================

        for station in stations:

            node = station_info[
                station
            ]["node"]

            h = station_info[
                station
            ]["h"]

            storage = station_storage[
                station
            ]

            # ------------------------------------------------
            # ZETA
            # ------------------------------------------------

            zeta_hourly = read_variable_node(
                nc,
                "zeta",
                node,
            )

            zeta_daily = np.nanmean(
                zeta_hourly
            )

            storage["point"][
                "zeta"
            ].append(
                zeta_daily
            )

            # ------------------------------------------------
            # PHYSICAL DEPTH OF ALL SIGMA LAYERS
            # ------------------------------------------------

            sigma_depth_daily = (
                -siglay
                * (
                    h
                    + zeta_daily
                )
            )

            storage[
                "sigma_depth"
            ].append(
                sigma_depth_daily
            )

            # ------------------------------------------------
            # DEPTH
            # ------------------------------------------------

            depth_hourly = read_variable_node(
                nc,
                "depth",
                node,
            )

            depth_daily = np.nanmean(
                depth_hourly
            )

            storage["point"][
                "depth"
            ].append(
                depth_daily
            )

            # ------------------------------------------------
            # PROFILE VARIABLES
            # ------------------------------------------------

            for varname in profile_variables:

                values = read_variable_node(
                    nc,
                    varname,
                    node,
                )

                # 24 hourly values -> 1 daily
                # mean for each of 10 sigma layers
                daily_mean = np.nanmean(
                    values,
                    axis=0,
                )

                storage["profile"][
                    varname
                ].append(
                    daily_mean
                )

            storage[
                "dates"
            ].append(
                date
            )

            print(
                f"    {station}: extracted "
                f"node {node}"
            )

        print(
            "    Daily extraction complete."
        )

    except Exception as e:

        print(
            f"    FAILED: {e}"
        )

    finally:

        # ====================================================
        # CLOSE FILE
        # ====================================================

        if nc is not None:
            nc.close()

        # ====================================================
        # DELETE TEMPORARY FILE
        # ====================================================

        if (
            tmp_path is not None
            and os.path.exists(tmp_path)
        ):

            os.remove(
                tmp_path
            )

            print(
                "    Temporary file deleted."
            )


# ============================================================
# READ VARIABLE ATTRIBUTES
#
# One temporary download is used for metadata.
# ============================================================

print()
print("=" * 70)
print("Reading variable metadata")
print("=" * 70)

metadata_tmp = download_daily_file(1)

variable_attrs = {}

try:

    ncmeta = netcdf_file(
        metadata_tmp,
        mode="r",
        mmap=False,
    )

    for varname in (
        profile_variables
        + point_variables
    ):

        if varname not in ncmeta.variables:
            continue

        var = ncmeta.variables[
            varname
        ]

        variable_attrs[
            varname
        ] = {}

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
                    ),
                ):

                    variable_attrs[
                        varname
                    ][
                        attr_name
                    ] = attr_value

            except Exception:
                pass

    ncmeta.close()

finally:

    if os.path.exists(
        metadata_tmp
    ):
        os.remove(
            metadata_tmp
        )


# ============================================================
# BUILD AND SAVE ONE NETCDF PER STATION
# ============================================================

print()
print("=" * 70)
print("BUILDING OUTPUT FILES")
print("=" * 70)


for station in stations:

    print()
    print(
        f"Creating {station}"
    )

    info = station_info[
        station
    ]

    storage = station_storage[
        station
    ]

    dates = pd.DatetimeIndex(
        storage["dates"]
    )

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    ds_out = xr.Dataset(
        coords={
            "time": dates,
            "siglay": siglay,
            "siglev": siglev,
        }
    )

    # --------------------------------------------------------
    # Profile variables
    # --------------------------------------------------------

    for varname in profile_variables:

        values = np.asarray(
            storage["profile"][
                varname
            ]
        )

        ds_out[varname] = xr.DataArray(
            values,
            dims=(
                "time",
                "siglay",
            ),
        )

    # --------------------------------------------------------
    # Point variables
    # --------------------------------------------------------

    for varname in point_variables:

        values = np.asarray(
            storage["point"][
                varname
            ]
        )

        ds_out[varname] = xr.DataArray(
            values,
            dims=("time",),
        )

    # --------------------------------------------------------
    # Physical sigma-layer depths
    # --------------------------------------------------------

    ds_out[
        "sigma_depth"
    ] = xr.DataArray(
        np.asarray(
            storage[
                "sigma_depth"
            ]
        ),
        dims=(
            "time",
            "siglay",
        ),
    )

    # --------------------------------------------------------
    # Static station information
    # --------------------------------------------------------

    ds_out["x"] = xr.DataArray(
        x[info["node"]]
    )

    ds_out["y"] = xr.DataArray(
        y[info["node"]]
    )

    ds_out["lon"] = xr.DataArray(
        info["lon"]
    )

    ds_out["lat"] = xr.DataArray(
        info["lat"]
    )

    ds_out["h"] = xr.DataArray(
        info["h"]
    )

    # --------------------------------------------------------
    # Global attributes
    # --------------------------------------------------------

    ds_out.attrs[
        "station"
    ] = station

    ds_out.attrs[
        "target_longitude"
    ] = stations[
        station
    ]["lon"]

    ds_out.attrs[
        "target_latitude"
    ] = stations[
        station
    ]["lat"]

    ds_out.attrs[
        "ssm_node"
    ] = info["node"]

    # --------------------------------------------------------
    # sigma_depth attributes
    # --------------------------------------------------------

    ds_out[
        "sigma_depth"
    ].attrs[
        "long_name"
    ] = (
        "physical depth of sigma layer "
        "below surface"
    )

    ds_out[
        "sigma_depth"
    ].attrs[
        "units"
    ] = "m"

    # --------------------------------------------------------
    # Coordinate attributes
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Copy original SSM variable attributes
    # --------------------------------------------------------

    for varname in (
        profile_variables
        + point_variables
    ):

        for attr_name, attr_value in (
            variable_attrs[
                varname
            ].items()
        ):

            ds_out[
                varname
            ].attrs[
                attr_name
            ] = attr_value

    # --------------------------------------------------------
    # Remove duplicate times
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out_fn = out_files[station]

    print(
        f"Saving:\n{out_fn}"
    )

    # Remove an existing output file so a failed write
    # cannot leave an incomplete file behind
    if out_fn.exists():
        print(
            f"    Removing existing file: "
            f"{out_fn}"
        )
        out_fn.unlink()

    # Clean boolean attributes from ALL variables
    for var in ds_out.variables:
        attrs = ds_out[var].attrs.copy()

        for key, value in attrs.items():
            if isinstance(value, (bool, np.bool_)):
                ds_out[var].attrs[key] = int(value)

    ds_out.to_netcdf(
        out_fn
    )

    print(
        f"    Time records: "
        f"{ds_out.sizes['time']}"
    )

    print(
        f"    Sigma layers: "
        f"{ds_out.sizes['siglay']}"
    )

    print(
        f"    SSM node: "
        f"{info['node']}"
    )

    print(
        f"    Model location: "
        f"{info['lon']:.5f}, "
        f"{info['lat']:.5f}"
    )

    print(
        f"    Bathymetry: "
        f"{info['h']:.3f} m"
    )

    print(
        f"    Saved: "
        f"{out_fn.exists()}"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

for station in stations:

    info = station_info[
        station
    ]

    print()
    print(
        f"{station}"
    )

    print(
        f"    Node:       "
        f"{info['node']}"
    )

    print(
        f"    Location:   "
        f"{info['lon']:.5f}, "
        f"{info['lat']:.5f}"
    )

    print(
        f"    Bathymetry: "
        f"{info['h']:.3f} m"
    )

    print(
        f"    Output:     "
        f"{out_files[station]}"
    )