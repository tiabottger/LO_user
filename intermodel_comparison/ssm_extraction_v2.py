# ============================================================
# Optimized SSM extraction
#
# Adds SSM values to an existing LO + SSC observation pickle.
#
# Strategy:
#   1. Load existing observation targets
#   2. Build SSM nearest-node KDTree
#   3. Group observations by calendar date
#   4. Open each daily SSM file ONCE
#   5. Extract all observations from that day
#   6. Return SSM dataframe with exactly the same row order
# ============================================================

'''
SSM WQM extraction at observation locations/times/depths

Uses:
  data = combined_bottle_2014_cas7_t1_x11ab_ssc.pkl 
  data["obs"] as the master observation targets

SSM data:
  https://s3.kopah.uw.edu/ssm/wqm/
  
SSM grid:
    EPSG:26910 (NAD83 / UTM Zone 10N)

No complete daily files are downloaded. fsspec + scipy
access the NetCDF-3 files remotely using HTTP range requests.
'''

import pickle
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from scipy.io import netcdf_file
from scipy.spatial import cKDTree
import pyproj
import gsw

from lo_tools import Lfun

import sys
from pathlib import Path

Ldir = Lfun.Lstart()

# ============================================================
# PATHS
# ============================================================

year = 2014

pkl_path = (
    Ldir['LOo'] /'obsmod' /
        f"combined_bottle_{year}_cas7_t1_x11ab_ssc.pkl"
)

out_pkl_path = (
    Ldir['LOo'] /'obsmod' /
            f"combined_bottle_{year}_cas7_t1_x11ab_ssc_ssmpnnl.pkl"
            )

ssm_base_url = "https://s3.kopah.uw.edu/ssm/wqm/2014/"


# ============================================================
# LOAD EXISTING PICKLE
# ============================================================

print("\nLoading existing pickle...")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

obs = data["obs"].copy()

print(f"Number of observations: {len(obs)}")

# ------------------------------------------------------------
# Make sure time is datetime
# ------------------------------------------------------------

obs["time"] = pd.to_datetime(obs["time"])

# ------------------------------------------------------------
# Preserve ORIGINAL observation row number
#
# This is the key to guaranteeing that the final SSM table
# has exactly the same rows and order as obs.
# ------------------------------------------------------------

obs["_obs_index"] = np.arange(len(obs))

print("Original observation index created.")


# ============================================================
# CHECK OBSERVATION DATE RANGE
# ============================================================

print("\nObservation date range:")
print(f"  First: {obs['time'].min()}")
print(f"  Last:  {obs['time'].max()}")

unique_dates = (
    obs["time"]
    .dt.normalize()
    .drop_duplicates()
    .sort_values()
)

print(f"Number of unique observation dates: {len(unique_dates)}")


# ============================================================
# OPEN FIRST SSM FILE TO GET GRID
#
# We use Jan 1 as the grid source.
# The SSM horizontal grid is static.
# ============================================================

print("\nOpening SSM grid...")

grid_url = (
    ssm_base_url +
    "ssm_FVCOMICM_00001.nc"
)

grid_fs = fsspec.filesystem(
    "http",
    block_size=1024 * 1024,
    cache_type="readahead"
)

grid_file = grid_fs.open(
    grid_url,
    mode="rb"
)

grid_nc = netcdf_file(
    grid_file,
    mode="r",
    mmap=False
)

# ------------------------------------------------------------
# Get x/y
# ------------------------------------------------------------

x = np.array(grid_nc.variables["x"][:]).astype(float)
y = np.array(grid_nc.variables["y"][:]).astype(float)

print(f"SSM number of nodes: {len(x)}")

# ------------------------------------------------------------
# Convert UTM Zone 10N -> lon/lat
#
# SSM x/y:
#   EPSG:26910 = NAD83 / UTM Zone 10N
# ------------------------------------------------------------

transformer = pyproj.Transformer.from_crs(
    "epsg:26910",
    "epsg:4326",
    always_xy=True
)

lon_ssm, lat_ssm = transformer.transform(x, y)

lon_ssm = np.asarray(lon_ssm)
lat_ssm = np.asarray(lat_ssm)

print(
    f"SSM longitude range: "
    f"{np.nanmin(lon_ssm):.3f} to {np.nanmax(lon_ssm):.3f}"
)

print(
    f"SSM latitude range: "
    f"{np.nanmin(lat_ssm):.3f} to {np.nanmax(lat_ssm):.3f}"
)


# ============================================================
# BUILD KD-TREE
# ============================================================

print("\nBuilding SSM KDTree...")

ssm_xy = np.column_stack([
    lon_ssm,
    lat_ssm
])

tree = cKDTree(ssm_xy)


# ============================================================
# FIND NEAREST SSM NODE FOR EVERY OBSERVATION
# ============================================================

print("\nFinding nearest SSM node for every observation...")

obs_xy = np.column_stack([
    obs["lon"].astype(float).values,
    obs["lat"].astype(float).values
])

horizontal_distance_deg, nearest_node = tree.query(
    obs_xy
)

nearest_node = nearest_node.astype(int)

# Approximate horizontal distance in km
horizontal_distance_km = (
    horizontal_distance_deg * 111.0
)

print(
    f"Mean nearest-node distance: "
    f"{np.nanmean(horizontal_distance_km):.3f} km"
)

print(
    f"Maximum nearest-node distance: "
    f"{np.nanmax(horizontal_distance_km):.3f} km"
)


# ============================================================
# STORE NEAREST NODE INFORMATION
# ============================================================

obs["_ssm_node"] = nearest_node

obs["_ssm_lon"] = lon_ssm[nearest_node]
obs["_ssm_lat"] = lat_ssm[nearest_node]

obs["_ssm_horizontal_distance_km"] = horizontal_distance_km


# ============================================================
# SSM SIGMA LAYERS
# ============================================================

siglay = np.array(
    grid_nc.variables["siglay"][:]
).astype(float)

print("\nSSM sigma layers:")
print(siglay)

print(f"Number of sigma layers: {len(siglay)}")


# ------------------------------------------------------------
# Close grid file
# ------------------------------------------------------------

grid_nc.close()
grid_file.close()


# ============================================================
# FUNCTION: OPEN DAILY SSM FILE
# ============================================================

def open_ssm_daily_file(file_date):
    """
    Open one SSM daily NetCDF file remotely.

    Parameters
    ----------
    file_date : pandas.Timestamp

    Returns
    -------
    nc : scipy.io.netcdf_file
    file_obj : remote file handle
    """

    day_of_year = file_date.dayofyear

    filename = (
        f"ssm_FVCOMICM_{day_of_year:05d}.nc"
    )

    url = ssm_base_url + filename

    print(f"\nOpening SSM file:")
    print(f"  {filename}")

    fs = fsspec.filesystem(
        "http",
        block_size=1024 * 1024,
        cache_type="readahead"
    )

    file_obj = fs.open(
        url,
        mode="rb"
    )

    nc = netcdf_file(
        file_obj,
        mode="r",
        mmap=False
    )

    return nc, file_obj


# ============================================================
# FUNCTION: GET SSM HOURLY TIMES
# ============================================================

def get_ssm_datetimes(file_date, n_time):
    """
    Construct SSM timestamps assuming the first record is
    00:00:00 on the file date and subsequent records occur
    every hour.
    """

    start_time = (
        file_date.normalize()
    )

    return pd.date_range(
        start=start_time,
        periods=n_time,
        freq="1h"
    )

# ============================================================
# FUNCTION: EXTRACT ONE DAY
# ============================================================

def extract_ssm_day(day_obs, file_date, siglay):
    """
    Extract SSM values for all observations on one date.

    Every row in day_obs produces exactly one row in the
    returned DataFrame.
    """

    # --------------------------------------------------------
    # Number of observations for this date
    # --------------------------------------------------------

    n_obs = len(day_obs)

    print("\n" + "-" * 60)
    print(
        f"Processing {file_date.date()} "
        f"({n_obs} observations)"
    )
    print("-" * 60)

    if n_obs == 0:
        return None

    # --------------------------------------------------------
    # Open daily file ONCE
    # --------------------------------------------------------

    nc, file_obj = open_ssm_daily_file(file_date)

    try:

        # ====================================================
        # TIME
        # ====================================================

        time_values = np.array(
            nc.variables["time"][:]
        ).astype(float)

        n_time = len(time_values)

        ssm_datetimes = get_ssm_datetimes(
            file_date,
            n_time
        )

        # ----------------------------------------------------
        # Match every observation time to nearest SSM hour
        # ----------------------------------------------------

        obs_times = pd.to_datetime(
            day_obs["time"]
        )

        time_idx = np.array([
            int(
                np.argmin(
                    np.abs(
                        ssm_datetimes - t
                    )
                )
            )
            for t in obs_times
        ])

        matched_ssm_times = (
            ssm_datetimes[time_idx]
        )

        # ----------------------------------------------------
        # Time difference
        # ----------------------------------------------------

        time_difference = (
            matched_ssm_times
            - obs_times.values
        )

        time_difference_minutes = (
            np.abs(
                time_difference
                / np.timedelta64(1, "m")
            )
        )

        print(
            f"Mean time difference: "
            f"{np.mean(time_difference_minutes):.2f} min"
        )

        print(
            f"Maximum time difference: "
            f"{np.max(time_difference_minutes):.2f} min"
        )

        # ====================================================
        # HORIZONTAL NODE
        # ====================================================

        node_idx = (
            day_obs["_ssm_node"]
            .astype(int)
            .values
        )

        # ====================================================
        # DEPTH
        #
        # SSM "depth" is total instantaneous water-column
        # depth at each node/time.
        #
        # Physical sigma-layer depth:
        #
        #     z = siglay * depth
        #
        # ====================================================

        depth_all = np.array(
            nc.variables["depth"][
                time_idx,
                node_idx
            ]
        ).astype(float)

        # ----------------------------------------------------
        # Calculate physical depth of all sigma layers
        #
        # Shape:
        #     n_obs x n_siglay
        # ----------------------------------------------------

        layer_depths = (
            depth_all[:, None]
            * siglay[None, :]
        )

        # ----------------------------------------------------
        # Observation depths
        # ----------------------------------------------------

        obs_z = (
            day_obs["z"]
            .astype(float)
            .values
        )

        # ----------------------------------------------------
        # Find nearest sigma layer for every observation
        # ----------------------------------------------------

        layer_idx = np.array([
            int(
                np.argmin(
                    np.abs(
                        layer_depths[i, :]
                        - obs_z[i]
                    )
                )
            )
            for i in range(n_obs)
        ])

        ssm_depth = layer_depths[
            np.arange(n_obs),
            layer_idx
        ]

        vertical_difference = (
            ssm_depth - obs_z
        )

        print(
            f"Mean |vertical difference|: "
            f"{np.mean(np.abs(vertical_difference)):.3f} m"
        )

        print(
            f"Maximum |vertical difference|: "
            f"{np.max(np.abs(vertical_difference)):.3f} m"
        )

        # ====================================================
        # FUNCTION FOR 3-D VARIABLES
        # ====================================================

        def extract_3d(varname):

            arr = np.array(
                nc.variables[varname][
                    time_idx,
                    layer_idx,
                    node_idx
                ]
            ).astype(float)

            return arr

        # ====================================================
        # EXTRACT RAW SSM VARIABLES
        # ====================================================

        salinity = extract_3d(
            "salinity"
        )

        temp = extract_3d(
            "temp"
        )

        DOXG = extract_3d(
            "DOXG"
        )

        NO3_raw = extract_3d(
            "NO3"
        )

        NH4_raw = extract_3d(
            "NH4"
        )

        TDIC = extract_3d(
            "TDIC"
        )

        TALK = extract_3d(
            "TALK"
        )

        # ====================================================
        # UNIT CONVERSIONS
        # ====================================================

        # ----------------------------------------------------
        # Dissolved oxygen
        #
        # SSM:
        #   MG/L
        #
        # Desired:
        #   umol/L
        #
        # molecular weight O2 = 31.998 g/mol
        # ----------------------------------------------------

        DO = (
            DOXG
            * 1000.0
            / 31.998
        )

        # ----------------------------------------------------
        # NO3
        #
        # SSM:
        #   gN m-3
        #
        # Numerically:
        #   mg N L-1
        #
        # Convert to umol N L-1
        # ----------------------------------------------------

        NO3 = (
            NO3_raw
            * 1000.0
            / 14.007
        )

        # ----------------------------------------------------
        # NH4
        # ----------------------------------------------------

        NH4 = (
            NH4_raw
            * 1000.0
            / 14.007
        )

        # ----------------------------------------------------
        # DIC
        #
        # SSM:
        #   mmol C m-3
        #
        # This is numerically equal to:
        #   umol C L-1
        # ----------------------------------------------------

        DIC = TDIC

        # ----------------------------------------------------
        # TA
        #
        # SSM:
        #   mmol m-3
        #
        # Numerically equal to:
        #   umol L-1
        # ----------------------------------------------------

        TA = TALK

        # ====================================================
        # PRESSURE
        # ====================================================

        obs_lon = (
            day_obs["lon"]
            .astype(float)
            .values
        )

        obs_lat = (
            day_obs["lat"]
            .astype(float)
            .values
        )

        # SSM z is negative downward
        pressure = gsw.p_from_z(
            ssm_depth,
            obs_lat
        )

        # ====================================================
        # ABSOLUTE SALINITY
        # ====================================================

        SA = gsw.SA_from_SP(
            salinity,
            pressure,
            obs_lon,
            obs_lat
        )

        # ====================================================
        # CONSERVATIVE TEMPERATURE
        # ====================================================

        CT = gsw.CT_from_t(
            SA,
            temp,
            pressure
        )

        Chl = np.full(
            n_obs,
            np.nan
        )

        # ====================================================
        # CREATE EXACTLY ONE ROW PER OBSERVATION
        # ====================================================

        ssm_day = pd.DataFrame({

            # ------------------------------------------------
            # ORIGINAL OBSERVATION INDEX
            # ------------------------------------------------

            "_obs_index":
                day_obs["_obs_index"].values,

            # ------------------------------------------------
            # ORIGINAL OBSERVATION TARGET INFORMATION
            # ------------------------------------------------

            "cid":
                day_obs["cid"].values,

            "lon":
                day_obs["lon"].values,

            "lat":
                day_obs["lat"].values,

            "time":
                day_obs["time"].values,

            "z":
                day_obs["z"].values,

            # ------------------------------------------------
            # MODEL MATCH INFORMATION
            # ------------------------------------------------

            "ssm_time":
                matched_ssm_times,

            "ssm_z":
                ssm_depth,

            "ssm_node":
                node_idx,

            "ssm_lon":
                lon_ssm[node_idx],

            "ssm_lat":
                lat_ssm[node_idx],

            "horizontal_distance_km":
                day_obs[
                    "_ssm_horizontal_distance_km"
                ].values,

            "vertical_difference_m":
                vertical_difference,

            "time_difference_min":
                time_difference_minutes,

            # ------------------------------------------------
            # MODEL VARIABLES
            # ------------------------------------------------

            "SA":
                SA,

            "CT":
                CT,

            "DO":
                DO,

            "NO3":
                NO3,

            "NH4":
                NH4,

            "Chl":
                Chl,

            "TA":
                TA,

            "DIC":
                DIC,
        })

        # ====================================================
        # CRITICAL CHECK
        #
        # THIS MUST ALWAYS BE TRUE.
        # ====================================================

        if len(ssm_day) != n_obs:

            raise RuntimeError(
                f"\nERROR: SSM extraction produced "
                f"{len(ssm_day)} rows for "
                f"{n_obs} observations on "
                f"{file_date.date()}."
            )

        # ----------------------------------------------------
        # Check original indices are one-to-one
        # ----------------------------------------------------

        if ssm_day["_obs_index"].duplicated().any():

            duplicates = (
                ssm_day
                .loc[
                    ssm_day["_obs_index"].duplicated(),
                    "_obs_index"
                ]
                .tolist()
            )

            raise RuntimeError(
                f"\nERROR: Duplicate _obs_index values "
                f"on {file_date.date()}:\n"
                f"{duplicates[:20]}"
            )

        return ssm_day

    finally:

        # ----------------------------------------------------
        # ALWAYS close remote file
        # ----------------------------------------------------

        nc.close()
        file_obj.close()


# ============================================================
# EXTRACT ALL DAYS
# ============================================================

print("\n" + "=" * 60)
print("STARTING SSM EXTRACTION")
print("=" * 60)

ssm_list = []

total_expected = len(obs)
total_processed = 0

# ------------------------------------------------------------
# Group observations by calendar date
# ------------------------------------------------------------

grouped_obs = obs.groupby(
    obs["time"].dt.normalize()
)

for day_number, (file_date, day_obs) in enumerate(
    grouped_obs,
    start=1
):

    print(
        f"\nDAY {day_number}/{len(grouped_obs)}"
    )

    # --------------------------------------------------------
    # Extract this date
    # --------------------------------------------------------

    ssm_day = extract_ssm_day(
        day_obs,
        file_date,
        siglay
    )

    if ssm_day is None:
        continue

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    total_processed += len(ssm_day)

    print(
        f"Extracted: {len(ssm_day)} rows"
    )

    print(
        f"Cumulative: "
        f"{total_processed}/{total_expected}"
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    ssm_list.append(
        ssm_day
    )


# ============================================================
# DAILY EXTRACTION COUNTS
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)

total_extracted = 0

for i, df in enumerate(ssm_list):

    n = len(df)

    total_extracted += n

    print(
        f"Day {i + 1:3d}: "
        f"{n:4d} rows"
    )

print("-" * 60)

print(
    f"Total extracted: {total_extracted}"
)

print(
    f"Expected:        {len(obs)}"
)


# ============================================================
# COMBINE ALL SSM RESULTS
# ============================================================

if len(ssm_list) == 0:

    raise RuntimeError(
        "No SSM observations were extracted."
    )

ssm = pd.concat(
    ssm_list,
    ignore_index=True
)

print(
    f"\nSSM rows before sorting: "
    f"{len(ssm)}"
)


# ============================================================
# CRITICAL ROW COUNT CHECK
# ============================================================

if len(ssm) != len(obs):

    print("\n" + "=" * 60)
    print("FATAL ALIGNMENT ERROR")
    print("=" * 60)

    print(
        f"Expected SSM rows: {len(obs)}"
    )

    print(
        f"Actual SSM rows:   {len(ssm)}"
    )

    print(
        f"Difference:        "
        f"{len(ssm) - len(obs)}"
    )

    raise RuntimeError(
        "SSM extraction did not produce exactly "
        "one row per observation. "
        "STOPPING before reordering/saving."
    )


# ============================================================
# RESTORE EXACT ORIGINAL OBSERVATION ORDER
#
# This is NOT a merge.
#
# _obs_index uniquely identifies every original observation.
# ============================================================

ssm = (
    ssm
    .sort_values("_obs_index")
    .reset_index(drop=True)
)

print(
    f"SSM rows after sorting: "
    f"{len(ssm)}"
)


# ============================================================
# VERIFY _obs_index
# ============================================================

expected_indices = np.arange(
    len(obs)
)

actual_indices = (
    ssm["_obs_index"]
    .values
)

if not np.array_equal(
    expected_indices,
    actual_indices
):

    raise RuntimeError(
        "SSM _obs_index does not match the "
        "original observation index."
    )

print(
    "Original observation order successfully restored."
)


# ============================================================
# ALIGNMENT CHECK
# ============================================================

print("\n" + "=" * 60)
print("ALIGNMENT CHECK")
print("=" * 60)

print(
    f"Number of observations: {len(obs)}"
)

print(
    f"Number of SSM rows:     {len(ssm)}"
)


# ------------------------------------------------------------
# CID
# ------------------------------------------------------------

cid_match = np.array_equal(
    obs["cid"].values,
    ssm["cid"].values
)

print(
    f"\nCID match: {cid_match}"
)

if not cid_match:

    bad = np.where(
        obs["cid"].values
        != ssm["cid"].values
    )[0]

    print(
        f"Number of CID mismatches: "
        f"{len(bad)}"
    )

    print(
        "\nFirst 10 CID mismatches:"
    )

    for i in bad[:10]:

        print(
            f"row {i}: "
            f"obs={obs.iloc[i]['cid']} | "
            f"ssm={ssm.iloc[i]['cid']}"
        )


# ------------------------------------------------------------
# TIME
# ------------------------------------------------------------

obs_time = pd.to_datetime(
    obs["time"]
).values

ssm_time = pd.to_datetime(
    ssm["time"]
).values

time_match = np.array_equal(
    obs_time,
    ssm_time
)

print(
    f"Time match: {time_match}"
)

if not time_match:

    time_diff = (
        pd.to_datetime(
            ssm["time"]
        ).reset_index(drop=True)

        -

        pd.to_datetime(
            obs["time"]
        ).reset_index(drop=True)
    )

    bad = np.where(
        time_diff != pd.Timedelta(0)
    )[0]

    print(
        f"Number of time mismatches: "
        f"{len(bad)}"
    )

    print(
        "\nFirst 10 time mismatches:"
    )

    for i in bad[:10]:

        print(
            f"row {i}: "
            f"obs={obs.iloc[i]['time']} | "
            f"ssm={ssm.iloc[i]['time']} | "
            f"SSM match={ssm.iloc[i]['ssm_time']} | "
            f"diff={time_diff.iloc[i]}"
        )


# ------------------------------------------------------------
# DEPTH
# ------------------------------------------------------------

obs_z = (
    obs["z"]
    .astype(float)
    .values
)

ssm_z = (
    ssm["z"]
    .astype(float)
    .values
)

z_diff = (
    ssm_z - obs_z
)

print(
    f"\nMaximum |depth difference|: "
    f"{np.nanmax(np.abs(z_diff)):.6f} m"
)

print(
    f"Mean |depth difference|: "
    f"{np.nanmean(np.abs(z_diff)):.6f} m"
)

z_match = np.allclose(
    obs_z,
    ssm_z,
    equal_nan=True,
    atol=0.5
)

print(
    f"Depth match within 0.5 m: "
    f"{z_match}"
)

if not z_match:

    bad = np.where(
        ~np.isclose(
            obs_z,
            ssm_z,
            equal_nan=True,
            atol=0.5
        )
    )[0]

    print(
        f"Number of depth mismatches: "
        f"{len(bad)}"
    )

    print(
        "\nFirst 10 depth mismatches:"
    )

    for i in bad[:10]:

        print(
            f"row {i}: "
            f"obs={obs.iloc[i]['z']:.3f} | "
            f"ssm={ssm.iloc[i]['z']:.3f} | "
            f"diff={z_diff[i]:.3f}"
        )


# ------------------------------------------------------------
# HORIZONTAL DISTANCE
# ------------------------------------------------------------

horizontal_distance = (
    ssm["horizontal_distance_km"]
    .astype(float)
    .values
)

print(
    f"\nMaximum horizontal distance: "
    f"{np.nanmax(horizontal_distance):.3f} km"
)

print(
    f"Mean horizontal distance: "
    f"{np.nanmean(horizontal_distance):.3f} km"
)


# ------------------------------------------------------------
# SSM MATCHED TIME
# ------------------------------------------------------------

time_difference = (
    ssm["time_difference_min"]
    .astype(float)
    .values
)

print(
    f"\nMaximum |time difference|: "
    f"{np.nanmax(time_difference):.2f} min"
)

print(
    f"Mean |time difference|: "
    f"{np.nanmean(time_difference):.2f} min"
)


# ============================================================
# CHECK UNIQUE OBSERVATION INDICES
# ============================================================

print("\nChecking observation indices...")

index_unique = (
    ssm["_obs_index"]
    .is_unique
)

print(
    f"SSM _obs_index unique: "
    f"{index_unique}"
)

if not index_unique:

    raise RuntimeError(
        "Duplicate observation indices found in SSM."
    )


# ============================================================
# CHECK FINAL ROW-BY-ROW IDENTITY
# ============================================================

print("\nChecking row-by-row identity...")

assert len(ssm) == len(obs)

assert np.array_equal(
    obs["cid"].values,
    ssm["cid"].values
)

assert np.array_equal(
    pd.to_datetime(
        obs["time"]
    ).values,
    pd.to_datetime(
        ssm["time"]
    ).values
)

assert np.allclose(
    obs["z"].astype(float).values,
    ssm["z"].astype(float).values,
    equal_nan=True
)

print(
    "PASS: SSM rows correspond one-to-one "
    "with obs rows."
)


# ============================================================
# REMOVE INTERNAL BOOKKEEPING COLUMNS
#
# Keep useful SSM matching diagnostics, but remove the
# temporary _obs_index and the internal nearest-node columns
# that were attached to obs.
# ============================================================

ssm = ssm.drop(
    columns=[
        "_obs_index"
    ]
)


# ============================================================
# ADD SSM TO DATA DICTIONARY
# ============================================================

data["ssm"] = ssm


# ============================================================
# REMOVE TEMPORARY COLUMNS FROM OBS
# ============================================================

obs_final = obs.drop(
    columns=[
        "_obs_index",
        "_ssm_node",
        "_ssm_lon",
        "_ssm_lat",
        "_ssm_horizontal_distance_km"
    ]
)

data["obs"] = obs_final


# ============================================================
# FINAL CHECK AFTER CLEANUP
# ============================================================

print("\n" + "=" * 60)
print("FINAL CHECK")
print("=" * 60)

print(
    f"obs rows: {len(data['obs'])}"
)

print(
    f"ssm rows: {len(data['ssm'])}"
)

assert len(data["obs"]) == len(data["ssm"])

assert np.array_equal(
    data["obs"]["cid"].values,
    data["ssm"]["cid"].values
)

assert np.array_equal(
    pd.to_datetime(
        data["obs"]["time"]
    ).values,
    pd.to_datetime(
        data["ssm"]["time"]
    ).values
)

assert np.allclose(
    data["obs"]["z"].astype(float).values,
    data["ssm"]["z"].astype(float).values,
    equal_nan=True
)

print(
    "PASS: final obs and SSM tables are aligned."
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving combined pickle...")

with open(
    out_pkl_path,
    "wb"
) as f:

    pickle.dump(
        data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print(
    f"Saved to:\n{out_pkl_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SSM EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Observations: {len(data['obs'])}"
)

print(
    f"SSM rows:     {len(data['ssm'])}"
)

print(
    f"Mean horizontal distance: "
    f"{np.nanmean(data['ssm']['horizontal_distance_km']):.3f} km"
)

print(
    f"Max horizontal distance: "
    f"{np.nanmax(data['ssm']['horizontal_distance_km']):.3f} km"
)

print(
    f"Mean |vertical difference|: "
    f"{np.nanmean(np.abs(data['ssm']['vertical_difference_m'])):.3f} m"
)

print(
    f"Max |vertical difference|: "
    f"{np.nanmax(np.abs(data['ssm']['vertical_difference_m'])):.3f} m"
)

print(
    f"Mean |time difference|: "
    f"{np.nanmean(data['ssm']['time_difference_min']):.2f} min"
)

print(
    f"Max |time difference|: "
    f"{np.nanmax(data['ssm']['time_difference_min']):.2f} min"
)

print("\nSSM columns:")
print(
    list(data["ssm"].columns)
)