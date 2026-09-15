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

import numpy as np
import pandas as pd
import pickle
import pyproj
import fsspec
import gsw

from scipy.io import netcdf_file
from scipy.spatial import cKDTree

from lo_tools import Lfun

import sys
from pathlib import Path

Ldir = Lfun.Lstart()


# ============================================================
# USER SETTINGS
# ============================================================

year = 2014

pickle_in = (
    Ldir['LOo'] /'obsmod' /
    f"combined_bottle_{year}_cas7_t1_x11ab_ssc.pkl"
)

pickle_out = (
    Ldir['LOo'] /'obsmod' /
        f"combined_bottle_{year}_cas7_t1_x11ab_ssc_ssmpnnl.pkl"
)

ssm_base_url = (
    "https://s3.kopah.uw.edu/ssm/wqm"
)


# ============================================================
# LOAD EXISTING PICKLE
# ============================================================

print("Loading existing pickle...")

with open(pickle_in, "rb") as f:
    data = pickle.load(f)

obs = data["obs"].copy()

obs["time"] = pd.to_datetime(obs["time"])

print("\nExisting pickle keys:")
print(data.keys())

print("\nNumber of observations:", len(obs))


# ============================================================
# SET UP SSM GRID
# ============================================================

print("\nOpening SSM grid file...")

grid_url = (
    f"{ssm_base_url}/2014/"
    "ssm_FVCOMICM_00001.nc"
)

fs = fsspec.filesystem("http")

grid_file = fs.open(grid_url, "rb")

ds_grid = netcdf_file(
    grid_file,
    mode="r"
)

# ------------------------------------------------------------
# SSM x/y coordinates
# ------------------------------------------------------------

x_ssm = (
    ds_grid.variables["x"]
    .data
    .copy()
)

y_ssm = (
    ds_grid.variables["y"]
    .data
    .copy()
)

# ------------------------------------------------------------
# Convert NAD83 UTM Zone 10N -> longitude/latitude
# ------------------------------------------------------------

transformer = pyproj.Transformer.from_crs(
    "epsg:26910",
    "epsg:4326",
    always_xy=True
)

lon_ssm, lat_ssm = transformer.transform(
    x_ssm,
    y_ssm
)

# ------------------------------------------------------------
# Build nearest-node KDTree
# ------------------------------------------------------------

tree_ssm = cKDTree(
    np.column_stack(
        [
            lon_ssm,
            lat_ssm
        ]
    )
)

# ------------------------------------------------------------
# Sigma layers
# ------------------------------------------------------------

siglay = (
    ds_grid.variables["siglay"]
    .data
    .copy()
)

print("SSM nodes:", len(x_ssm))
print("SSM sigma layers:", len(siglay))

ds_grid.close()
grid_file.close()


# ============================================================
# PRECOMPUTE NEAREST SSM NODE FOR EVERY OBSERVATION
# ============================================================

print("\nFinding nearest SSM node for observations...")

obs_points = np.column_stack(
    [
        obs["lon"].values,
        obs["lat"].values
    ]
)

node_distances, node_indices = tree_ssm.query(
    obs_points
)

obs["_ssm_node"] = node_indices.astype(int)
obs["_ssm_node_distance_deg"] = node_distances


# ============================================================
# GROUP OBSERVATIONS BY CALENDAR DATE
# ============================================================

obs["_ssm_date"] = (
    obs["time"]
    .dt.normalize()
)

unique_dates = (
    obs["_ssm_date"]
    .drop_duplicates()
    .sort_values()
)

print(
    "Number of unique observation dates:",
    len(unique_dates)
)


# ============================================================
# FUNCTION TO GET DAILY SSM FILE
# ============================================================

def get_ssm_url(date):

    year = date.year
    day_of_year = date.dayofyear

    filename = (
        f"ssm_FVCOMICM_{day_of_year:05d}.nc"
    )

    return (
        f"{ssm_base_url}/{year}/{filename}"
    )


# ============================================================
# FUNCTION TO EXTRACT ONE DAILY SSM FILE
# ============================================================

def extract_one_day(
    date,
    day_obs,
    fs
):

    print(
        f"\nProcessing {date.date()} "
        f"({len(day_obs)} observations)"
    )

    url = get_ssm_url(date)

    print(
        "  Opening:",
        url
    )

    remote_file = fs.open(
        url,
        "rb"
    )

    ds = netcdf_file(
        remote_file,
        mode="r"
    )

    try:

        # ====================================================
        # READ TIME
        # ====================================================

        time_values = (
            ds.variables["time"]
            .data
            .copy()
        )

        n_times = len(time_values)

        # ----------------------------------------------------
        # SSM daily files have hourly records.
        #
        # The first record is approximately 00:59:50.
        # Subsequent records are one hour apart.
        #
        # We construct actual datetimes from the file date.
        # ----------------------------------------------------

        ssm_datetimes = pd.date_range(
            start=(
                date
                + pd.Timedelta(seconds=3590)
            ),
            periods=n_times,
            freq="1h"
        )

        # ====================================================
        # OBSERVATION TIMES -> SSM TIME INDICES
        # ====================================================

        obs_times = (
            pd.to_datetime(
                day_obs["time"]
            )
        )

        # Convert timestamps to nanoseconds
        # so we can efficiently find nearest time.
        obs_ns = (
            obs_times.astype("int64")
            .values
        )

        ssm_ns = (
            ssm_datetimes.astype("int64")
            .values
        )

        # Find nearest hourly record
        time_indices = np.array(
            [
                np.argmin(
                    np.abs(
                        ssm_ns - t
                    )
                )
                for t in obs_ns
            ],
            dtype=int
        )

        ssm_times_selected = (
            ssm_datetimes[time_indices]
        )

        time_differences_hours = (
            np.abs(
                ssm_times_selected
                - obs_times
            )
            / pd.Timedelta(hours=1)
        )

        # ====================================================
        # NODE INDICES
        # ====================================================

        node_indices_day = (
            day_obs["_ssm_node"]
            .astype(int)
            .values
        )

        # ====================================================
        # OBSERVATION DEPTHS
        # ====================================================

        obs_depths = (
            day_obs["z"]
            .astype(float)
            .values
        )

        # ====================================================
        # GET TOTAL WATER-COLUMN DEPTH
        #
        # depth is (time, node)
        # ====================================================

        depth_all = (
            ds.variables["depth"]
            .data
        )

        H = np.asarray(
            depth_all[
                time_indices,
                node_indices_day
            ],
            dtype=float
        )

        # ====================================================
        # CALCULATE PHYSICAL SSM DEPTHS
        #
        # siglay = 10 sigma-layer center coordinates
        #
        # Result:
        #     (n_observations, 10)
        # ====================================================

        layer_depths = (
            H[:, None]
            * siglay[None, :]
        )

        # ====================================================
        # FIND NEAREST SSM VERTICAL LAYER
        # ====================================================

        vertical_difference = np.abs(
            layer_depths
            - obs_depths[:, None]
        )

        layer_indices = (
            np.argmin(
                vertical_difference,
                axis=1
            )
        )

        ssm_depths = (
            layer_depths[
                np.arange(len(day_obs)),
                layer_indices
            ]
        )

        vertical_difference = (
            np.abs(
                ssm_depths
                - obs_depths
            )
        )

        # ====================================================
        # FUNCTION FOR EXTRACTING 3-D VARIABLES
        # ====================================================

        def get_3d_variable(name):

            arr = (
                ds.variables[name]
                .data
            )

            return np.asarray(
                arr[
                    time_indices,
                    layer_indices,
                    node_indices_day
                ],
                dtype=float
            )

        # ====================================================
        # EXTRACT RAW SSM VARIABLES
        # ====================================================

        DO_mgL = get_3d_variable(
            "DOXG"
        )

        temp = get_3d_variable(
            "temp"
        )

        salinity = get_3d_variable(
            "salinity"
        )

        NO3_gNm3 = get_3d_variable(
            "NO3"
        )

        NH4_gNm3 = get_3d_variable(
            "NH4"
        )

        TDIC = get_3d_variable(
            "TDIC"
        )

        TALK = get_3d_variable(
            "TALK"
        )

        # ====================================================
        # UNIT CONVERSIONS
        # ====================================================

        # ----------------------------------------------------
        # DO
        #
        # mg/L -> umol/L
        # ----------------------------------------------------

        DO = (
            DO_mgL
            * 1000
            / 31.998
        )

        # ----------------------------------------------------
        # NO3
        #
        # g N/m3 -> umol N/L
        # ----------------------------------------------------

        NO3 = (
            NO3_gNm3
            * 1000
            / 14.007
        )

        # ----------------------------------------------------
        # NH4
        #
        # g N/m3 -> umol N/L
        # ----------------------------------------------------

        NH4 = (
            NH4_gNm3
            * 1000
            / 14.007
        )

        # ----------------------------------------------------
        # DIC and TA
        #
        # mmol/m3 == umol/L
        # ----------------------------------------------------

        DIC = TDIC.copy()
        TA = TALK.copy()

        # ====================================================
        # CALCULATE SA AND CT
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

        pressure = gsw.p_from_z(
            ssm_depths,
            obs_lat
        )

        SA = gsw.SA_from_SP(
            salinity,
            pressure,
            obs_lon,
            obs_lat
        )

        CT = gsw.CT_from_t(
            SA,
            temp,
            pressure
        )

        # ====================================================
        # BUILD RESULT DATAFRAME
        # ====================================================

        result = pd.DataFrame(
            {
                # --------------------------------------------
                # Original observation identifiers
                # --------------------------------------------

                "cid": day_obs["cid"].values,

                "lon": day_obs["lon"].values,

                "lat": day_obs["lat"].values,

                "time": day_obs["time"].values,

                "z": day_obs["z"].values,

                # --------------------------------------------
                # SSM values in comparison units
                # --------------------------------------------

                "SA": SA,

                "CT": CT,

                "DO": DO,

                "NO3": NO3,

                "NH4": NH4,

                "TA": TA,

                "DIC": DIC,

                # Chlorophyll is left as NaN until we
                # identify the appropriate SSM phytoplankton
                # variable.
                "Chl": np.nan,

                # --------------------------------------------
                # SSM matching information
                # --------------------------------------------

                "ssm_node": node_indices_day,

                "ssm_lon": (
                    lon_ssm[
                        node_indices_day
                    ]
                ),

                "ssm_lat": (
                    lat_ssm[
                        node_indices_day
                    ]
                ),

                "ssm_node_distance_deg": (
                    day_obs[
                        "_ssm_node_distance_deg"
                    ].values
                ),

                "ssm_time_idx": time_indices,

                "ssm_time": (
                    ssm_times_selected
                ),

                "ssm_time_difference_hours": (
                    np.asarray(
                        time_differences_hours,
                        dtype=float
                    )
                ),

                "ssm_layer": layer_indices,

                "ssm_depth": ssm_depths,

                "ssm_vertical_difference": (
                    vertical_difference
                )
            }
        )

        return result

    finally:

        ds.close()
        remote_file.close()


# ============================================================
# LOOP THROUGH UNIQUE DAYS
# ============================================================

all_ssm_results = []

for date in unique_dates:

    # --------------------------------------------------------
    # Select observations for this day
    # --------------------------------------------------------

    day_mask = (
        obs["_ssm_date"] == date
    )

    day_obs = (
        obs.loc[day_mask]
        .copy()
    )

    # --------------------------------------------------------
    # Extract this entire day from one SSM file
    # --------------------------------------------------------

    try:

        day_result = extract_one_day(
            date,
            day_obs,
            fs
        )

        all_ssm_results.append(
            day_result
        )

    except Exception as e:

        print(
            f"\nERROR processing {date.date()}:"
        )

        print(e)

        # Keep rows so final SSM dataframe still
        # has exactly the same number of observations.

        error_result = pd.DataFrame(
            {
                "cid": day_obs["cid"].values,
                "lon": day_obs["lon"].values,
                "lat": day_obs["lat"].values,
                "time": day_obs["time"].values,
                "z": day_obs["z"].values,
            }
        )

        all_ssm_results.append(
            error_result
        )


# ============================================================
# COMBINE ALL DAYS
# ============================================================

print("\nCombining daily SSM results...")

ssm = pd.concat(
    all_ssm_results,
    ignore_index=True
)


# ============================================================
# RESTORE ORIGINAL OBSERVATION ORDER
# ============================================================

# The daily grouping changed the order, so we need to
# explicitly restore the original observation order.

original_order = (
    obs[
        [
            "cid",
            "lon",
            "lat",
            "time",
            "z"
        ]
    ]
    .reset_index()
    .rename(
        columns={
            "index": "_original_index"
        }
    )
)

ssm = ssm.merge(
    original_order,
    on=[
        "cid",
        "lon",
        "lat",
        "time",
        "z"
    ],
    how="left"
)

ssm = (
    ssm
    .sort_values("_original_index")
    .drop(columns="_original_index")
    .reset_index(drop=True)
)


# ============================================================
# REMOVE TEMPORARY OBSERVATION COLUMNS
# ============================================================

obs = (
    obs
    .drop(
        columns=[
            "_ssm_node",
            "_ssm_node_distance_deg",
            "_ssm_date"
        ]
    )
)


# ============================================================
# STRICT ALIGNMENT CHECK
# ============================================================

print("\n" + "=" * 60)
print("ALIGNMENT CHECK")
print("=" * 60)

print(f"Number of observations: {len(obs)}")
print(f"Number of SSM rows:     {len(ssm)}")

# ------------------------------------------------------------
# 1. Check CID
# ------------------------------------------------------------

cid_match = np.array_equal(
    obs["cid"].values,
    ssm["cid"].values
)

print(f"\nCID match: {cid_match}")

if not cid_match:
    bad = np.where(obs["cid"].values != ssm["cid"].values)[0]

    print(f"Number of CID mismatches: {len(bad)}")
    print("\nFirst 10 CID mismatches:")

    for i in bad[:10]:
        print(
            f"row {i}: "
            f"obs={obs.iloc[i]['cid']} | "
            f"ssm={ssm.iloc[i]['cid']}"
        )


# ------------------------------------------------------------
# 2. Check time
# ------------------------------------------------------------

obs_time = pd.to_datetime(obs["time"]).values
ssm_time = pd.to_datetime(ssm["time"]).values

time_match = np.array_equal(obs_time, ssm_time)

print(f"\nTime match: {time_match}")

if not time_match:

    time_diff = (
        pd.to_datetime(ssm["time"]).reset_index(drop=True)
        - pd.to_datetime(obs["time"]).reset_index(drop=True)
    )

    bad = np.where(time_diff != pd.Timedelta(0))[0]

    print(f"Number of time mismatches: {len(bad)}")

    print("\nFirst 10 time mismatches:")

    for i in bad[:10]:
        print(
            f"row {i}: "
            f"obs={obs.iloc[i]['time']} | "
            f"ssm={ssm.iloc[i]['time']} | "
            f"diff={time_diff.iloc[i]}"
        )


# ------------------------------------------------------------
# 3. Check depth
# ------------------------------------------------------------

obs_z = obs["z"].astype(float).values
ssm_z = ssm["z"].astype(float).values

z_diff = ssm_z - obs_z

print(f"\nMaximum |depth difference|: "
      f"{np.nanmax(np.abs(z_diff)):.6f} m")

print(f"Mean |depth difference|: "
      f"{np.nanmean(np.abs(z_diff)):.6f} m")

z_match = np.allclose(
    obs_z,
    ssm_z,
    equal_nan=True,
    atol=0.5
)

print(f"Depth match within 0.5 m: {z_match}")

if not z_match:

    bad = np.where(
        ~np.isclose(obs_z, ssm_z, equal_nan=True, atol=0.5)
    )[0]

    print(f"Number of depth mismatches: {len(bad)}")

    print("\nFirst 10 depth mismatches:")

    for i in bad[:10]:
        print(
            f"row {i}: "
            f"obs={obs.iloc[i]['z']:.3f} | "
            f"ssm={ssm.iloc[i]['z']:.3f} | "
            f"diff={z_diff[i]:.3f}"
        )


# ------------------------------------------------------------
# 4. Check longitude / latitude
# ------------------------------------------------------------

lon_diff = (
    ssm["lon"].astype(float).values
    - obs["lon"].astype(float).values
)

lat_diff = (
    ssm["lat"].astype(float).values
    - obs["lat"].astype(float).values
)

horizontal_distance_km = (
    np.sqrt(
        lon_diff**2 +
        lat_diff**2
    ) * 111.0
)

print(f"\nMaximum horizontal distance: "
      f"{np.nanmax(horizontal_distance_km):.3f} km")

print(f"Mean horizontal distance: "
      f"{np.nanmean(horizontal_distance_km):.3f} km")


# ------------------------------------------------------------
# 5. Check exact row identity
# ------------------------------------------------------------

key_cols = ["cid", "lon", "lat", "time", "z"]

print("\nChecking observation keys...")

obs_keys = obs[key_cols].copy()
ssm_keys = ssm[key_cols].copy()

for col in ["lon", "lat", "z"]:
    obs_keys[col] = obs_keys[col].astype(float).round(6)
    ssm_keys[col] = ssm_keys[col].astype(float).round(6)

obs_keys["time"] = pd.to_datetime(obs_keys["time"])
ssm_keys["time"] = pd.to_datetime(ssm_keys["time"])

exact_key_match = obs_keys.equals(ssm_keys)

print(f"Exact key match: {exact_key_match}")


# ------------------------------------------------------------
# 6. Check whether the same observations exist, regardless
#    of order
# ------------------------------------------------------------

obs_key_counts = (
    obs_keys
    .value_counts()
    .sort_index()
)

ssm_key_counts = (
    ssm_keys
    .value_counts()
    .sort_index()
)

same_keys_unordered = obs_key_counts.equals(ssm_key_counts)

print(f"Same observation keys ignoring order: "
      f"{same_keys_unordered}")


# ------------------------------------------------------------
# 7. If order differs, identify what happened
# ------------------------------------------------------------

if not exact_key_match:

    print("\n" + "-" * 60)
    print("ORDER / KEY DIAGNOSTICS")
    print("-" * 60)

    # Keys in obs but not SSM
    missing_from_ssm = obs_key_counts.subtract(
        ssm_key_counts,
        fill_value=0
    )

    missing_from_ssm = missing_from_ssm[
        missing_from_ssm > 0
    ]

    print(
        f"\nObservation keys missing from SSM: "
        f"{len(missing_from_ssm)}"
    )

    # Keys in SSM but not obs
    extra_in_ssm = ssm_key_counts.subtract(
        obs_key_counts,
        fill_value=0
    )

    extra_in_ssm = extra_in_ssm[
        extra_in_ssm > 0
    ]

    print(
        f"SSM keys not present in observations: "
        f"{len(extra_in_ssm)}"
    )

    if same_keys_unordered:
        print(
            "\nIMPORTANT: The same observations are present "
            "in both tables, but the ROW ORDER differs."
        )
    else:
        print(
            "\nIMPORTANT: The SSM table does not contain "
            "exactly the same observation keys as obs."
        )


# ============================================================
# ADD SSM TO ORIGINAL DATA DICTIONARY
# ============================================================

data["ssm"] = ssm


# ============================================================
# SAVE
# ============================================================

print("\nSaving combined pickle...")

with open(
    pickle_out,
    "wb"
) as f:

    pickle.dump(
        data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print(
    "\nSaved:"
)

print(pickle_out)

print(
    "\nFinal keys:"
)

print(data.keys())