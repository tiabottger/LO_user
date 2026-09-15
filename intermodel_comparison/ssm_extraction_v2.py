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
# CHECK ALIGNMENT
# ============================================================

print("\n============================================")
print("ALIGNMENT CHECK")
print("============================================")

print(
    "Number of observations:",
    len(obs)
)

print(
    "Number of SSM rows:",
    len(ssm)
)

print(
    "\nCID match:",
    obs["cid"]
    .reset_index(drop=True)
    .equals(
        ssm["cid"]
        .reset_index(drop=True)
    )
)

print(
    "Time match:",
    pd.to_datetime(
        obs["time"]
    )
    .reset_index(drop=True)
    .equals(
        pd.to_datetime(
            ssm["time"]
        )
        .reset_index(drop=True)
    )
)

print(
    "Depth match:",
    np.allclose(
        obs["z"].values,
        ssm["z"].values,
        equal_nan=True
    )
)


# ============================================================
# CHECK TIME MATCHING
# ============================================================

print("\n============================================")
print("SSM TIME MATCHING")
print("============================================")

print(
    ssm[
        [
            "time",
            "ssm_time",
            "ssm_time_difference_hours"
        ]
    ].head(10)
)

print(
    "\nMaximum SSM time difference:",
    ssm[
        "ssm_time_difference_hours"
    ].max(),
    "hours"
)

print(
    "Mean SSM time difference:",
    ssm[
        "ssm_time_difference_hours"
    ].mean(),
    "hours"
)


# ============================================================
# CHECK VERTICAL MATCHING
# ============================================================

print("\n============================================")
print("SSM VERTICAL MATCHING")
print("============================================")

print(
    "Mean vertical difference:",
    ssm[
        "ssm_vertical_difference"
    ].mean(),
    "m"
)

print(
    "Maximum vertical difference:",
    ssm[
        "ssm_vertical_difference"
    ].max(),
    "m"
)


# ============================================================
# CHECK HORIZONTAL MATCHING
# ============================================================

print("\n============================================")
print("SSM HORIZONTAL MATCHING")
print("============================================")

print(
    "Mean node distance:",
    ssm[
        "ssm_node_distance_deg"
    ].mean(),
    "degrees"
)

print(
    "Maximum node distance:",
    ssm[
        "ssm_node_distance_deg"
    ].max(),
    "degrees"
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