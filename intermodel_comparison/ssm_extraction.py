'''
SSM WQM extraction at observation locations/times/depths

Uses:
  data = combined_bottle_2014_cas7_t1_x11ab_ssc.pkl 
  data["obs"] as the master observation targets

SSM data:
  https://s3.kopah.uw.edu/ssm/wqm/2014/
  
SSM grid:
    EPSG:26910 (NAD83 / UTM Zone 10N)

No complete daily files are downloaded. fsspec + scipy
access the NetCDF-3 files remotely using HTTP range requests.
'''

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
# SETTINGS
# ============================================================

year = 2014

pickle_path = (
    Ldir['LOo'] /'obsmod' /
    f"combined_bottle_{year}_cas7_t1_x11ab_ssc.pkl"
)

pickle_out = (
    Ldir['LOo'] /'obsmod' /
        f"combined_bottle_{year}_cas7_t1_x11ab_ssc_ssmpnnl.pkl"
)

ssm_base_url = (
    f"https://s3.kopah.uw.edu/ssm/wqm/{year}/"
)


# ============================================================
# LOAD EXISTING COMBINED DATA
# ============================================================

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

obs = data["obs"].copy()

print("Number of observations:", len(obs))


# ============================================================
# OPEN ONE SSM FILE TO GET STATIC GRID INFORMATION
# ============================================================

grid_url = (
    ssm_base_url +
    "ssm_FVCOMICM_00001.nc"
)

fs = fsspec.filesystem("https")


with fs.open(grid_url, "rb") as f:

    nc = netcdf_file(
        f,
        mode="r",
        mmap=False
    )

    # --------------------------------------------------------
    # SSM horizontal coordinates
    # --------------------------------------------------------

    x = nc.variables["x"][:]
    y = nc.variables["y"][:]

    # SSM coordinates are NAD83 / UTM Zone 10N
    transformer = pyproj.Transformer.from_crs(
        "epsg:26910",
        "epsg:4326",
        always_xy=True
    )

    lon_ssm, lat_ssm = transformer.transform(
        x,
        y
    )

    # --------------------------------------------------------
    # Nearest-node tree
    # --------------------------------------------------------

    ssm_tree = cKDTree(
        np.column_stack(
            (lon_ssm, lat_ssm)
        )
    )

    # --------------------------------------------------------
    # Sigma layers
    # --------------------------------------------------------

    siglay = nc.variables["siglay"][:]


print("SSM nodes:", len(x))
print("SSM sigma layers:", len(siglay))


# ============================================================
# HELPER: GET SSM FILE NAME
# ============================================================

def get_ssm_filename(timestamp):

    timestamp = pd.Timestamp(timestamp)

    day_of_year = timestamp.dayofyear

    return (
        f"ssm_FVCOMICM_{day_of_year:05d}.nc"
    )


# ============================================================
# HELPER: EXTRACT ONE OBSERVATION
# ============================================================

def extract_ssm_observation(row):

    # --------------------------------------------------------
    # Observation information
    # --------------------------------------------------------

    obs_time = pd.Timestamp(row["time"])

    obs_lon = float(row["lon"])
    obs_lat = float(row["lat"])
    obs_depth = float(row["z"])

    # --------------------------------------------------------
    # Daily SSM file
    # --------------------------------------------------------

    filename = get_ssm_filename(obs_time)

    url = ssm_base_url + filename

    # --------------------------------------------------------
    # Nearest SSM node
    # --------------------------------------------------------

    distance, node_idx = ssm_tree.query(
        [obs_lon, obs_lat]
    )

    # --------------------------------------------------------
    # Open SSM daily file remotely
    # --------------------------------------------------------

    with fs.open(url, "rb") as f:

        nc = netcdf_file(
            f,
            mode="r",
            mmap=False
        )

        # ====================================================
        # TIME
        # ====================================================

        time_values = nc.variables["time"][:]

        # SSM time is seconds after midnight
        seconds_since_midnight = (
            obs_time.hour * 3600
            + obs_time.minute * 60
            + obs_time.second
            + obs_time.microsecond / 1e6
        )

        time_idx = np.argmin(
            np.abs(
                time_values -
                seconds_since_midnight
            )
        )

        # Difference between observation and SSM time
        ssm_seconds = float(
            time_values[time_idx]
        )

        time_difference_seconds = (
            ssm_seconds -
            seconds_since_midnight
        )

        # ====================================================
        # WATER-COLUMN DEPTH
        # ====================================================

        H = float(
            nc.variables["depth"][
                time_idx,
                node_idx
            ]
        )

        # ====================================================
        # PHYSICAL DEPTH OF SSM SIGMA LAYERS
        # ====================================================

        layer_depths = siglay * H

        # ----------------------------------------------------
        # Nearest SSM vertical layer
        # ----------------------------------------------------

        layer_idx = np.argmin(
            np.abs(
                layer_depths -
                obs_depth
            )
        )

        ssm_depth = float(
            layer_depths[layer_idx]
        )

        # ====================================================
        # EXTRACT SSM VARIABLES
        # ====================================================

        DOXG = float(
            nc.variables["DOXG"][
                time_idx,
                layer_idx,
                node_idx
            ]
        )

        temp = float(
            nc.variables["temp"][
                time_idx,
                layer_idx,
                node_idx
            ]
        )

        salinity = float(
            nc.variables["salinity"][
                time_idx,
                layer_idx,
                node_idx
            ]
        )

        NO3 = float(
            nc.variables["NO3"][
                time_idx,
                layer_idx,
                node_idx
            ]
        )

        NH4 = float(
            nc.variables["NH4"][
                time_idx,
                layer_idx,
                node_idx
            ]
        )

        TDIC = float(
            nc.variables["TDIC"][
                time_idx,
                layer_idx,
                node_idx
            ]
        )

        TALK = float(
            nc.variables["TALK"][
                time_idx,
                layer_idx,
                node_idx
            ]
        )

        # ====================================================
        # CONVERSIONS
        # ====================================================

        # ----------------------------------------------------
        # Dissolved oxygen
        #
        # MG/L -> umol/L
        # ----------------------------------------------------

        DO_uM = (
            DOXG *
            1000.0 /
            31.998
        )

        # ----------------------------------------------------
        # Nutrients
        #
        # g N m-3 -> umol N L-1
        # ----------------------------------------------------

        NO3_uM = (
            NO3 *
            1000.0 /
            14.007
        )

        NH4_uM = (
            NH4 *
            1000.0 /
            14.007
        )

        # ----------------------------------------------------
        # DIC and TA
        #
        # mmol m-3 = umol L-1
        # ----------------------------------------------------

        DIC_uM = TDIC

        TA_uM = TALK

        # ====================================================
        # GSW CONVERSION
        # ====================================================

        # ----------------------------------------------------
        # Pressure from depth
        #
        # gsw expects positive pressure in dbar.
        # ----------------------------------------------------

        pressure = gsw.p_from_z(
            ssm_depth,
            obs_lat
        )

        # ----------------------------------------------------
        # Absolute Salinity
        # ----------------------------------------------------

        SA = gsw.SA_from_SP(
            salinity,
            pressure,
            obs_lon,
            obs_lat
        )

        # ----------------------------------------------------
        # Conservative Temperature
        # ----------------------------------------------------

        CT = gsw.CT_from_t(
            SA,
            temp,
            pressure
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "cid": row["cid"],
        "cruise": row["cruise"],
        "time": row["time"],
        "lat": row["lat"],
        "lon": row["lon"],
        "name": row["name"],
        "z": row["z"],
        "source": row["source"],

        # ----------------------------------------------------
        # Model coordinates / matching information
        # ----------------------------------------------------

        "ssm_node": node_idx,
        "ssm_distance": distance,
        "ssm_time_idx": time_idx,
        "ssm_time_diff_sec": time_difference_seconds,

        "ssm_H": H,
        "ssm_layer": layer_idx,
        "ssm_depth": ssm_depth,

        # ----------------------------------------------------
        # Comparison variables
        # ----------------------------------------------------

        "SA": SA,
        "CT": CT,
        "DO": DO_uM,
        "NO3": NO3_uM,
        "NH4": NH4_uM,
        "TA": TA_uM,
        "DIC": DIC_uM,

        # ----------------------------------------------------
        # Chlorophyll
        #
        # Leave NaN until phytoplankton carbon pool is
        # identified.
        # ----------------------------------------------------

        "Chl (mg m-3)": np.nan,

        # # ----------------------------------------------------
        # # Raw SSM values for checking
        # # ----------------------------------------------------

        # "SSM_DOXG (mg L-1)": DOXG,
        # "SSM_temp": temp,
        # "SSM_salinity": salinity,
        # "SSM_NO3 (gN m-3)": NO3,
        # "SSM_NH4 (gN m-3)": NH4,
        # "SSM_TDIC (mmolC m-3)": TDIC,
        # "SSM_TALK (mmol m-3)": TALK,
    }


# ============================================================
# TEST ONE OBSERVATION FIRST
# ============================================================

# test_result = extract_ssm_observation(
#     obs.iloc[0]
# )

# print("\nTest observation:")
# for key, value in test_result.items():
#     print(f"{key}: {value}")


# ============================================================
# EXTRACT ALL OBSERVATIONS
# ============================================================

print("\nExtracting SSM values for all observations...")

ssm_results = []

for i, row in obs.iterrows():

    if i % 100 == 0:
        print(
            f"Processing observation "
            f"{i + 1} / {len(obs)}"
        )

    try:

        result = extract_ssm_observation(row)

    except Exception as e:

        print(
            f"ERROR at observation {i}: {e}"
        )

        # Keep observation identity even if extraction fails
        result = {
            "cid": row["cid"],
            "lon": row["lon"],
            "lat": row["lat"],
            "time": row["time"],
            "z": row["z"],
        }

    ssm_results.append(result)
    
    # ============================================================
# CREATE SSM DATAFRAME
# ============================================================

ssm = pd.DataFrame(
    ssm_results
)


# ============================================================
# CHECK THAT ROWS MATCH OBSERVATIONS
# ============================================================

print("\nChecking observation alignment...")

print(
    "Number of obs:",
    len(obs)
)

print(
    "Number of SSM rows:",
    len(ssm)
)

print(
    "CID match:",
    obs["cid"].reset_index(drop=True).equals(
        ssm["cid"].reset_index(drop=True)
    )
)

print(
    "Time match:",
    pd.to_datetime(
        obs["time"]
    ).reset_index(drop=True).equals(
        pd.to_datetime(
            ssm["time"]
        ).reset_index(drop=True)
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
# ADD SSM TO EXISTING DICTIONARY
# ============================================================

data["ssm"] = ssm


# ============================================================
# SAVE NEW PICKLE
# ============================================================

with open(pickle_out, "wb") as f:

    pickle.dump(
        data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print("\nFinished!")

print(
    f"Saved combined data to:\n{pickle_out}"
)

print("\nFinal keys:")
print(data.keys())