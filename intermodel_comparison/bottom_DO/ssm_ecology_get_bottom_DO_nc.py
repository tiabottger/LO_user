
"""
Calculate bottom dissolved oxygen and hypoxic-layer thickness
from an SSM/FVCOM box extraction.

Input dimensions:
    time, siglay, local_node
    time, siglev, local_node (interfaces are derived from siglev)

Output dimensions:
    time, local_node

The output retains the extracted FVCOM mesh coordinates and
connectivity for spatial plotting.
"""
from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# USER INPUTS
# ============================================================

year = 2014

input_fp = Path(
    f"/dat2/tbottger/SSM_ecology/ecology_box/{year}/"
    f"Ecology_{year}_box_subset.nc"
)

out_dir = Path(
    f"/dat2/tbottger/SSM_ecology/bottom_DO/{year}"
)
out_dir.mkdir(parents=True, exist_ok=True)

HYPOXIC_THRESHOLD = 2.0
ONE_MGL_THRESHOLD = 1.0
THREE_MGL_THRESHOLD = 3.0


# ============================================================
# PROCESS DATA
# ============================================================

with xr.open_dataset(input_fp) as ds:

    time = ds["time"].values

    # Static bathymetry, positive downward
    h = ds["bottom_depth"].values.astype(float)

    # Daily mean surface elevation
    zeta = ds["zeta"].values.astype(float)

    # Daily mean dissolved oxygen
    oxygen = ds["DO"].values.astype(float)

    # Reconstructed layer-center depths, negative downward
    layer_depth = ds["layer_depth"].values.astype(float)

    lon = ds["lon"].values
    lat = ds["lat"].values

    Nt, Nnode, Nlay = oxygen.shape

    assert zeta.shape == (Nt, Nnode, Nlay)
    assert layer_depth.shape == (Nt, Nnode, Nlay)

    # zeta is repeated across layers in the current extraction.
    zeta_node = zeta[:, :, 0]

    # Bottom layer is the last layer.
    DO_bot = oxygen[:, :, -1].copy()

    # Physical water-column depth
    water_depth = h[None, :] + zeta_node

    # Mask invalid columns
    valid_column = (
        np.isfinite(h)[None, :]
        & np.isfinite(zeta_node)
        & (h[None, :] > 0)
        & (water_depth > 0)
    )

    DO_bot[~valid_column] = np.nan

    # --------------------------------------------------------
    # Bottom-layer thickness
    # --------------------------------------------------------
    #
    # The layer-center depths alone do not provide exact
    # layer interfaces. Do not calculate exact thicknesses
    # from center-depth differences.
    #
    # Add layer_thickness_fraction to the extraction from
    # the Ecology MAT grid metadata before calculating this.
    #
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Assemble output
    # --------------------------------------------------------

    ds_out = xr.Dataset(
        data_vars={
            "depth_bot": (
                ("local_node",),
                h,
            ),
            "DO_bot": (
                ("time", "local_node"),
                DO_bot,
            ),
            "lon": (
                ("local_node",),
                lon,
            ),
            "lat": (
                ("local_node",),
                lat,
            ),
        },
        coords={
            "time": time,
            "local_node": ds["local_node"].values,
        },
    )

    ds_out["depth_bot"].attrs.update({
        "long_name": "water column depth at mean sea level",
        "units": "m",
    })

    ds_out["DO_bot"].attrs.update({
        "long_name": "dissolved oxygen concentration in bottom layer",
        "units": "mg/L",
    })

    ds_out.attrs["description"] = (
        "Ecology bottom dissolved oxygen calculated from "
        "daily box extraction."
    )
    ds_out.attrs["year"] = year
    ds_out.attrs["hypoxia_threshold_mgL"] = HYPOXIC_THRESHOLD

    output_fp = out_dir / (
        f"Ecology_{year}_bottom_DO_info.nc"
    )

    ds_out.to_netcdf(
        output_fp,
        encoding={
            name: {"zlib": True, "complevel": 4}
            for name in ds_out.data_vars
            if ds_out[name].dtype.kind == "f"
        },
    )

    print(f"Saved: {output_fp}")
    print(ds_out)
