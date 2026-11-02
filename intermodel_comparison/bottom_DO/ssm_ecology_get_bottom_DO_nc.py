
"""
Calculate bottom dissolved oxygen and hypoxic-layer thickness
from an SSM/FVCOM Ecology NetCDF box extraction.

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
# HELPER FUNCTIONS
# ============================================================

def add_metadata(ds):
    """Add descriptive metadata to the output dataset."""

    metadata = {
        "depth_bot": (
            "water column depth at mean sea level",
            "m",
        ),
        "DO_bot": (
            "dissolved oxygen concentration in bottom sigma layer",
            "mg/L",
        ),
        "thick_bot": (
            "thickness of bottom sigma layer",
            "m",
        ),
        "hyp_thick": (
            "water column thickness with DO <= 2 mg/L",
            "m",
        ),
        "one_mgL_thick": (
            "water column thickness with DO <= 1 mg/L",
            "m",
        ),
        "three_mgL_thick": (
            "water column thickness with DO <= 3 mg/L",
            "m",
        ),
    }
    
    for var_name, (long_name, units) in metadata.items():
        ds[var_name].attrs["long_name"] = long_name
        ds[var_name].attrs["units"] = units

    ds["depth_bot"].attrs["comment"] = (
        "Static bathymetric depth; positive downward."
    )

    ds.attrs["description"] = (
        "Ecology bottom dissolved oxygen and hypoxic "
        "thickness calculated from a regional box extraction."
    )

    ds.attrs["year"] = year
    ds.attrs["oxygen_units"] = "mg/L"
    ds.attrs["hypoxia_threshold_mgL"] = HYPOXIC_THRESHOLD

    return ds

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
    
    layer_thickness_fraction = (
        ds["layer_thickness_fraction"].values.astype(float))

    lon = ds["lon"].values
    lat = ds["lat"].values

    Nt = len(time)
    Nnode = len(h)
    Nlay = oxygen.shape[2]

    assert oxygen.shape == (Nt, Nnode, Nlay)
    assert zeta.shape == (Nt, Nnode, Nlay)
    assert layer_depth.shape == (Nt, Nnode, Nlay)
    assert layer_thickness_fraction.shape == (Nnode, Nlay)
    
    assert lon.shape == (Nnode,)
    assert lat.shape == (Nnode,)
    
    # --------------------------------------------------------
    # Validate layer thickness fractions
    # --------------------------------------------------------

    fraction_sums = np.nansum(
        layer_thickness_fraction,
        axis=1,
    )

    if not np.allclose(
        fraction_sums,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Layer-thickness fractions do not sum to 1 "
            "for every node. Check grd.vert.thickness."
        )

    if np.any(layer_thickness_fraction < 0):
        raise ValueError(
            "Negative layer-thickness fractions found."
        )

    print(
        "Layer-thickness fractions validated: "
        "sum to 1 at all nodes."
    )
    
    # --------------------------------------------------------
    # Extract node-level variables
    # --------------------------------------------------------

    # zeta is repeated across layers in the current extraction.
    zeta_node = zeta[:, :, 0]

    # Bottom layer is the last layer.
    DO_bot = oxygen[:, :, -1].copy()
    
        # Total water-column depth, positive downward.
    water_depth = h[None, :] + zeta_node

    # Physical thickness of each layer:
    # dz = layer-thickness fraction * total water depth
    # Shape: (time, local_node, layer)

    dz = (
        water_depth[:, :, None]
        * layer_thickness_fraction[None, :, :]
    )

    assert dz.shape == oxygen.shape
    
    # --------------------------------------------------------
    # Validate water-column geometry
    # --------------------------------------------------------

    valid_column = (
        np.isfinite(h)[None, :]
        & np.isfinite(zeta_node)
        & (h[None, :] > 0)
        & (water_depth > 0)
    )

    # Set invalid layer thicknesses to NaN.
    dz = np.where(
        valid_column[:, :, None],
        dz,
        np.nan,
    )

    # --------------------------------------------------------
    # Allocate output arrays
    # --------------------------------------------------------

    DO_bot = np.full(
        (Nt, Nnode),
        np.nan,
    )

    thick_bot = np.full(
        (Nt, Nnode),
        np.nan,
    )

    hyp_thick = np.full(
        (Nt, Nnode),
        np.nan,
    )

    one_mgL_thick = np.full(
        (Nt, Nnode),
        np.nan,
    )

    three_mgL_thick = np.full(
        (Nt, Nnode),
        np.nan,
    )

    # --------------------------------------------------------
    # Calculate bottom DO and hypoxic thickness
    # --------------------------------------------------------

    print("Calculating bottom DO and hypoxic thickness...")

    for t in range(Nt):

        if t % 30 == 0:
            print(
                f"  Processing day {t + 1}/{Nt}"
            )

        oxy = oxygen[t, :, :]
        layer_dz = dz[t, :, :]

        valid_oxygen = np.isfinite(oxy)

        # A column is valid if geometry is valid and at least
        # one oxygen layer contains valid data.
        valid = (
            valid_column[t, :]
            & np.any(valid_oxygen, axis=1)
        )

        # ----------------------------------------------------
        # Bottom-layer oxygen and thickness
        # ----------------------------------------------------

        DO_bot[t, :] = oxy[:, -1]
        thick_bot[t, :] = layer_dz[:, -1]

        # ----------------------------------------------------
        # Hypoxic and low-oxygen thickness
        # ----------------------------------------------------

        hyp_thick[t, :] = np.sum(
            np.where(
                valid_oxygen
                & (oxy <= HYPOXIC_THRESHOLD),
                layer_dz,
                0.0,
            ),
            axis=1,
        )

        one_mgL_thick[t, :] = np.sum(
            np.where(
                valid_oxygen
                & (oxy <= ONE_MGL_THRESHOLD),
                layer_dz,
                0.0,
            ),
            axis=1,
        )

        three_mgL_thick[t, :] = np.sum(
            np.where(
                valid_oxygen
                & (oxy <= THREE_MGL_THRESHOLD),
                layer_dz,
                0.0,
            ),
            axis=1,
        )

        # ----------------------------------------------------
        # Mask invalid columns
        # ----------------------------------------------------

        for arr in (
            DO_bot,
            thick_bot,
            hyp_thick,
            one_mgL_thick,
            three_mgL_thick,
        ):
            arr[t, ~valid] = np.nan


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
            "thick_bot": (
                ("time", "local_node"),
                thick_bot,
            ),
            "hyp_thick": (
                ("time", "local_node"),
                hyp_thick,
            ),
            "one_mgL_thick": (
                ("time", "local_node"),
                one_mgL_thick,
            ),
            "three_mgL_thick": (
                ("time", "local_node"),
                three_mgL_thick,
            ),
            "lon": (
                ("local_node",),
                lon,
            ),
            "lat": (
                ("local_node",),
                lat,
            ),
             "original_node_index": (
                ("local_node",),
                ds["original_node_index"].values,
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
