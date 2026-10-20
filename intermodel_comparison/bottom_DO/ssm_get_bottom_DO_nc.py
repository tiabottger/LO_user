
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
    f"/dat2/tbottger/SSM_pnnl/ssm_pugetsound/{year}/SSM_{year}_box_subset.nc"
)

out_dir = Path(
    f"/dat2/tbottger/SSM_pnnl/bottom_DO/{year}"
)
out_dir.mkdir(parents=True, exist_ok=True)

# Dissolved oxygen thresholds (mg/L)
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
        "Static bathymetric depth h; positive downward."
    )

    ds.attrs["description"] = (
        "SSM/FVCOM bottom dissolved oxygen and hypoxic "
        "thickness calculated from a regional box extraction."
    )
    ds.attrs["year"] = year
    ds.attrs["oxygen_units"] = "mg/L"
    ds.attrs["hypoxia_threshold_mgL"] = HYPOXIC_THRESHOLD

    return ds


# ============================================================
# PROCESS DATA
# ============================================================

print("Processing started...")
print(f"Input: {input_fp}")

with xr.open_dataset(input_fp) as ds_raw:

    # --------------------------------------------------------
    # Read dimensions and static grid data
    # --------------------------------------------------------

    time = ds_raw["time"]
    h = ds_raw["h"].values.astype(float)
    zeta_all = ds_raw["zeta"].values.astype(float)
    oxygen_all = ds_raw["DOXG"].values.astype(float)

    siglay = ds_raw["siglay"].values.astype(float)
    siglev = ds_raw["siglev"].values.astype(float)

    lon = ds_raw["lon"].values
    lat = ds_raw["lat"].values
    nv = ds_raw["nv"].values

    Nt = len(time)
    Nnode = len(h)
    Nlay = len(siglay)

    print(f"Time records: {Nt}")
    print(f"Nodes: {Nnode}")
    print(f"Vertical layers: {Nlay}")
    print(f"Elements: {nv.shape[0]}")

    # --------------------------------------------------------
    # Validate input dimensions and vertical coordinates
    # --------------------------------------------------------

    assert oxygen_all.shape == (Nt, Nlay, Nnode)
    assert zeta_all.shape == (Nt, Nnode)
    assert len(siglev) == Nlay + 1
    assert nv.ndim == 2 and nv.shape[1] == 3

    # This script assumes sigma interfaces run from surface
    # (approximately 0) to bottom (-1).
    if not (
        np.all(np.diff(siglev) < 0)
        and np.isclose(siglev[0], 0, atol=1e-5)
        and np.isclose(siglev[-1], -1, atol=1e-5)
    ):
        raise ValueError(
            "Unexpected siglev ordering. Inspect siglev before "
            "calculating physical layer thicknesses."
        )

    # --------------------------------------------------------
    # Allocate output arrays
    # --------------------------------------------------------

    DO_bot = np.full((Nt, Nnode), np.nan)

    thick_bot = np.full((Nt, Nnode), np.nan)

    hyp_thick = np.full((Nt, Nnode), np.nan)
    one_mgL_thick = np.full((Nt, Nnode), np.nan)
    three_mgL_thick = np.full((Nt, Nnode), np.nan)

    # --------------------------------------------------------
    # Calculate physical interfaces and oxygen metrics
    # --------------------------------------------------------

    print("Calculating vertical thicknesses and DO metrics...")

    for t in range(Nt):

        if t % 30 == 0:
            print(f"  Processing day {t + 1}/{Nt}")

        zeta = zeta_all[t, :]
        oxy = oxygen_all[t, :, :]

        # Physical elevation of FVCOM vertical interfaces:
        #
        # z = zeta + sigma * (h + zeta)
        #
        # Result shape: (siglev, local_node)
        z_w = (
            zeta[None, :]
            + siglev[:, None]
            * (h[None, :] + zeta[None, :])
        )

        # Physical thickness of each layer.
        # FVCOM interfaces are ordered surface -> bottom,
        # so take the absolute difference.
        dz = np.abs(np.diff(z_w, axis=0))

        # Exclude dry/invalid columns.
        valid_column = (
            np.isfinite(h)
            & np.isfinite(zeta)
            & (h > 0)
            & ((h + zeta) > 0)
        )

        valid_oxygen = np.isfinite(oxy)

        # ----------------------------------------------------
        # Bottom-layer oxygen and thickness
        # ----------------------------------------------------

        # Last sigma layer is nearest the seabed.
        DO_bot[t, :] = oxy[-1, :]

        thick_bot[t, :] = dz[-1, :]

        # ----------------------------------------------------
        # Hypoxic and low-oxygen thickness
        # ----------------------------------------------------

        # Missing oxygen values are not counted as hypoxic.
        # Columns without any valid oxygen are reset to NaN.
        hyp_thick[t, :] = np.sum(
            np.where(
                valid_oxygen & (oxy <= HYPOXIC_THRESHOLD),
                dz,
                0.0,
            ),
            axis=0,
        )

        one_mgL_thick[t, :] = np.sum(
            np.where(
                valid_oxygen & (oxy <= ONE_MGL_THRESHOLD),
                dz,
                0.0,
            ),
            axis=0,
        )

        three_mgL_thick[t, :] = np.sum(
            np.where(
                valid_oxygen & (oxy <= THREE_MGL_THRESHOLD),
                dz,
                0.0,
            ),
            axis=0,
        )

        # Mask columns with no valid oxygen data or invalid
        # water-column geometry.
        invalid_column = (
            ~valid_column
            | ~np.any(valid_oxygen, axis=0)
        )

        for arr in (
            hyp_thick,
            one_mgL_thick,
            three_mgL_thick,
        ):
            arr[t, invalid_column] = np.nan

        DO_bot[t, ~valid_column] = np.nan
        thick_bot[t, ~valid_column] = np.nan

    # --------------------------------------------------------
    # Assemble output dataset
    # --------------------------------------------------------

    print("Assembling output dataset...")

    coords = {
        "time": time.values,
        "local_node": ds_raw["local_node"].values,
    }

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
            "nv": (
                ("elem", "three"),
                nv,
            ),
            "original_node_index": (
                ("local_node",),
                ds_raw["original_node_index"].values,
            ),
            "original_elem_index": (
                ("elem",),
                ds_raw["original_elem_index"].values,
            ),
        },
        coords={
            **coords,
            "elem": ds_raw["elem"].values,
            "three": ds_raw["three"].values,
        },
    )

    ds_out = add_metadata(ds_out)

    # Preserve useful source metadata.
    ds_out.attrs["source_file"] = input_fp.name
    ds_out.attrs["bbox"] = ds_raw.attrs.get("bbox", "unspecified")
    ds_out.attrs["vertical_coordinate"] = (
        "FVCOM sigma coordinates; physical interfaces calculated "
        "from h, zeta, and siglev"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_fp = out_dir / (
        f"SSM_{year}_pugetsound_bottom_DO_info.nc"
    )

    print(f"Saving: {output_fp}")

    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in ds_out.data_vars
        if ds_out[name].dtype.kind == "f"
    }

    ds_out.to_netcdf(output_fp, encoding=encoding)
    ds_out.close()

print("Done.")
