"""
Inspect the Salish Sea Model NetCDF output before building
the observation-matching workflow.

The script:
  1. Reports dimensions, variables, and metadata.
  2. Examines time coverage and time-coordinate conventions.
  3. Identifies mesh coordinates and connectivity.
  4. Examines vertical coordinates and sigma layers.
  5. Samples selected variables without loading the full file.
  6. Reports missing values and approximate sampled ranges.

The full NetCDF is never loaded into memory.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ============================================================
# CONFIGURATION
# ============================================================

nc_path = Path("/dat2/tbottger/SSM_ecology/2014_Exist_W3_WQ_hot10_Base4_nodes.nc")

# Maximum number of horizontal locations sampled per variable
MAX_HORIZONTAL_SAMPLES = 100

# Maximum number of samples along other large dimensions
MAX_OTHER_SAMPLES = 20

# Number of time points to inspect
N_TIME_SAMPLES = 3


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def short_value(value, max_chars=180):
    """Convert metadata values to a readable, short string."""
    text = str(value).replace("\n", " ")
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def get_attr(obj, name, default=""):
    return obj.attrs.get(name, default)


def sample_indices(size, max_samples):
    """Evenly distributed indices including the endpoints."""
    if size <= max_samples:
        return np.arange(size)

    return np.unique(
        np.linspace(
            0, size - 1, max_samples, dtype=int
        )
    )


def inspect_sample(da, ds):
    """
    Read a small sample from a DataArray.

    Time dimensions: first, middle, and last records.
    Horizontal dimensions: at most MAX_HORIZONTAL_SAMPLES.
    Vertical dimensions: retain all levels when reasonably small.
    Other large dimensions: sample sparsely.
    """

    indexers = {}

    time_names = {
        "time", "ntime", "ocean_time",
        "it", "mt", "time_counter"
    }

    horizontal_names = {
        "node", "nodes", "nnode", "nele",
        "element", "elements", "cell",
        "nface", "npoints"
    }

    vertical_names = {
        "siglay", "siglev", "layer",
        "layers", "depth", "deptht",
        "nz", "nlayer", "nlevel"
    }

    for dim in da.dims:
        size = da.sizes[dim]
        dim_lower = dim.lower()

        if dim_lower in time_names:
            indices = sample_indices(
                size, N_TIME_SAMPLES
            )
            indexers[dim] = indices

        elif dim_lower in horizontal_names:
            step = max(
                1,
                int(np.ceil(size / MAX_HORIZONTAL_SAMPLES))
            )
            indexers[dim] = slice(0, size, step)

        elif dim_lower in vertical_names:
            if size > MAX_OTHER_SAMPLES:
                step = max(
                    1,
                    int(np.ceil(size / MAX_OTHER_SAMPLES))
                )
                indexers[dim] = slice(0, size, step)

        elif size > MAX_OTHER_SAMPLES:
            step = max(
                1,
                int(np.ceil(size / MAX_OTHER_SAMPLES))
            )
            indexers[dim] = slice(0, size, step)

    sampled = da.isel(indexers).values

    if not np.issubdtype(sampled.dtype, np.number):
        print("    Sample statistics: non-numeric variable")
        return

    values = np.asarray(sampled, dtype=float).ravel()

    # Handle fill values that xarray may not have masked
    fill_value = da.attrs.get("_FillValue")
    if fill_value is not None:
        values[values == fill_value] = np.nan

    finite = values[np.isfinite(values)]

    print(f"    Sample shape: {sampled.shape}")
    print(f"    Finite samples: {finite.size:,}/{values.size:,}")

    if finite.size:
        print(f"    Sample minimum: {finite.min():.6g}")
        print(f"    Sample maximum: {finite.max():.6g}")
        print(f"    Sample mean:    {finite.mean():.6g}")
    else:
        print("    No finite numeric values in sample")


# ============================================================
# CHECK FILE
# ============================================================

if not nc_path.is_file():
    raise FileNotFoundError(
        f"Cannot find NetCDF file: {nc_path.resolve()}\n"
        "Update nc_path at the top of this script."
    )

print(f"File: {nc_path.resolve()}")
print(f"File size: {nc_path.stat().st_size / 1e9:.2f} GB")

print("\nOpening NetCDF metadata...")
print("The full dataset will not be loaded into memory.")

# decode_times=False lets us inspect the original time
# variables and units before deciding how to interpret them.
ds = xr.open_dataset(
    nc_path,
    decode_times=False,
    cache=False
)

try:

    # ========================================================
    # 1. GLOBAL INFORMATION AND DIMENSIONS
    # ========================================================

    print_section("1. GLOBAL ATTRIBUTES")

    if ds.attrs:
        for key, value in ds.attrs.items():
            print(f"{key}: {short_value(value)}")
    else:
        print("No global attributes found.")

    print_section("2. DIMENSIONS")

    for dim, size in ds.sizes.items():
        print(f"{dim:25s} {size:>15,}")

    # ========================================================
    # 2. ALL VARIABLES AND THEIR METADATA
    # ========================================================

    print_section("3. VARIABLES AND METADATA")

    for name, var in ds.variables.items():
        print(f"\n{name}")
        print(f"    Dimensions: {var.dims}")
        print(f"    Shape:      {var.shape}")
        print(f"    Data type:  {var.dtype}")

        for attr in (
            "long_name", "standard_name", "units",
            "positive", "axis", "coordinates",
            "grid", "mesh", "location",
            "cf_role", "start_index"
        ):
            if attr in var.attrs:
                print(
                    f"    {attr}: "
                    f"{short_value(var.attrs[attr])}"
                )

    # ========================================================
    # 3. TIME COORDINATES
    # ========================================================

    print_section("4. TIME COORDINATES")

    var = ds['Var_1']

    print(f"\nVariable: {name}")
    print(f"    Dimensions: {var.dims}")
    print(f"    Shape: {var.shape}")
    print(f"    Units: {get_attr(var, 'units', 'not specified')}")
    print(f"    Calendar: {get_attr(var, 'calendar', 'not specified')}")

    if var.ndim == 0:
        print(f"    Value: {var.values}")

    elif var.size <= 20:
        print(f"    Values: {var.values}")

    elif var.ndim == 1:
        indices = sample_indices(var.size, 5)
        print(f"    Sample indices: {indices}")
        print(f"    Sample values: {var.isel({var.dims[0]: indices}).values}")

    else:
        print("    Multidimensional time-related variable; "
                "not reading the entire variable.")

    # ========================================================
    # 4. SPATIAL GRID AND MESH
    # ========================================================

    print_section("5. SPATIAL COORDINATES AND MESH")

    spatial_tokens = (
        "lon", "lat", "longitude", "latitude",
        "coord", "node", "nele", "element",
        "connect", "nv", "x", "y", "xc", "yc",
        "h", "bathym"
    )

    for name, var in ds.variables.items():
        if any(token in name.lower() for token in spatial_tokens):
            print(
                f"{name:25s} "
                f"dims={str(var.dims):35s} "
                f"shape={str(var.shape):20s} "
                f"units={get_attr(var, 'units', '')}"
            )

    print(
        "\nInspect connectivity metadata (for example, "
        "nv or element-node arrays) to determine how "
        "the triangular mesh is represented."
    )

    # ========================================================
    # 5. VERTICAL COORDINATES
    # ========================================================

    print_section("6. VERTICAL COORDINATES")

    vertical_tokens = (
        "siglay", "siglev", "sigma", "depth",
        "layer", "level", "zeta", "elevation",
        "bathym", "h"
    )

    for name, var in ds.variables.items():
        if any(token in name.lower() for token in vertical_tokens):
            print(f"\n{name}")
            print(f"    Dimensions: {var.dims}")
            print(f"    Shape: {var.shape}")

            for attr in (
                "long_name", "units", "positive",
                "standard_name", "formula_terms"
            ):
                if attr in var.attrs:
                    print(
                        f"    {attr}: "
                        f"{short_value(var.attrs[attr])}"
                    )

            if var.ndim <= 2 and var.size <= 100:
                print(f"    Values: {var.values}")

    # ========================================================
    # 6. WATER-QUALITY AND PHYSICAL VARIABLES
    # ========================================================

    print_section("7. LIKELY MODEL VARIABLES")

    keywords = (
        "do", "oxygen", "dox", "salt", "salinity",
        "temp", "temperature", "no3", "nitrate",
        "nh4", "ammonium", "chl", "chlorophyll",
        "po4", "phosphate", "turb", "ph", "alk",
        "dic", "ta", "carbon"
    )

    candidates = []

    for name, var in ds.data_vars.items():
        searchable = (
            name + " " +
            str(get_attr(var, "long_name", "")) + " " +
            str(get_attr(var, "standard_name", ""))
        ).lower()

        if any(keyword in searchable for keyword in keywords):
            candidates.append(name)

    if candidates:
        for name in candidates:
            var = ds[name]
            print(f"\n{name}")
            print(f"    Dimensions: {var.dims}")
            print(f"    Shape: {var.shape}")
            print(f"    Units: {get_attr(var, 'units', 'not specified')}")
            print(f"    Description: {get_attr(var, 'long_name', '')}")
            inspect_sample(var, ds)
    else:
        print(
            "No variables matched the search terms. "
            "Review the complete variable listing above."
        )

    # ========================================================
    # 7. ALL THREE-DIMENSIONAL AND FOUR-DIMENSIONAL VARIABLES
    # ========================================================

    print_section("8. VARIABLES WITH 3 OR MORE DIMENSIONS")

    for name, var in ds.data_vars.items():
        if var.ndim >= 3 and name not in candidates:
            print(f"\n{name}: dims={var.dims}, shape={var.shape}")
            print(f"    Units: {get_attr(var, 'units', '')}")
            inspect_sample(var, ds)



finally:
    ds.close()
    print("\nClosed NetCDF file.")
