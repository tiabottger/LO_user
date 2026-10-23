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

    print("\n" + "=" * 70)
    print("TIME VARIABLES AND ATTRIBUTES")
    print("=" * 70)

    for name, var in ds.variables.items():
        if "time" in name.lower() or name.lower() in (
            "gemss_time", "mat_time"
        ):
            print(f"\n{name}")
            print("  dimensions:", var.dims)
            print("  attributes:", var.attrs)

            if var.size <= 10000:
                print("  values:", var.values.flat[:5])
                print("  first:", var.values.flat[0])
                print("  last:", var.values.flat[-1])

    # ============================================================
    # 3. INSPECT VAR_9
    # ============================================================

    print("\n" + "=" * 70)
    print("VAR_9 METADATA")
    print("=" * 70)

    # Locate the variable without assuming capitalization.
    var9_name = next(
        (name for name in ds.variables
        if name.lower() == "var_9"),
        None
    )

    if var9_name is None:
        print("Var_9 not found. Check the variable list above.")
    else:
        z = ds[var9_name]

        print("Name:", var9_name)
        print("Dimensions:", z.dims)
        print("Shape:", z.shape)
        print("Data type:", z.dtype)
        print("Attributes:", z.attrs)

        # Identify a possible time dimension.
        time_dims = [
            dim for dim in z.dims
            if "time" in dim.lower()
        ]

        if time_dims:
            time_dim = time_dims[0]
            ntime = z.sizes[time_dim]

            sample_indices = sorted(set([
                0, ntime // 2, ntime - 1
            ]))

            for ti in sample_indices:
                # Read only one time slice.
                sample = z.isel({time_dim: ti}).values
                finite = sample[np.isfinite(sample)]

                print(f"\nVar_9 at {time_dim} index {ti}")
                print("  shape:", sample.shape)
                print("  finite count:", finite.size)

                if finite.size:
                    print("  min:", finite.min())
                    print("  max:", finite.max())
                    print("  first 12 values:", finite.flat[:12])

            # Compare first and last sampled times if shapes agree.
            z_first = z.isel({time_dim: 0}).values
            z_last = z.isel({time_dim: ntime - 1}).values

            valid = np.isfinite(z_first) & np.isfinite(z_last)

            if valid.any():
                diff = z_last[valid] - z_first[valid]

                print("\nFirst-to-last depth comparison:")
                print("  valid paired values:", valid.sum())
                print("  maximum absolute difference:",
                    np.max(np.abs(diff)))
                print("  median difference:", np.median(diff))
        else:
            print("No obvious time dimension in Var_9.")
            print("Inspect its dimensions above to determine its layout.")

    # ============================================================
    # 5. CHECK NODE-LAYER DIMENSIONS
    # ============================================================

    print("\n" + "=" * 70)
    print("DIMENSIONS RELEVANT TO FLATTENED INDEXING")
    print("=" * 70)

    for dim, size in ds.sizes.items():
        print(f"{dim:25s} {size}")

    print("\nInspection complete.")
    ds.close()
    

finally:
    ds.close()
    print("\nClosed NetCDF file.")
