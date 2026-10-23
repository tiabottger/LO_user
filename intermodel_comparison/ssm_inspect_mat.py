
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import scipy.io as sio

mat_path = Path(
    "/dat2/tbottger/SSM_ecology/"
    "2014_Exist_W3_WQ_hot10_Base4_nodes.mat"
)

print("File:", mat_path)
print("Size:", mat_path.stat().st_size, "bytes")

data = loadmat(mat_path, squeeze_me=False)

print("\nMATLAB variables:")
for name, value in data.items():
    if name.startswith("__"):
        continue

    arr = np.asarray(value)

    print(f"\n{name}")
    print("  Shape:", arr.shape)
    print("  Dtype:", arr.dtype)

    if arr.size <= 30:
        print("  Values:", arr)
    elif arr.dtype.kind in "iuf":
        finite = arr[np.isfinite(arr)]
        if finite.size:
            print("  Min:", finite.min())
            print("  Max:", finite.max())
            print("  First values:", arr.ravel()[:10])
    else:
        print("  First values:", arr.ravel()[:5])

mat = sio.loadmat(
    mat_path,
    squeeze_me=True,
    struct_as_record=False
)

grd = mat["grd"]
horiz = grd.horiz
cellidx = grd.cellidx

# ------------------------------------------------------------
# 1. Inspect horizontal-grid fields
# ------------------------------------------------------------
print("=" * 70)
print("HORIZONTAL GRID FIELDS")
print("=" * 70)

print("Fields:", horiz._fieldnames)

for field in horiz._fieldnames:
    value = getattr(horiz, field)
    arr = np.asarray(value)

    print(
        f"{field:20s} shape={str(arr.shape):15s} "
        f"dtype={arr.dtype}"
    )

# ------------------------------------------------------------
# 2. Find longitude and latitude arrays
# ------------------------------------------------------------
def find_field(obj, candidates):
    """Find a field using case-insensitive candidate names."""
    fields = obj._fieldnames or []
    lookup = {field.lower(): field for field in fields}

    for candidate in candidates:
        if candidate.lower() in lookup:
            field = lookup[candidate.lower()]
            return field, np.asarray(
                getattr(obj, field)
            ).squeeze()

    return None, None


lon_name, lon = find_field(
    horiz, ["lon", "longitude", "long"]
)
lat_name, lat = find_field(
    horiz, ["lat", "latitude"]
)

if lon is None or lat is None:
    raise ValueError(
        "Could not identify longitude/latitude fields. "
        "Check the horizontal-grid fields printed above."
    )

lon = np.asarray(lon).ravel()
lat = np.asarray(lat).ravel()

print("\n" + "=" * 70)
print("COORDINATE CHECK")
print("=" * 70)

print("Longitude field:", lon_name)
print("Latitude field:", lat_name)
print("Longitude shape:", lon.shape)
print("Latitude shape:", lat.shape)

assert len(lon) == len(lat), "Coordinate lengths differ"
assert len(lon) == 16012, "Unexpected number of horizontal nodes"

print("Number of nodes:", len(lon))
print("Finite longitude:", np.isfinite(lon).sum())
print("Finite latitude:", np.isfinite(lat).sum())
print("Longitude range:", np.nanmin(lon), np.nanmax(lon))
print("Latitude range:", np.nanmin(lat), np.nanmax(lat))

# ------------------------------------------------------------
# 3. Inspect selected 1-based node IDs
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("SELECTED NODE COORDINATES (1-BASED NODE IDs)")
print("=" * 70)

node_ids = [1, 2, 3, 9515, 16012]

for node_id in node_ids:
    i = node_id - 1  # Convert to Python's 0-based index
    print(
        f"Node {node_id:5d}: "
        f"lon={lon[i]:.6f}, lat={lat[i]:.6f}"
    )

# ------------------------------------------------------------
# 4. Check cellidx node/layer mapping
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("CELL INDEX MAPPING")
print("=" * 70)

print("cellidx fields:", cellidx._fieldnames)

for field in ["idx", "I", "J", "K", "IJK"]:
    if field not in cellidx._fieldnames:
        continue

    arr = np.asarray(getattr(cellidx, field)).squeeze()
    print(f"\n{field}: shape={arr.shape}, dtype={arr.dtype}")
    print("First 20:", arr.ravel()[:20])

# If I is the horizontal node ID for each flattened entry,
# it should repeat each node ID for its ten layers.
if "I" in cellidx._fieldnames:
    node_mapping = np.asarray(cellidx.I).squeeze().ravel()

    expected_nodes = np.repeat(
        np.arange(1, 16012 + 1), 10
    )

    print(
        "\ncellidx.I matches node-major mapping:",
        np.array_equal(node_mapping, expected_nodes)
    )
