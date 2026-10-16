
from pathlib import Path
import numpy as np
from scipy.io import loadmat

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
