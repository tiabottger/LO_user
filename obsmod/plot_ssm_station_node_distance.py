import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import xarray as xr
from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

Ldir = Lfun.Lstart()

# ============================================================
# LOAD OBSERVATION / MODEL DATA
# ============================================================

Ldir = Lfun.Lstart(gridname="cas7")

pkl_path = (
    Path(Ldir["LOo"])
    / "obsmod"
    / "combined_bottle_2014_cas7_t1_x11ab_ssc_ssmpnnl_ecology.pkl"
)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

ecology = data["ecology"]
plot_df = ecology.copy()

# Keep rows with valid coordinates and distance.
plot_df = plot_df.dropna(
    subset=["lon", "lat", "horizontal_distance_km"]
)

# ============================================================
# LOAD GRID FOR MAP BACKGROUND
# ============================================================

grid_fn = Ldir["grid"] / "grid.nc"
grid_ds = xr.open_dataset(grid_fn)

lon_rho = grid_ds["lon_rho"].values
lat_rho = grid_ds["lat_rho"].values
h = grid_ds["h"].values
mask_rho = grid_ds["mask_rho"].values

# Mask land.
h_plot = np.where(mask_rho == 1, h, np.nan)

# Map extent
lat_low = 46.95
lat_high = 48.35
lon_low = -123.3
lon_high = -122.1

# ============================================================
# SET COLOR SCALE
# ============================================================

# Use the maximum distance so all observations are represented
# across the full color scale.
distance_max = plot_df["horizontal_distance_km"].max()

# ============================================================
# PLOT STATION LOCATIONS COLORED BY DISTANCE TO ECOLOGY NODE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Bathymetry background.
pcm = ax.pcolormesh(
    lon_rho,
    lat_rho,
    h_plot,
    cmap="Greys",
    shading="auto",
    zorder=1
)

# Add coastlines
pfun.add_coast(ax)

# Observation locations colored by distance to nearest Ecology node.
sc = ax.scatter(
    plot_df["lon"],
    plot_df["lat"],
    c=plot_df["horizontal_distance_km"],
    cmap="viridis",
    vmin=0,
    #vmax=distance_max,
    vmax = 1.5,
    s=90,
    edgecolor="black",
    linewidth=0.4,
    zorder=10
)

# Colorbar.
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Distance to nearest Ecology node (km)")

# Axes and title.
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Observation Locations: Distance to Nearest Ecology Node")

# Apply map extent
ax.set_xlim(lon_low, lon_high)
ax.set_ylim(lat_low, lat_high)

ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.3)

plt.tight_layout()

# ============================================================
# SAVE FIGURE
# ============================================================

plot_dir = Path(Ldir["LOo"]) / "obsmod_plots"
plot_dir.mkdir(parents=True, exist_ok=True)

out_fn = plot_dir / "ssm_station_node_distance_ecology.png"

fig.savefig(
    out_fn,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Figure saved to:\n{out_fn}")