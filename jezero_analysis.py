
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.enums import Resampling

# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DEM_FILE = (
    BASE_DIR
    / "data"
    / "regional"
    / "jezero"
    / "M20_JezeroCrater_CTXDEM_20m.tif"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREVIEW_FILE = OUTPUT_DIR / "jezero_elevation_preview.png"

# -------------------------------------------------
# CHECK DATASET
# -------------------------------------------------

if not DEM_FILE.exists():
    raise FileNotFoundError(f"DEM not found: {DEM_FILE}")

print("\n===== JEZERO CRATER DATASET =====")

with rasterio.open(DEM_FILE) as src:

    print("Width:", src.width, "pixels")
    print("Height:", src.height, "pixels")
    print("Coordinate reference system:", src.crs)
    print("Pixel size:", src.res, "map units")
    print("NoData value:", src.nodata)
    print("Data type:", src.dtypes[0])
    print("Map bounds:", src.bounds)
    print("Band units:", src.units)

    # Read the full-resolution DEM for statistics
    full_data = src.read(1, masked=True)

    if full_data.count() == 0:
        raise ValueError("No valid elevation values found.")

    print("\n===== ELEVATION STATISTICS =====")
    print("Valid pixels:", full_data.count())
    print("Minimum:", float(full_data.min()), "m")
    print("Maximum:", float(full_data.max()), "m")
    print("Average:", round(float(full_data.mean()), 2), "m")

    # Read a smaller version for fast visualisation
    scale = min(1.0, 1400 / max(src.width, src.height))

    preview_height = max(1, int(src.height * scale))
    preview_width = max(1, int(src.width * scale))

    preview = src.read(
        1,
        out_shape=(preview_height, preview_width),
        masked=True,
        resampling=Resampling.average
    )

    bounds = src.bounds

    extent = [
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ]

# -------------------------------------------------
# DISPLAY ELEVATION MAP
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 9))

image = ax.imshow(
    preview,
    cmap="terrain",
    extent=extent,
    origin="upper"
)

ax.set_title(
    "JEZERO CRATER | CTX DIGITAL ELEVATION MODEL",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Mars Easting (map units)")
ax.set_ylabel("Mars Northing (map units)")

fig.colorbar(
    image,
    ax=ax,
    label="Elevation (metres)"
)

plt.tight_layout()

plt.savefig(
    PREVIEW_FILE,
    dpi=180,
    bbox_inches="tight"
)

print("\nElevation preview saved:")
print(PREVIEW_FILE)

plt.show()