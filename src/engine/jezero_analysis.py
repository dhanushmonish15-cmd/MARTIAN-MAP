
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

# Project paths
ROOT = Path(__file__).resolve().parents[2]

DEM_PATH = ROOT / "data" / "regional" / "jezero" / "M20_JezeroCrater_CTXDEM_20m.tif"
OUTPUT_PATH = ROOT / "data" / "processed"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("Loading Jezero Crater CTX DEM...")

if not DEM_PATH.exists():
    raise FileNotFoundError(f"DEM not found: {DEM_PATH}")

with rasterio.open(DEM_PATH) as src:

    print("\n===== JEZERO DEM INFORMATION =====")
    print("Width:", src.width)
    print("Height:", src.height)
    print("CRS:", src.crs)
    print("Pixel size:", src.res)
    print("NoData:", src.nodata)
    print("Bounds:", src.bounds)
    print("Data type:", src.dtypes[0])

    # Read elevation data
    elevation = src.read(1, masked=True)

    valid = elevation.compressed()

    if valid.size == 0:
        raise ValueError("The DEM contains no valid elevation values.")

    print("\n===== ELEVATION STATISTICS =====")
    print("Minimum:", round(float(valid.min()), 2), "m")
    print("Maximum:", round(float(valid.max()), 2), "m")
    print("Mean:", round(float(valid.mean()), 2), "m")

    # Create a smaller preview for display
    preview = src.read(
        1,
        masked=True,
        out_shape=(1400, 1400),
        resampling=Resampling.average
    )

    bounds = src.bounds

    extent = [
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ]

# Save elevation preview
plt.figure(figsize=(12, 8))

image = plt.imshow(
    preview,
    cmap="terrain",
    extent=extent,
    origin="upper"
)

plt.colorbar(image, label="Elevation (m)")
plt.title("Jezero Crater - CTX Digital Elevation Model")
plt.xlabel("Mars Easting (m)")
plt.ylabel("Mars Northing (m)")

plt.tight_layout()

preview_path = OUTPUT_PATH / "jezero_elevation_preview.png"
plt.savefig(preview_path, dpi=150)

print("\nPreview saved:", preview_path)

plt.show()
