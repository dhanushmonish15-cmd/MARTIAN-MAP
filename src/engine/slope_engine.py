
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

ROOT = Path(__file__).resolve().parents[2]

DEM_PATH = ROOT / "data/regional/jezero/M20_JezeroCrater_CTXDEM_20m.tif"
OUTPUT_DIR = ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Jezero elevation data...")

with rasterio.open(DEM_PATH) as src:
    elevation = src.read(1, masked=True).astype("float32")
    pixel_x = abs(src.transform.a)
    pixel_y = abs(src.transform.e)

# Fill NoData temporarily to avoid false edges
data = elevation.filled(np.nan)
valid = np.isfinite(data)

# Calculate elevation gradients
filled = np.where(valid, data, np.nanmedian(data[valid]))

dz_dy = sobel(filled, axis=0, mode="nearest") / (8 * pixel_y)
dz_dx = sobel(filled, axis=1, mode="nearest") / (8 * pixel_x)

slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
slope[~valid] = np.nan

# Save slope data
slope_path = OUTPUT_DIR / "jezero_slope_degrees.npy"
np.save(slope_path, slope)

print("\n===== JEZERO SLOPE ANALYSIS =====")
print("Minimum slope:", round(float(np.nanmin(slope)), 2), "degrees")
print("Maximum slope:", round(float(np.nanmax(slope)), 2), "degrees")
print("Average slope:", round(float(np.nanmean(slope)), 2), "degrees")
print("Slope data saved:", slope_path)

# Display slope map
plt.figure(figsize=(12, 8))

image = plt.imshow(
    slope,
    cmap="inferno",
    vmin=0,
    vmax=35
)

plt.colorbar(image, label="Slope (degrees)")
plt.title("Jezero Crater - Terrain Slope Map")
plt.xlabel("Pixel East-West")
plt.ylabel("Pixel North-South")

plt.tight_layout()

preview_path = OUTPUT_DIR / "jezero_slope_preview.png"
plt.savefig(preview_path, dpi=150)
plt.show()

print("Slope preview saved:", preview_path)
