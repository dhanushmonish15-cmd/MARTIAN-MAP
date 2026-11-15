
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]

DEM_PATH = ROOT / "data/regional/jezero/M20_JezeroCrater_CTXDEM_20m.tif"
OUTPUT_DIR = ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Jezero DEM...")

with rasterio.open(DEM_PATH) as src:
    elevation = src.read(1, masked=True).astype("float32")

data = elevation.filled(np.nan)
valid = np.isfinite(data)

print("Preparing elevation data...")

# Replace NoData values temporarily
filled = np.where(valid, data, 0).astype(np.float32)
valid_float = valid.astype(np.float32)

print("Calculating local mean...")

# Calculate local valid-cell count and mean
count = uniform_filter(valid_float, size=3, mode="constant") * 9
sum_values = uniform_filter(filled, size=3, mode="constant") * 9

mean = np.divide(
    sum_values,
    count,
    out=np.zeros_like(sum_values),
    where=count > 0
)

print("Calculating local variance...")

# Calculate E[z²] - E[z]²
squared = np.where(valid, data * data, 0).astype(np.float32)

sum_squares = uniform_filter(
    squared, size=3, mode="constant"
) * 9

mean_square = np.divide(
    sum_squares,
    count,
    out=np.zeros_like(sum_squares),
    where=count > 0
)

variance = mean_square - mean * mean
variance = np.maximum(variance, 0)

roughness = np.sqrt(variance)
roughness[~valid] = np.nan

print("Saving roughness data...")

output_file = OUTPUT_DIR / "jezero_roughness.npy"
np.save(output_file, roughness)

print("\n===== TERRAIN ROUGHNESS =====")
print("Minimum:", round(float(np.nanmin(roughness)), 3), "m")
print("Maximum:", round(float(np.nanmax(roughness)), 3), "m")
print("Average:", round(float(np.nanmean(roughness)), 3), "m")

print("Saved:", output_file)

print("Creating preview...")

# Downsample only for display
step = 4
preview = roughness[::step, ::step]

plt.figure(figsize=(12, 8))
image = plt.imshow(preview, cmap="magma")
plt.colorbar(image, label="Local elevation standard deviation (m)")
plt.title("Jezero Crater - Terrain Roughness Map")
plt.xlabel("Pixel East-West")
plt.ylabel("Pixel North-South")
plt.tight_layout()

preview_file = OUTPUT_DIR / "jezero_roughness_preview.png"
plt.savefig(preview_file, dpi=150)
plt.show()

print("Preview saved:", preview_file)
