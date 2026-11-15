
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

slope_path = DATA_DIR / "jezero_slope_degrees.npy"
roughness_path = DATA_DIR / "jezero_roughness.npy"

if not slope_path.exists():
    raise FileNotFoundError("Slope data missing. Run slope_engine.py first.")

if not roughness_path.exists():
    raise FileNotFoundError("Roughness data missing. Run roughness_engine.py first.")

print("Loading terrain layers...")

slope = np.load(slope_path)
roughness = np.load(roughness_path)

if slope.shape != roughness.shape:
    raise ValueError("Slope and roughness grids do not match.")

valid = np.isfinite(slope) & np.isfinite(roughness)

if not np.any(valid):
    raise ValueError("No valid terrain cells found.")

print("Normalising terrain values...")

# Use the 95th percentile to reduce the influence of extreme values.
slope_limit = np.percentile(slope[valid], 95)
roughness_limit = np.percentile(roughness[valid], 95)

if slope_limit <= 0 or roughness_limit <= 0:
    raise ValueError("Unexpected terrain values; check the source layers.")

slope_norm = np.clip(slope / slope_limit, 0, 1)
roughness_norm = np.clip(roughness / roughness_limit, 0, 1)

# Initial experimental weights, not validated mobility limits.
terrain_cost = (
    0.65 * slope_norm +
    0.35 * roughness_norm
)

terrain_cost[~valid] = np.nan

output_path = DATA_DIR / "jezero_terrain_cost.npy"
np.save(output_path, terrain_cost)

print("\n===== TERRAIN INTELLIGENCE =====")
print("Slope contribution: 65%")
print("Roughness contribution: 35%")
print("Average cost:", round(float(np.nanmean(terrain_cost)), 3))
print("Maximum cost:", round(float(np.nanmax(terrain_cost)), 3))
print("Saved:", output_path)

# Create a smaller preview
preview = terrain_cost[::4, ::4]

plt.figure(figsize=(12, 8))
image = plt.imshow(
    preview,
    cmap="RdYlGn_r",
    vmin=0,
    vmax=1
)

plt.colorbar(image, label="Relative terrain cost (0–1)")
plt.title("MARTIAN MAP - Jezero Terrain Intelligence")
plt.xlabel("Pixel East-West")
plt.ylabel("Pixel North-South")
plt.tight_layout()

preview_path = DATA_DIR / "jezero_terrain_cost_preview.png"
plt.savefig(preview_path, dpi=150)
plt.show()

print("Preview saved:", preview_path)
