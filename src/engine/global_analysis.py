
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data" / "global" / "mars_mola_elevation.img"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Original MOLA dataset
ROWS = 2880
COLS = 5760
EXPECTED_BYTES = ROWS * COLS * 2

# Mars mean radius in metres
MARS_RADIUS = 3_396_200.0

if not DATA_FILE.exists():
    raise FileNotFoundError("MOLA elevation file not found.")

if DATA_FILE.stat().st_size != EXPECTED_BYTES:
    raise ValueError("MOLA file size is incorrect.")

print("Loading global MOLA data...")

raw = np.fromfile(DATA_FILE, dtype=">i2")
original = raw.reshape(ROWS, COLS)

# Reduce resolution by averaging 4 x 4 cells.
# Original: 16 pixels per degree.
# Output: 4 pixels per degree.

elevation = original.reshape(
    720, 4, 1440, 4
).astype(np.float32).mean(axis=(1, 3))

print("Reduced elevation shape:", elevation.shape)

# Save global elevation grid
elevation_path = OUTPUT_DIR / "mars_global_elevation_4ppd.npy"
np.save(elevation_path, elevation)

# ------------------------------------------------
# Calculate broad-scale global slope
# ------------------------------------------------

print("Calculating global slope...")

rows, cols = elevation.shape

pixel_angle = np.deg2rad(0.25)

# Latitude at the centre of each output cell
latitude = 90 - (np.arange(rows) + 0.5) * 0.25

# North-south distance between neighbouring cells
dy = MARS_RADIUS * pixel_angle

# East-west distance changes with latitude
dx = (
    MARS_RADIUS
    * np.cos(np.deg2rad(latitude))
    * pixel_angle
)

# Elevation gradient north-south
dz_dy = np.gradient(elevation, dy, axis=0)

# Circular east-west gradient (wrap around longitude)
dz_dx = (
    np.roll(elevation, -1, axis=1)
    - np.roll(elevation, 1, axis=1)
) / (2 * dx[:, None])

# Convert gradient to slope angle in degrees
gradient = np.hypot(dz_dx, dz_dy)

slope = np.degrees(np.arctan(gradient))

# Avoid unreliable longitude-based calculations near the poles
polar_rows = np.abs(latitude) > 85
slope[polar_rows, :] = np.nan

# Save global slope grid
slope_path = OUTPUT_DIR / "mars_global_slope_4ppd.npy"
np.save(slope_path, slope)

print("\n===== GLOBAL MARS ANALYSIS =====")
print("Elevation shape:", elevation.shape)
print("Slope shape:", slope.shape)

print(
    "Minimum elevation:",
    round(float(np.min(elevation)), 2), "m"
)

print(
    "Maximum elevation:",
    round(float(np.max(elevation)), 2), "m"
)

print(
    "Average elevation:",
    round(float(np.mean(elevation)), 2), "m"
)

print(
    "Average global slope:",
    round(float(np.nanmean(slope)), 2), "degrees"
)

print("\nSaved:")
print(elevation_path)
print(slope_path)

print("\nGLOBAL ANALYSIS COMPLETED!")
