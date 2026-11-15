
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data" / "global" / "mars_mola_elevation.img"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Official MOLA MEG016 dimensions
ROWS = 2880
COLS = 5760
EXPECTED_BYTES = ROWS * COLS * 2

# Verify the downloaded file
if not DATA_FILE.exists():
    raise FileNotFoundError(f"MOLA file missing: {DATA_FILE}")

actual_bytes = DATA_FILE.stat().st_size

print("MOLA file size:", actual_bytes)
print("Expected size:", EXPECTED_BYTES)

if actual_bytes != EXPECTED_BYTES:
    raise ValueError(
        "File size does not match the documented MOLA product. "
        "Check the downloaded file and label."
    )

# Read signed 16-bit big-endian elevation data
raw = np.fromfile(DATA_FILE, dtype=">i2")

# Convert into a 2D elevation grid
elevation = raw.reshape((ROWS, COLS)).astype(np.float32)
# Create reduced elevation grid for the global 3D globe
# Average every 16 x 16 elevation cells

globe_elevation = elevation.reshape(
    180, 16, 360, 16
).mean(axis=(1, 3)).astype(np.float32)

globe_file = OUTPUT_DIR / "mars_mola_globe_360x180.npy"

np.save(globe_file, globe_elevation)

print("Globe elevation shape:", globe_elevation.shape)
print("Globe elevation saved:", globe_file)

print("\n===== MARS GLOBAL ELEVATION =====")
print("Grid shape:", elevation.shape)
print("Minimum elevation:", np.min(elevation), "m")
print("Maximum elevation:", np.max(elevation), "m")
print("Average elevation:", round(float(np.mean(elevation)), 2), "m")

# Generate global elevation preview
plt.figure(figsize=(15, 7))

image = plt.imshow(
    elevation,
    cmap="terrain",
    extent=[0, 360, -90, 90],
    origin="upper",
    aspect="auto"
)

plt.title("MARS GLOBAL ELEVATION - MOLA")
plt.xlabel("East Longitude (degrees)")
plt.ylabel("Latitude (degrees)")

plt.colorbar(image, label="Elevation above MOLA areoid (metres)")

plt.tight_layout()

output_file = OUTPUT_DIR / "mars_global_elevation.png"

plt.savefig(output_file, dpi=180)

print("\nElevation preview saved to:")
print(output_file)

plt.show()
