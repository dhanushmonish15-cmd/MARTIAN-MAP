
from pathlib import Path
import heapq
import math
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

DEM = ROOT / "data" / "regional" / "jezero" / "M20_JezeroCrater_CTXDEM_20m.tif"
COST = DATA / "jezero_terrain_cost.npy"

FACTOR = 10

# Different route strategies
STRATEGIES = {
    "Shortest": 0.0,
    "Balanced": 4.0,
    "Terrain_Avoiding": 10.0
}


def astar(grid, start, goal, weight):

    rows, cols = grid.shape

    directions = [
        (-1, 0, 1.0), (1, 0, 1.0),
        (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2))
    ]

    queue = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}

    while queue:

        _, current = heapq.heappop(queue)

        if current == goal:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            return path[::-1]

        for dr, dc, movement in directions:

            nr = current[0] + dr
            nc = current[1] + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if not np.isfinite(grid[nr, nc]):
                continue

            # Prevent diagonal corner cutting
            if dr != 0 and dc != 0:
                if (
                    not np.isfinite(grid[current[0] + dr, current[1]])
                    or not np.isfinite(grid[current[0], current[1] + dc])
                ):
                    continue

            neighbour = (nr, nc)

            terrain = (grid[current] + grid[nr, nc]) / 2

            step_cost = movement * (1 + weight * terrain)

            new_cost = g_score[current] + step_cost

            if new_cost < g_score.get(neighbour, float("inf")):

                came_from[neighbour] = current
                g_score[neighbour] = new_cost

                heuristic = math.hypot(
                    nr - goal[0],
                    nc - goal[1]
                )

                heapq.heappush(
                    queue,
                    (new_cost + heuristic, neighbour)
                )

    return None


def calculate_distance(path, pixel_x, pixel_y):

    distance = 0.0

    for a, b in zip(path[:-1], path[1:]):

        dr = abs(b[0] - a[0]) * pixel_y
        dc = abs(b[1] - a[1]) * pixel_x

        distance += math.hypot(dr, dc)

    return distance


if not COST.exists():
    raise FileNotFoundError("Run terrain_cost_engine.py first.")

print("Loading Jezero terrain...")

with rasterio.open(DEM) as src:

    cost = np.load(COST)

    if cost.shape != (src.height, src.width):
        raise ValueError("Terrain cost and DEM dimensions differ.")

    grid = cost[::FACTOR, ::FACTOR].copy()

    pixel_x = abs(src.res[0]) * FACTOR
    pixel_y = abs(src.res[1]) * FACTOR

print("Route grid:", grid.shape)

# Temporary demonstration endpoints.
# Later, users will select these on the interactive map.
valid_counts = np.isfinite(grid).sum(axis=1)
start_row = int(np.argmax(valid_counts))

valid_columns = np.flatnonzero(np.isfinite(grid[start_row]))

if len(valid_columns) < 2:
    raise ValueError("Not enough valid cells.")

start = (
    start_row,
    int(valid_columns[int(len(valid_columns) * 0.10)])
)

goal = (
    start_row,
    int(valid_columns[int(len(valid_columns) * 0.90)])
)

results = {}

for name, weight in STRATEGIES.items():

    print(f"\nCalculating {name} route...")

    path = astar(grid, start, goal, weight)

    if path is None:
        print(name, "route not found.")
        continue

    distance = calculate_distance(path, pixel_x, pixel_y)

    terrain_values = [grid[r, c] for r, c in path]
    average_terrain = float(np.mean(terrain_values))

    results[name] = {
        "path": path,
        "distance": distance,
        "average_terrain": average_terrain
    }

    filename = DATA / f"jezero_{name.lower()}_route.csv"

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["row", "column"])

        writer.writerows(path)

    print("Distance:", round(distance, 2), "m")
    print("Average terrain cost:", round(average_terrain, 4))
    print("Saved:", filename)

if not results:
    raise RuntimeError("No routes could be calculated.")

# Display route comparison
plt.figure(figsize=(13, 9))

plt.imshow(
    grid,
    cmap="terrain",
    vmin=0,
    vmax=1
)

for name, result in results.items():

    path = result["path"]

    rows = [p[0] for p in path]
    cols = [p[1] for p in path]

    plt.plot(cols, rows, linewidth=2, label=name)

plt.scatter(start[1], start[0], marker="o", s=90, label="Start")
plt.scatter(goal[1], goal[0], marker="*", s=150, label="Destination")

plt.title("MARTIAN MAP - Three Route Strategies")
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")
plt.legend()
plt.colorbar(label="Relative terrain cost")

plt.tight_layout()

output = DATA / "jezero_route_comparison.png"
plt.savefig(output, dpi=150)
plt.show()

print("\n===== ROUTE COMPARISON COMPLETE =====")
print("Comparison map:", output)
