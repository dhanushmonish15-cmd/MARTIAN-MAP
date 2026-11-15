"""Terrain analysis helpers for MARTIAN MAP.

The global map remains based on Viking MDIM imagery and MOLA MEGDR elevation.
This module uses the optional Jezero regional DEM-derived arrays when present.
The terrain cost is an experimental relative index, not a human/rover safety score.
"""
from __future__ import annotations

from pathlib import Path
from heapq import heappush, heappop
from math import hypot
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
JEZERO_DIR = BASE_DIR / "data" / "regional" / "jezero"

COST_PATH = PROCESSED_DIR / "jezero_terrain_cost.npy"
SLOPE_PATH = PROCESSED_DIR / "jezero_slope_degrees.npy"
ROUGHNESS_PATH = PROCESSED_DIR / "jezero_roughness.npy"


def derive_terrain_cost(slope: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    """Create the same experimental 65/35 relative cost index used previously."""
    slope = np.asarray(slope, dtype=np.float32)
    roughness = np.asarray(roughness, dtype=np.float32)
    if slope.shape != roughness.shape:
        raise ValueError("Slope and roughness grids do not match.")
    valid = np.isfinite(slope) & np.isfinite(roughness)
    if not np.any(valid):
        raise ValueError("No valid terrain cells found.")
    slope_limit = float(np.nanpercentile(slope[valid], 95))
    roughness_limit = float(np.nanpercentile(roughness[valid], 95))
    if slope_limit <= 0 or roughness_limit <= 0:
        raise ValueError("Unexpected terrain values; check the source layers.")
    slope_norm = np.clip(slope / slope_limit, 0.0, 1.0)
    roughness_norm = np.clip(roughness / roughness_limit, 0.0, 1.0)
    cost = (0.65 * slope_norm + 0.35 * roughness_norm).astype(np.float32)
    cost[~valid] = np.nan
    return cost


def load_jezero_layers(build_cost: bool = True):
    """Load optional Jezero slope/roughness/cost NPY files.

    Returns a dictionary. Missing layers are returned as None; the GUI can still
    run without regional files.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    slope = np.load(SLOPE_PATH) if SLOPE_PATH.exists() else None
    roughness = np.load(ROUGHNESS_PATH) if ROUGHNESS_PATH.exists() else None
    cost = np.load(COST_PATH) if COST_PATH.exists() else None
    if cost is None and build_cost and slope is not None and roughness is not None:
        cost = derive_terrain_cost(slope, roughness)
        np.save(COST_PATH, cost)
    return {"slope": slope, "roughness": roughness, "cost": cost}


def downsample_grid(array: np.ndarray, max_side: int = 600):
    """Return (downsampled_array, stride); preserves values by regular sampling."""
    arr = np.asarray(array)
    stride = max(1, int(np.ceil(max(arr.shape) / max_side)))
    return arr[::stride, ::stride], stride


def _nearest_valid(cost: np.ndarray, point: tuple[int, int], radius: int = 40):
    y, x = point
    h, w = cost.shape
    y = int(np.clip(y, 0, h - 1))
    x = int(np.clip(x, 0, w - 1))
    if np.isfinite(cost[y, x]):
        return y, x
    for r in range(1, radius + 1):
        y0, y1 = max(0, y-r), min(h, y+r+1)
        x0, x1 = max(0, x-r), min(w, x+r+1)
        patch = cost[y0:y1, x0:x1]
        ys, xs = np.where(np.isfinite(patch))
        if len(ys):
            idx = np.argmin((ys + y0-y)**2 + (xs + x0-x)**2)
            return int(ys[idx] + y0), int(xs[idx] + x0)
    raise ValueError("No valid terrain cell found near the selected point.")


def astar_grid(cost: np.ndarray, start_xy: tuple[int, int], goal_xy: tuple[int, int],
               max_expansions: int = 500_000):
    """A* on a downsampled terrain-cost grid.

    start_xy and goal_xy are (x, y) grid coordinates. Returns (path_yx, score).
    Diagonal corner-cutting is prevented. The score is a relative path cost.
    """
    grid = np.asarray(cost, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError("Terrain cost must be a 2D grid.")
    start = _nearest_valid(grid, (start_xy[1], start_xy[0]))
    goal = _nearest_valid(grid, (goal_xy[1], goal_xy[0]))
    if start == goal:
        return [start], 0.0

    h, w = grid.shape
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 2**0.5), (-1, 1, 2**0.5),
             (1, -1, 2**0.5), (1, 1, 2**0.5)]

    def heuristic(y, x):
        return hypot(goal[0] - y, goal[1] - x)

    open_heap = []
    heappush(open_heap, (heuristic(*start), 0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    expansions = 0

    while open_heap:
        _, current_g, current = heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, float(current_g)
        closed.add(current)
        expansions += 1
        if expansions > max_expansions:
            raise RuntimeError("Route search limit reached; choose closer points or use a coarser grid.")

        cy, cx = current
        for dy, dx, length in moves:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w) or not np.isfinite(grid[ny, nx]):
                continue
            if dy != 0 and dx != 0:
                if not (np.isfinite(grid[cy, nx]) and np.isfinite(grid[ny, cx])):
                    continue
            neighbour = (ny, nx)
            if neighbour in closed:
                continue
            edge = length * (1.0 + 4.0 * float((grid[cy, cx] + grid[ny, nx]) / 2.0))
            tentative = current_g + edge
            if tentative < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                heappush(open_heap, (tentative + heuristic(ny, nx), tentative, neighbour))

    raise RuntimeError("No route found through valid terrain cells.")


def path_length_km(path_yx, pixel_size_m: float = 20.0, stride: int = 1) -> float:
    """Approximate path length from raster pixel size and downsample stride."""
    if len(path_yx) < 2:
        return 0.0
    total = 0.0
    for (y0, x0), (y1, x1) in zip(path_yx[:-1], path_yx[1:]):
        total += hypot(x1 - x0, y1 - y0)
    return total * pixel_size_m * stride / 1000.0


def summarize(array: np.ndarray):
    valid = np.asarray(array)[np.isfinite(array)]
    if valid.size == 0:
        return None
    return {
        "min": float(np.min(valid)),
        "median": float(np.median(valid)),
        "mean": float(np.mean(valid)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
    }


if __name__ == "__main__":
    layers = load_jezero_layers(build_cost=True)
    if layers["cost"] is None:
        raise SystemExit("Jezero terrain arrays not found. Prepare slope and roughness NPY files first.")
    stats = summarize(layers["cost"])
    print("MARTIAN MAP — JEZERO TERRAIN COST")
    print("Relative index (0–1), not a validated mobility or safety score")
    print(f"Mean: {stats['mean']:.3f} | Median: {stats['median']:.3f} | P95: {stats['p95']:.3f}")
    print(f"Saved/loaded: {COST_PATH}")
