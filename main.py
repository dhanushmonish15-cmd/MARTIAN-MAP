"""MARTIAN MAP — Interplanetary Survival & Exploration Intelligence System

Hackathon-ready desktop prototype. Uses the two agreed global datasets:
  1) Viking Orbiter Global Mosaic (MDIM): global surface imagery
  2) MOLA MEGDR: global elevation/topography

Other live-looking dashboards are explicitly labelled simulations. Water/ice,
minerals, radiation, weather and atmospheric measurements are NOT inferred from
Viking imagery + MOLA elevation. MEDA values are shown only after importing an
archived PDS CSV. Orbital paths are schematic animation, not live ephemeris.
Optional Jezero terrain arrays are loaded only when present under data/processed/.
"""
from __future__ import annotations

import sys
import os
import math
import csv
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
except Exception:  # The GUI can open even if rasterio is unavailable.
    rasterio = None
    Resampling = None

from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, QObject, Signal, QTimer, QRectF, QUrl
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QLinearGradient, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QStackedWidget,
    QComboBox, QListWidget, QListWidgetItem, QLineEdit, QProgressBar, QSpinBox,
    QDoubleSpinBox, QScrollArea, QMessageBox, QSizePolicy, QSpacerItem,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView
)

from terrain_engine import (
    load_jezero_layers, downsample_grid, astar_grid, path_length_km, summarize
)


# ============================================================
# PROJECT PATHS AND PLANET CONSTANTS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
GLOBAL_DIR = BASE_DIR / "data" / "global"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGE_PATH = GLOBAL_DIR / "mars_viking_global_925m.tif"
GLOBE_ELEVATION_PATH = PROCESSED_DIR / "mars_mola_globe_360x180.npy"
GLOBAL_ELEVATION_PATH = PROCESSED_DIR / "mars_global_elevation_4ppd.npy"
GLOBAL_SLOPE_PATH = PROCESSED_DIR / "mars_global_slope_4ppd.npy"
MARS_MEAN_RADIUS_KM = 3389.5

# Coordinates are planetocentric latitude and east longitude in degrees.
# Landing-site coordinates follow the published landing-site table and NASA/NSSDC data.
MARS_PLACES = [
    {"name": "Jezero Crater", "mission": "Perseverance", "lat": 18.44, "lon": 77.50, "kind": "Rover landing site"},
    {"name": "Gale Crater", "mission": "Curiosity", "lat": -4.590, "lon": 137.442, "kind": "Rover landing site"},
    {"name": "Gusev Crater", "mission": "Spirit", "lat": -14.572, "lon": 175.478, "kind": "Rover landing site"},
    {"name": "Meridiani Planum", "mission": "Opportunity", "lat": -1.948, "lon": 354.474, "kind": "Rover landing site"},
    {"name": "Chryse Planitia", "mission": "Viking 1", "lat": 22.269, "lon": 312.051, "kind": "Lander site"},
    {"name": "Utopia Planitia", "mission": "Viking 2", "lat": 47.668, "lon": 134.284, "kind": "Lander site"},
    {"name": "Vastitas Borealis", "mission": "Phoenix", "lat": 68.219, "lon": 234.249, "kind": "Lander site"},
    {"name": "Elysium Planitia", "mission": "InSight", "lat": 4.498, "lon": 135.618, "kind": "Lander site"},
    {"name": "Ares Vallis", "mission": "Pathfinder", "lat": 19.098, "lon": 326.742, "kind": "Lander and rover site"},
    {"name": "Utopia Planitia (Zhurong)", "mission": "Zhurong", "lat": 24.80, "lon": 110.30, "kind": "Rover landing site"},
    {"name": "Olympus Mons", "mission": "Volcanic province", "lat": 18.65, "lon": 226.2, "kind": "Major landform"},
    {"name": "Valles Marineris", "mission": "Canyon system", "lat": -13.9, "lon": 296.0, "kind": "Major landform"},
    {"name": "Hellas Planitia", "mission": "Impact basin", "lat": -42.4, "lon": 70.5, "kind": "Major landform"},
    {"name": "Arcadia Planitia", "mission": "Northern plains", "lat": 49.02, "lon": 188.15, "kind": "Region of interest"},
]
PLACE_BY_NAME = {p["name"]: p for p in MARS_PLACES}

# Orbital asset status is based on the JPL Mars conjunction-assessment report
# dated 2026-11-11. Orbital geometry below is deliberately SCHEMATIC and is
# used only to animate labelled spacecraft around the globe; it is NOT a
# real-time ephemeris, not to scale, and not a navigation product.
ORBITAL_ASSETS = [
    {"name": "Mars Odyssey", "short": "ODYSSEY", "operator": "NASA", "status": "ACTIVE", "role": "Science mapping and relay support", "color": "#70c7ff", "radius": 1.30, "inclination": 32, "raan": 8, "phase": 0.2, "visual_rate": 0.90},
    {"name": "Mars Reconnaissance Orbiter", "short": "MRO", "operator": "NASA", "status": "ACTIVE", "role": "High-resolution imaging, climate science and relay support", "color": "#65e6d1", "radius": 1.34, "inclination": 18, "raan": 62, "phase": 1.1, "visual_rate": 1.10},
    {"name": "Mars Express", "short": "MARS EXPRESS", "operator": "ESA", "status": "ACTIVE", "role": "Atmospheric, surface and subsurface science", "color": "#ffca78", "radius": 1.39, "inclination": 42, "raan": 115, "phase": 2.0, "visual_rate": 0.78},
    {"name": "ExoMars Trace Gas Orbiter", "short": "TGO", "operator": "ESA", "status": "ACTIVE", "role": "Trace-gas science and communications relay", "color": "#b9a2ff", "radius": 1.43, "inclination": 24, "raan": 175, "phase": 0.8, "visual_rate": 1.02},
    {"name": "Emirates Mars Mission (Hope)", "short": "HOPE / EMM", "operator": "UAE Space Agency", "status": "ACTIVE", "role": "Upper-atmosphere and climate science", "color": "#ff9c83", "radius": 1.48, "inclination": 12, "raan": 220, "phase": 2.7, "visual_rate": 0.67},
    {"name": "Tianwen-1 Orbiter", "short": "TIANWEN-1", "operator": "CNSA", "status": "ACTIVE", "role": "Remote sensing, imaging and relay support", "color": "#d8e67a", "radius": 1.52, "inclination": 28, "raan": 285, "phase": 1.7, "visual_rate": 0.95},
]

# Natural satellites are shown separately from human-made spacecraft.
NATURAL_SATELLITES = [
    {"name": "Phobos", "short": "PHOBOS", "operator": "Natural moon of Mars", "status": "NATURAL SATELLITE", "role": "Inner moon; orbital path shown schematically", "color": "#d6d0c5", "radius": 1.24, "inclination": 4, "raan": 35, "phase": 1.4, "visual_rate": 1.85},
    {"name": "Deimos", "short": "DEIMOS", "operator": "Natural moon of Mars", "status": "NATURAL SATELLITE", "role": "Outer moon; orbital path shown schematically", "color": "#aaaeb7", "radius": 1.59, "inclination": 2, "raan": 92, "phase": 0.3, "visual_rate": 0.72},
]
ORBITAL_OBJECTS = ORBITAL_ASSETS + NATURAL_SATELLITES
MEDA_ARCHIVE_URL = "https://atmos.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_1740_1859/"
MEDA_INFO_URL = "https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/PERSEVERANCE/meda.html"
PDS_RELEASE_URL = "https://pds.nasa.gov/datasearch/subscription-service/SS-Release.shtml"
JPL_ORBITER_STATUS_URL = "https://cnex.jpl.nasa.gov/NASA/2026/11/11/reports/Mars-2026-11-11-21-45-log.html"


def great_circle_distance_km(point_a, point_b, radius_km=MARS_MEAN_RADIUS_KM):
    """Spherical great-circle distance between (planetocentric lat, east lon)."""
    lat1, lon1 = np.radians(point_a)
    lat2, lon2 = np.radians(point_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = float(np.clip(a, 0.0, 1.0))
    return float(radius_km * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def coordinates_text(point):
    if point is None:
        return "Not selected"
    return f"{point[0]:.3f}° N/S, {point[1] % 360:.3f}° E"


def add_panel(title, subtitle=None):
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(7)
    heading = QLabel(title.upper())
    heading.setObjectName("panelTitle")
    layout.addWidget(heading)
    if subtitle:
        text = QLabel(subtitle)
        text.setObjectName("muted")
        text.setWordWrap(True)
        layout.addWidget(text)
    return frame, layout


def metric_card(title, value, detail=""):
    frame = QFrame()
    frame.setObjectName("metricCard")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(3)
    t = QLabel(title.upper())
    t.setObjectName("metricTitle")
    v = QLabel(str(value))
    v.setObjectName("metricValue")
    d = QLabel(detail)
    d.setObjectName("muted")
    d.setWordWrap(True)
    layout.addWidget(t)
    layout.addWidget(v)
    if detail:
        layout.addWidget(d)
    return frame, v, d


# ============================================================
# SHARED APPLICATION STATE
# ============================================================
class MissionSelection(QObject):
    selectionChanged = Signal()
    routeChanged = Signal()
    activeChanged = Signal()

    def __init__(self):
        super().__init__()
        self.selected_point = None
        self.start_point = None
        self.destination_point = None
        self.active_target = None
        self.selected_name = "Unselected location"

    def arm(self, target):
        if target not in ("start", "destination"):
            return
        self.active_target = target
        self.activeChanged.emit()

    def set_selected(self, latitude, longitude, name="Selected location"):
        self.selected_point = (float(latitude), float(longitude) % 360.0)
        self.selected_name = str(name)
        self.selectionChanged.emit()

    def set_route_point(self, target, latitude, longitude, name=None):
        point = (float(latitude), float(longitude) % 360.0)
        if target == "start":
            self.start_point = point
        elif target == "destination":
            self.destination_point = point
        else:
            return
        self.active_target = None
        if name:
            self.selected_name = str(name)
            self.selected_point = point
        self.routeChanged.emit()
        self.selectionChanged.emit()
        self.activeChanged.emit()

    def accept_globe_click(self, latitude, longitude):
        if self.active_target:
            target = self.active_target
            self.set_route_point(target, latitude, longitude, f"Custom {target.title()} point")
        else:
            self.set_selected(latitude, longitude, "Custom selected location")

    def clear_route(self):
        self.start_point = None
        self.destination_point = None
        self.routeChanged.emit()


class MissionModel(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.report = None

    def simulate(self, crew, sols, supplies, rates):
        required = {
            "Oxygen": crew * sols * rates["Oxygen"],
            "Water": crew * sols * rates["Water"],
            "Food": crew * sols * rates["Food"],
            "Energy": crew * sols * rates["Energy"],
        }
        remaining = {key: supplies[key] - required[key] for key in required}
        adequate = {key: remaining[key] >= 0 for key in required}
        coverage = {
            key: (supplies[key] / required[key] * 100.0 if required[key] > 0 else 100.0)
            for key in required
        }
        self.report = {
            "crew": crew, "sols": sols, "supplies": dict(supplies),
            "rates": dict(rates), "required": required,
            "remaining": remaining, "adequate": adequate,
            "coverage": coverage,
            "feasible_under_assumptions": all(adequate.values()),
        }
        self.changed.emit()
        return self.report


# ============================================================
# 3D MARS GLOBE WITH NAMED LOCATIONS AND ROUTE OVERLAY
# ============================================================
class MarsGlobe(QWidget):
    def __init__(self, selection: MissionSelection, parent=None):
        super().__init__(parent)
        self.selection = selection
        self.figure = Figure(figsize=(7, 6), facecolor="#0b1420")
        self.canvas = FigureCanvas(self.figure)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self._route_artists = []
        self._place_artists = []
        self._orbital_lines = []
        self._orbital_artists = {}
        self._orbit_tick = 0.0
        self._loaded = False
        self._load_and_draw()
        if self._loaded:
            self._draw_orbital_layer()
            self.orbit_timer = QTimer(self)
            self.orbit_timer.setInterval(110)
            self.orbit_timer.timeout.connect(self._animate_orbital_assets)
            self.orbit_timer.start()
        self.canvas.mpl_connect("button_press_event", self.on_globe_click)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.selection.routeChanged.connect(self.draw_route)
        self.selection.selectionChanged.connect(self.draw_route)
    def on_scroll(self, event):
        """Zoom the Mars globe using the mouse scroll wheel."""
        if not self._loaded:
            return

        if event.button == "up":
            self.zoom_factor *= 0.85
        elif event.button == "down":
            self.zoom_factor *= 1.18

        self.zoom_factor = float(np.clip(self.zoom_factor, 0.35, 2.0))

        limit = 1.68 * self.zoom_factor

        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)

        self.canvas.draw_idle()
    def _load_and_draw(self):
        if rasterio is None:
            self._show_unavailable("Rasterio is not installed. Install requirements.txt to load the scientific globe.")
            return
        if not IMAGE_PATH.exists() or not GLOBE_ELEVATION_PATH.exists():
            missing = []
            if not IMAGE_PATH.exists():
                missing.append(str(IMAGE_PATH))
            if not GLOBE_ELEVATION_PATH.exists():
                missing.append(str(GLOBE_ELEVATION_PATH))
            self._show_unavailable("Required global dataset missing:\n" + "\n".join(missing))
            return
        try:
            with rasterio.open(IMAGE_PATH) as src:
                if src.count < 3:
                    raise ValueError("The Viking mosaic must contain at least 3 image bands.")
                bands = src.read(
                    [1, 2, 3], out_shape=(3, 180, 360),
                    resampling=Resampling.average, masked=True
                )
            image = np.ma.transpose(bands, (1, 2, 0)).filled(0).astype(np.float32)
            # Robustly normalise integer imagery for Matplotlib face colours.
            texture = np.zeros_like(image, dtype=np.float32)
            for band in range(3):
                plane = image[:, :, band]
                p2, p98 = np.nanpercentile(plane, [2, 98])
                if p98 <= p2:
                    texture[:, :, band] = np.clip(plane / 255.0, 0, 1)
                else:
                    texture[:, :, band] = np.clip((plane - p2) / (p98 - p2), 0, 1)
            texture = np.roll(texture, -180, axis=1)

            elevation = np.load(GLOBE_ELEVATION_PATH).astype(np.float32)
            if elevation.shape != (180, 360):
                raise ValueError(f"Expected MOLA globe array (180, 360), got {elevation.shape}.")
            elevation_vertices = np.asarray(
                Image.fromarray(elevation).resize((361, 181), Image.Resampling.BILINEAR),
                dtype=np.float32
            )
            radius = 1.0 + elevation_vertices / 3_396_000.0 * 10.0
            longitude = np.linspace(0, 2 * np.pi, 361)
            latitude = np.linspace(np.pi / 2, -np.pi / 2, 181)
            lon_grid, lat_grid = np.meshgrid(longitude, latitude)
            x = radius * np.cos(lat_grid) * np.cos(lon_grid)
            y = radius * np.cos(lat_grid) * np.sin(lon_grid)
            z = radius * np.sin(lat_grid)

            self.ax.clear()
            self.ax.plot_surface(
                x, y, z, facecolors=texture, rstride=2, cstride=2,
                linewidth=0, antialiased=False, shade=False
            )
            self._configure_axes()
            self._loaded = True
            self._draw_place_labels()
            self.figure.text(
                0.5, 0.025, "VIKING MDIM  •  MOLA MEGDR  •  ORBITAL TRACKS SCHEMATIC / NOT LIVE EPHEMERIS",
                color="#c7d4df", ha="center", fontsize=7.5
            )
            self.figure.subplots_adjust(left=0, right=1, bottom=0.04, top=1)
            self.canvas.draw_idle()
            print("3D Mars globe created from Viking MDIM + MOLA MEGDR.")
        except Exception as error:
            self._show_unavailable(f"Unable to load the global Mars datasets:\n{error}")

    def _configure_axes(self):
        # Extra room around Mars for spacecraft and natural-moon orbit tracks.
        self.ax.set_xlim(-1.68, 1.68)
        self.ax.set_ylim(-1.68, 1.68)
        self.ax.set_zlim(-1.68, 1.68)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_proj_type("ortho")
        self.ax.view_init(elev=20, azim=35)
        self.ax.set_axis_off()
        self.ax.set_facecolor("#0b1420")

    def _show_unavailable(self, message):
        self.ax.clear()
        self.ax.set_facecolor("#0b1420")
        self.ax.text2D(0.5, 0.57, "MARS GLOBE DATA UNAVAILABLE", transform=self.ax.transAxes,
                       color="#f1a06b", ha="center", fontsize=13, weight="bold")
        self.ax.text2D(0.5, 0.43, message, transform=self.ax.transAxes,
                       color="#d4dde6", ha="center", va="center", wrap=True, fontsize=8)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    @staticmethod
    def _unit_vector(point):
        lat, lon = np.radians(point)
        return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    def _draw_place_labels(self):
        if not self._loaded:
            return
        for artist in self._place_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._place_artists.clear()
        elev = np.radians(float(self.ax.elev))
        azim = np.radians(float(self.ax.azim))
        camera = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        for place in MARS_PLACES:
            vec = self._unit_vector((place["lat"], place["lon"]))
            visibility = float(np.dot(vec, camera))
            if visibility > 0.03:
                point = vec * 1.075
                marker = self.ax.scatter([point[0]], [point[1]], [point[2]],
                                         s=15 if "landing" in place["kind"].lower() else 10,
                                         c="#ffd27a" if "landing" in place["kind"].lower() else "#63e7e0",
                                         edgecolors="#15212b", linewidths=0.45, depthshade=False)
                label = self.ax.text(point[0] * 1.013, point[1] * 1.013, point[2] * 1.013,
                                     place["name"], color="#f4e7d1", fontsize=5.8,
                                     ha="center", va="center")
                self._place_artists.extend([marker, label])

    @staticmethod
    def _schematic_orbit_position(asset, angle):
        """Return an illustrative inclined-orbit position in normalized globe units."""
        radius = float(asset["radius"])
        inc = np.radians(float(asset["inclination"]))
        raan = np.radians(float(asset["raan"]))
        # Start with a circular orbit in its plane, incline it, then rotate the node.
        x0 = radius * np.cos(angle)
        y0 = radius * np.sin(angle) * np.cos(inc)
        z0 = radius * np.sin(angle) * np.sin(inc)
        x = x0 * np.cos(raan) - y0 * np.sin(raan)
        y = x0 * np.sin(raan) + y0 * np.cos(raan)
        return np.array([x, y, z0], dtype=float)

    def _draw_orbital_layer(self):
        """Draw schematic orbit paths and labelled spacecraft/moon markers."""
        if not self._loaded:
            return
        theta = np.linspace(0.0, 2.0 * np.pi, 220)
        for asset in ORBITAL_OBJECTS:
            points = np.array([self._schematic_orbit_position(asset, t) for t in theta])
            line, = self.ax.plot(
                points[:, 0], points[:, 1], points[:, 2],
                color=asset["color"], linewidth=0.75,
                alpha=0.36 if asset in ORBITAL_ASSETS else 0.48,
                linestyle="--" if asset in ORBITAL_ASSETS else ":",
                zorder=3,
            )
            pos = self._schematic_orbit_position(asset, float(asset["phase"]))
            marker = self.ax.scatter(
                [pos[0]], [pos[1]], [pos[2]],
                s=34 if asset in ORBITAL_ASSETS else 25,
                c=asset["color"], marker="o" if asset in ORBITAL_ASSETS else "D",
                edgecolors="#f2f5f7", linewidths=0.65, depthshade=False, zorder=30,
            )
            label_pos = pos * 1.035
            label = self.ax.text(
                label_pos[0], label_pos[1], label_pos[2], asset["short"],
                color=asset["color"], fontsize=5.1, fontweight="bold",
                ha="center", va="center", zorder=31,
            )
            self._orbital_lines.append(line)
            self._orbital_artists[asset["name"]] = (marker, label, asset)
        self.canvas.draw_idle()

    def _animate_orbital_assets(self):
        """Animate markers for visual orientation only; this is not ephemeris."""
        if not self._loaded or not self._orbital_artists:
            return
        self._orbit_tick += 0.035
        for marker, label, asset in self._orbital_artists.values():
            angle = float(asset["phase"]) + self._orbit_tick * float(asset["visual_rate"])
            pos = self._schematic_orbit_position(asset, angle)
            marker._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            label.set_position_3d(tuple(pos * 1.035))
        self.canvas.draw_idle()

    def _on_mouse_release(self, _event):
        if self._loaded:
            QTimer.singleShot(80, self._refresh_after_rotation)

    def _refresh_after_rotation(self):
        self._draw_place_labels()
        self.draw_route()
        self.canvas.draw_idle()

    def focus_location(self, place):
        if not self._loaded:
            return
        self.ax.view_init(elev=float(np.clip(place["lat"], -75, 75)), azim=float(place["lon"]))
        self._draw_place_labels()
        self.draw_route()
        self.canvas.draw_idle()

    def draw_route(self):
        if not self._loaded:
            return
        for artist in self._route_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._route_artists.clear()
        start, dest = self.selection.start_point, self.selection.destination_point
        if start is None or dest is None:
            self.canvas.draw_idle()
            return
        a, b = self._unit_vector(start), self._unit_vector(dest)
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        angle = float(np.arccos(dot))
        t = np.linspace(0, 1, 160)
        if angle < 1e-8:
            vectors = np.tile(a, (len(t), 1))
        elif abs(np.sin(angle)) < 1e-8:
            self.canvas.draw_idle()
            return
        else:
            vectors = (np.sin((1-t)*angle)[:, None] * a + np.sin(t*angle)[:, None] * b) / np.sin(angle)
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        route = vectors * 1.065
        line, = self.ax.plot(route[:, 0], route[:, 1], route[:, 2], color="#ff4fba", linewidth=2.2)
        self._route_artists.append(line)
        for point, color in ((start, "#53ef9a"), (dest, "#ff6969")):
            vec = self._unit_vector(point) * 1.078
            marker = self.ax.scatter([vec[0]], [vec[1]], [vec[2]], s=48, c=color,
                                     edgecolors="white", linewidths=0.8, depthshade=False)
            self._route_artists.append(marker)
        self.canvas.draw_idle()

    def on_globe_click(self, event):
        if not self._loaded or event.inaxes != self.ax or event.x is None or event.y is None:
            return
        try:
            bounds = self.ax.get_window_extent()
            if bounds.width <= 0 or bounds.height <= 0:
                return
            x_ndc = 2 * (event.x - bounds.x0) / bounds.width - 1
            y_ndc = 2 * (event.y - bounds.y0) / bounds.height - 1
            inverse = np.linalg.inv(self.ax.get_proj())
            near = inverse @ np.array([x_ndc, y_ndc, -1.0, 1.0])
            far = inverse @ np.array([x_ndc, y_ndc, 1.0, 1.0])
            if abs(near[3]) < 1e-12 or abs(far[3]) < 1e-12:
                return
            near, far = near[:3] / near[3], far[:3] / far[3]
            direction = far - near
            norm = np.linalg.norm(direction)
            if norm == 0:
                return
            direction /= norm
            b = float(np.dot(near, direction))
            c = float(np.dot(near, near) - 1.0)
            disc = b*b - c
            if disc < 0:
                return
            distance = -b - np.sqrt(disc)
            if distance < 0:
                distance = -b + np.sqrt(disc)
            if distance < 0:
                return
            point = near + distance * direction
            latitude = float(np.degrees(np.arcsin(np.clip(point[2], -1, 1))))
            longitude = float(np.degrees(np.arctan2(point[1], point[0])) % 360.0)
            self.selection.accept_globe_click(latitude, longitude)
        except Exception as error:
            print("Globe selection error:", error)


# ============================================================
# COMMAND CENTRE — MAIN LANDING PAGE + NAMED PLACE PANEL
# ============================================================
class CommandCentrePage(QWidget):
    """Landing page: planetary globe, searchable named places and mission endpoints."""

    def __init__(self, selection: MissionSelection, parent=None):
        super().__init__(parent)
        self.selection = selection
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(10)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("PLANETARY COMMAND CENTRE")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Explore Mars • inspect known sites • prepare mission endpoints")
        subtitle.setObjectName("muted")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch()
        self.header_status = QLabel("PLANETARY DATA READY")
        self.header_status.setObjectName("statusPill")
        header.addWidget(self.header_status, 0, Qt.AlignVCenter)
        outer.addLayout(header)

        cards = QGridLayout()
        cards.setHorizontalSpacing(9)
        cards.setVerticalSpacing(8)
        card_specs = [
            ("GLOBAL FOUNDATION", "VIKING + MOLA", "Imagery and elevation"),
            ("NAMED LOCATIONS", str(len(MARS_PLACES)), "Landing sites and landforms"),
            ("ORBITAL ASSETS", "6 ACTIVE", "2 natural moons • schematic orbits"),
            ("DATA INTEGRITY", "SOURCE-AWARE", "Unavailable data stays marked"),
        ]
        for col, (name, value, detail) in enumerate(card_specs):
            card, _, _ = metric_card(name, value, detail)
            card.setMinimumHeight(84)
            cards.addWidget(card, 0, col)
        outer.addLayout(cards)

        body = QHBoxLayout()
        body.setSpacing(10)
        globe_panel, globe_panel_layout = add_panel(
            "MARS GLOBAL VIEW",
            "Rotate the globe. Select a named site from the right panel, or arm a route endpoint in Mars NAV."
        )
        self.globe = MarsGlobe(selection)
        self.globe.setMinimumHeight(420)
        globe_panel_layout.addWidget(self.globe, 1)
        body.addWidget(globe_panel, 7)

        side, side_layout = add_panel(
            "KNOWN MARS LOCATIONS",
            "Search and select a landing site, major landform or region of interest."
        )
        self.place_search = QLineEdit()
        self.place_search.setObjectName("searchField")
        self.place_search.setPlaceholderText("Search Mars locations…")
        side_layout.addWidget(self.place_search)

        self.place_list = QListWidget()
        self.place_list.setObjectName("placeList")
        self.place_list.setMinimumWidth(250)
        self.place_list.setAlternatingRowColors(False)
        for place in MARS_PLACES:
            item = QListWidgetItem(f"{place['name']}  ·  {place['mission']}")
            item.setData(Qt.UserRole, place["name"])
            self.place_list.addItem(item)
        side_layout.addWidget(self.place_list, 1)

        self.place_info = QLabel("Choose a location to view its coordinates and mission context.")
        self.place_info.setObjectName("locationInfo")
        self.place_info.setWordWrap(True)
        self.place_info.setMinimumHeight(105)
        side_layout.addWidget(self.place_info)

        action_row = QHBoxLayout()
        self.focus_btn = QPushButton("FOCUS GLOBE")
        self.focus_btn.setObjectName("primaryButton")
        self.start_btn = QPushButton("SET START")
        self.start_btn.setObjectName("secondaryButton")
        self.dest_btn = QPushButton("SET DESTINATION")
        self.dest_btn.setObjectName("secondaryButton")
        action_row.addWidget(self.focus_btn)
        side_layout.addLayout(action_row)
        endpoint_row = QHBoxLayout()
        endpoint_row.addWidget(self.start_btn)
        endpoint_row.addWidget(self.dest_btn)
        side_layout.addLayout(endpoint_row)

        self.route_summary = QLabel("ROUTE ENDPOINTS  ·  Start: not set  /  Destination: not set")
        self.route_summary.setObjectName("routeSummary")
        side_layout.addWidget(self.route_summary)
        body.addWidget(side, 3)
        outer.addLayout(body, 1)

        self.place_search.textChanged.connect(self.filter_places)
        self.place_list.currentRowChanged.connect(self.update_place_info)
        self.focus_btn.clicked.connect(self.focus_selected)
        self.start_btn.clicked.connect(lambda: self.set_selected_as("start"))
        self.dest_btn.clicked.connect(lambda: self.set_selected_as("destination"))
        self.selection.routeChanged.connect(self.update_route_summary)
        self.selection.selectionChanged.connect(self.update_route_summary)
        if MARS_PLACES:
            self.place_list.setCurrentRow(0)
        self.update_route_summary()

    def filter_places(self, query):
        query = query.strip().casefold()
        for index in range(self.place_list.count()):
            item = self.place_list.item(index)
            item.setHidden(query not in item.text().casefold())

    def current_place(self):
        item = self.place_list.currentItem()
        if item is None:
            return None
        return PLACE_BY_NAME.get(item.data(Qt.UserRole))

    def update_place_info(self, _row):
        place = self.current_place()
        if not place:
            self.place_info.setText("No matching location selected.")
            return
        self.place_info.setText(
            f"{place['kind']}\nMission / feature: {place['mission']}\n"
            f"Latitude: {place['lat']:.3f}°\nEast longitude: {place['lon']:.3f}°\n\n"
            "Coordinates identify the named location; local measurements depend on available datasets."
        )

    def focus_selected(self):
        place = self.current_place()
        if place:
            self.selection.set_selected(place["lat"], place["lon"], place["name"])
            self.globe.focus_location(place)
            self.header_status.setText(f"FOCUSED  ·  {place['name'].upper()}")

    def set_selected_as(self, target):
        place = self.current_place()
        if place:
            self.selection.set_route_point(target, place["lat"], place["lon"], place["name"])
            self.header_status.setText(f"{target.upper()} SET  ·  {place['name'].upper()}")

    def update_route_summary(self):
        start = self.selection.start_point
        dest = self.selection.destination_point
        start_text = "not set" if start is None else f"{start[0]:.1f}°, {start[1]:.1f}°E"
        dest_text = "not set" if dest is None else f"{dest[0]:.1f}°, {dest[1]:.1f}°E"
        self.route_summary.setText(
            f"ROUTE ENDPOINTS  ·  Start: {start_text}  /  Destination: {dest_text}"
        )


# ============================================================
# ORBITAL ASSETS — ACTIVE SPACECRAFT AND NATURAL MOONS
# ============================================================
class OrbitalAssetsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setSpacing(10)
        title = QLabel("ORBITAL ASSETS & RELAY NETWORK")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        intro = QLabel(
            "Named Mars orbiters and natural moons. Their markers move on the Command Centre globe, "
            "but the displayed paths and animation are schematic—not live spacecraft ephemeris."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        root.addWidget(intro)

        source_row = QHBoxLayout()
        source = QLabel(
            "STATUS BASIS: JPL Mars conjunction-assessment report • 11 NOV 2026 • "
            "6 active spacecraft + 2 natural satellites"
        )
        source.setObjectName("statusPill")
        source_row.addWidget(source, 1)
        source_button = QPushButton("OPEN JPL STATUS SOURCE")
        source_button.setObjectName("secondaryButton")
        source_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(JPL_ORBITER_STATUS_URL)))
        source_row.addWidget(source_button)
        root.addLayout(source_row)

        body = QHBoxLayout()
        body.setSpacing(12)
        left, left_layout = add_panel("ACTIVE ORBITERS", "International spacecraft supporting science and/or communications at Mars.")
        self.asset_list = QListWidget()
        self.asset_list.setObjectName("placeList")
        for asset in ORBITAL_ASSETS:
            item = QListWidgetItem(f"{asset['short']}   ·   {asset['operator']}")
            item.setData(Qt.UserRole, asset["name"])
            self.asset_list.addItem(item)
        left_layout.addWidget(self.asset_list, 1)
        body.addWidget(left, 5)

        right, right_layout = add_panel("ASSET BRIEF", "Select a spacecraft to inspect its role and status.")
        self.asset_name = QLabel("MARS ODYSSEY")
        self.asset_name.setObjectName("metricValue")
        self.asset_operator = QLabel("NASA")
        self.asset_operator.setObjectName("muted")
        self.asset_role = QLabel("Science mapping and relay support")
        self.asset_role.setObjectName("locationInfo")
        self.asset_role.setWordWrap(True)
        self.asset_status = QLabel("ACTIVE IN JPL ASSESSMENT")
        self.asset_status.setObjectName("statusPill")
        right_layout.addWidget(self.asset_name)
        right_layout.addWidget(self.asset_operator)
        right_layout.addWidget(self.asset_status)
        right_layout.addWidget(self.asset_role)
        right_layout.addSpacing(12)
        moon_note = QLabel(
            "NATURAL SATELLITES\n"
            "PHOBOS  ·  inner moon\n"
            "DEIMOS  ·  outer moon\n\n"
            "PAST MISSION NOTE\n"
            "MAVEN is not shown as active. NASA reported loss of signal on 6 Dec 2025 and later declared the mission unrecoverable."
        )
        moon_note.setObjectName("locationInfo")
        moon_note.setWordWrap(True)
        right_layout.addWidget(moon_note)
        right_layout.addStretch(1)
        body.addWidget(right, 4)
        root.addLayout(body, 1)

        note = QLabel(
            "VISUAL HONESTY: orbital radii, inclinations, phases and animation speeds are illustrative only. "
            "Use NAIF SPICE kernels or verified ephemeris for precise spacecraft positions."
        )
        note.setObjectName("warningText")
        note.setWordWrap(True)
        root.addWidget(note)
        self.asset_list.currentRowChanged.connect(self.show_asset)
        self.asset_list.setCurrentRow(0)

    def show_asset(self, row):
        if row < 0 or row >= self.asset_list.count():
            return
        name = self.asset_list.item(row).data(Qt.UserRole)
        asset = next((item for item in ORBITAL_ASSETS if item["name"] == name), None)
        if not asset:
            return
        self.asset_name.setText(asset["name"].upper())
        self.asset_operator.setText(asset["operator"])
        self.asset_role.setText(asset["role"])
        self.asset_status.setText("ACTIVE IN JPL ASSESSMENT • 11 NOV 2026")


# ============================================================
# MEDA OBSERVATIONS — IMPORTABLE ARCHIVED MEASUREMENTS
# ============================================================
class MEDAObservationsPage(QWidget):
    """Explore archived MEDA ATS engineering observations from local PDS CSV files."""

    CHANNELS = [f"ATS_ATS{i}" for i in range(1, 6)]
    DATA_FILES = {
        "Sol 1": "WE__0001___________ESD_ATS_________________J02.csv",
        "Sol 1740": "WE__1740___________ESD_ATS_________________J01.csv",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.datasets = {}
        self.data_dir = BASE_DIR / "data" / "meda"

        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("MEDA | OBSERVATION LAB")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        intro = QLabel(
            "Explore archived Mars Environmental Dynamics Analyzer observations from the Perseverance rover. "
            "The charts compare measurements recorded at the rover site; they are not a live or planet-wide weather feed."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        root.addWidget(intro)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("DATASET"))
        self.dataset_selector = QComboBox()
        self.dataset_selector.addItem("Compare Sol 1 + Sol 1740", "compare")
        controls.addWidget(self.dataset_selector, 2)

        controls.addWidget(QLabel("ATS CHANNEL"))
        self.channel_selector = QComboBox()
        for channel in self.CHANNELS:
            self.channel_selector.addItem(channel, channel)
        controls.addWidget(self.channel_selector, 1)

        self.open_archive = QPushButton("OPEN MEDA PDS ARCHIVE")
        self.open_archive.setObjectName("secondaryButton")
        self.open_info = QPushButton("INSTRUMENT DETAILS")
        self.open_info.setObjectName("secondaryButton")
        self.load_csv = QPushButton("IMPORT ANOTHER CSV")
        self.load_csv.setObjectName("primaryButton")
        controls.addWidget(self.open_archive)
        controls.addWidget(self.open_info)
        controls.addWidget(self.load_csv)
        root.addLayout(controls)

        self.file_status = QLabel("SCANNING LOCAL MEDA ARCHIVE FILES...")
        self.file_status.setObjectName("statusPill")
        self.file_status.setWordWrap(True)
        root.addWidget(self.file_status)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("muted")
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        self.figure = Figure(figsize=(9, 3.4), dpi=100, facecolor="#0b1420")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(285)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas, 3)

        table_title = QLabel("OBSERVATION RECORD PREVIEW")
        table_title.setObjectName("sectionTitle")
        root.addWidget(table_title)

        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(215)
        root.addWidget(self.table, 2)

        note = QLabel(
            "SCIENCE NOTE: ATS_ATS1-ATS_ATS5 are displayed as raw engineering counts from the selected ESD_ATS files. "
            "They are NOT degrees Celsius. A calibrated temperature conversion requires the correct PDS calibration product "
            "and instrument documentation. PDS observations are archived measurements, not live telemetry."
        )
        note.setObjectName("warningText")
        note.setWordWrap(True)
        root.addWidget(note)

        self.open_archive.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(MEDA_ARCHIVE_URL)))
        self.open_info.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(MEDA_INFO_URL)))
        self.load_csv.clicked.connect(self.import_csv)
        self.dataset_selector.currentIndexChanged.connect(self.refresh_dashboard)
        self.channel_selector.currentIndexChanged.connect(self.refresh_dashboard)

        self.load_bundled_datasets()
        self.refresh_dashboard()

    @staticmethod
    def _parse_ltst_hours(value):
        """Convert a CSV LTST clock such as '0001 15:27:28' to decimal hours."""
        try:
            clock = str(value).strip().split()[-1]
            hour, minute, second = clock.split(":")
            return int(hour) + int(minute) / 60.0 + float(second) / 3600.0
        except (ValueError, IndexError, AttributeError):
            return np.nan

    def read_ats_csv(self, path):
        path = Path(path)
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(handle, dialect=dialect)
            headers = reader.fieldnames or []
            if not headers:
                raise ValueError("The selected CSV has no header row.")

            missing = [name for name in self.CHANNELS if name not in headers]
            if missing:
                raise ValueError("This file does not contain all five ATS channels: " + ", ".join(missing))

            times = []
            channel_values = {name: [] for name in self.CHANNELS}
            preview_rows = []
            first_ltst = ""
            last_ltst = ""
            record_count = 0

            for row in reader:
                if not row or not any(str(value or "").strip() for value in row.values()):
                    continue
                record_count += 1
                ltst = str(row.get("LTST", "") or "").strip()
                if ltst:
                    if not first_ltst:
                        first_ltst = ltst
                    last_ltst = ltst
                times.append(self._parse_ltst_hours(ltst))

                for channel in self.CHANNELS:
                    raw_value = str(row.get(channel, "") or "").strip()
                    try:
                        channel_values[channel].append(float(raw_value))
                    except ValueError:
                        channel_values[channel].append(np.nan)

                if len(preview_rows) < 45:
                    preview_rows.append([str(row.get(header, "") or "") for header in headers])

        if record_count == 0:
            raise ValueError("No observation records were found in this CSV.")

        x_values = np.asarray(times, dtype=float)
        has_ltst = bool(np.isfinite(x_values).any())
        if not has_ltst:
            x_values = np.arange(record_count, dtype=float)

        return {
            "path": path,
            "headers": headers,
            "count": record_count,
            "x": x_values,
            "has_ltst": has_ltst,
            "channels": {key: np.asarray(values, dtype=float) for key, values in channel_values.items()},
            "preview": preview_rows,
            "first_ltst": first_ltst or "not available",
            "last_ltst": last_ltst or "not available",
        }

    def load_bundled_datasets(self):
        loaded = []
        missing = []
        for label, filename in self.DATA_FILES.items():
            path = self.data_dir / filename
            if not path.exists():
                missing.append(filename)
                continue
            try:
                self.datasets[label] = self.read_ats_csv(path)
                loaded.append(f"{label}: {self.datasets[label]['count']:,} records")
                self.dataset_selector.addItem(label, label)
            except Exception as error:
                missing.append(f"{filename} ({error})")

        if loaded:
            self.file_status.setText("ARCHIVED MEDA DATA READY  |  " + "  |  ".join(loaded))
        else:
            self.file_status.setText("NO LOCAL MEDA DATA LOADED  |  Use IMPORT ANOTHER CSV or check data/meda")

        if missing:
            self.summary_label.setText("Not loaded: " + " | ".join(missing))

    def _selected_dataset_names(self):
        choice = self.dataset_selector.currentData()
        if choice == "compare":
            return [name for name in ("Sol 1", "Sol 1740") if name in self.datasets]
        if choice in self.datasets:
            return [choice]
        return list(self.datasets.keys())[:1]

    def refresh_dashboard(self, *_):
        channel = self.channel_selector.currentData() or self.CHANNELS[0]
        selected_names = self._selected_dataset_names()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.patch.set_facecolor("#0b1420")
        ax.set_facecolor("#101c2a")
        ax.tick_params(colors="#d6e2ef", labelsize=8)
        ax.xaxis.label.set_color("#d6e2ef")
        ax.yaxis.label.set_color("#d6e2ef")
        ax.title.set_color("#f3f6fb")
        for spine in ax.spines.values():
            spine.set_color("#40536b")
        ax.grid(True, color="#40536b", alpha=0.38, linewidth=0.7)
        ax.set_title(f"{channel} | ARCHIVED ATS OBSERVATIONS", fontsize=11, pad=12, fontweight="bold")
        ax.set_ylabel("Raw engineering counts (uncalibrated)", fontsize=8)

        line_colors = {"Sol 1": "#ff855e", "Sol 1740": "#64d8ed"}
        plotted = 0
        summary_parts = []
        for name in selected_names:
            data = self.datasets[name]
            values = data["channels"][channel]
            x_values = data["x"]
            valid = np.isfinite(values) & np.isfinite(x_values)
            if not valid.any():
                continue

            x_plot = x_values[valid]
            y_plot = values[valid]
            if len(x_plot) > 2200:
                sample_idx = np.linspace(0, len(x_plot) - 1, 2200).astype(int)
                x_plot = x_plot[sample_idx]
                y_plot = y_plot[sample_idx]

            ax.plot(
                x_plot,
                y_plot,
                label=f"{name}  |  {data['count']:,} records",
                color=line_colors.get(name, "#b9a2ff"),
                linewidth=1.25,
                alpha=0.95,
            )
            plotted += 1
            summary_parts.append(
                f"{name}: {data['count']:,} records; LTST {data['first_ltst']} to {data['last_ltst']}"
            )

        if plotted:
            if all(self.datasets[name]["has_ltst"] for name in selected_names if name in self.datasets):
                ax.set_xlabel("Local True Solar Time (hours)", fontsize=8)
                ax.set_xlim(0, 24)
            else:
                ax.set_xlabel("Observation record sequence", fontsize=8)
            ax.legend(loc="best", facecolor="#152337", edgecolor="#40536b", labelcolor="#e7eef7", fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "No ATS observations loaded\nCheck data/meda or import a CSV",
                transform=ax.transAxes, ha="center", va="center", color="#d6e2ef", fontsize=11,
            )
            ax.set_xlabel("Local True Solar Time (hours)", fontsize=8)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()
        self.summary_label.setText("   |   ".join(summary_parts) if summary_parts else "No dataset is currently selected.")
        self.refresh_table(selected_names[0] if selected_names else None)

    def refresh_table(self, dataset_name):
        self.table.clear()
        if not dataset_name or dataset_name not in self.datasets:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        data = self.datasets[dataset_name]
        headers = data["headers"]
        rows = data["preview"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                self.table.setItem(row_index, column_index, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a NASA/PDS MEDA ATS CSV file", str(self.data_dir), "CSV files (*.csv);;All files (*.*)"
        )
        if not path:
            return
        try:
            data = self.read_ats_csv(path)
            base_name = Path(path).stem
            label = base_name
            suffix = 2
            while label in self.datasets:
                label = f"{base_name} ({suffix})"
                suffix += 1
            self.datasets[label] = data
            self.dataset_selector.addItem(label, label)
            self.dataset_selector.setCurrentIndex(self.dataset_selector.count() - 1)
            self.file_status.setText(
                f"IMPORTED ARCHIVED ATS DATA  |  {Path(path).name}  |  {data['count']:,} records"
            )
            self.refresh_dashboard()
        except Exception as error:
            QMessageBox.warning(self, "MEDA CSV import failed", str(error))


# ============================================================
# SURVIVAL INTELLIGENCE — EDUCATIONAL GUIDANCE WITH DATA GAPS
# ============================================================

class SurvivalDecisionCard(QFrame):
    """Expandable evidence card: evidence, meaning, crew action and limits."""

    TONES = {
        "available": ("#66e1c8", "#123c3a", "#1d6c62"),
        "partial": ("#ffc46e", "#43311e", "#8a6232"),
        "missing": ("#ff8d91", "#45252c", "#8b4650"),
        "info": ("#83caff", "#1c344b", "#416e91"),
    }

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("decisionCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._tone = "info"

        root = QVBoxLayout(self)
        root.setContentsMargins(15, 14, 15, 13)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.heading = QLabel(title.upper())
        self.heading.setObjectName("decisionHeading")
        self.status = QLabel("CHECKING")
        self.status.setObjectName("decisionStatus")
        self.status.setAlignment(Qt.AlignCenter)
        top.addWidget(self.heading, 1)
        top.addWidget(self.status, 0, Qt.AlignRight)
        root.addLayout(top)

        self.summary = QLabel("Checking project data…")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("decisionSummary")
        root.addWidget(self.summary)

        action_frame = QFrame()
        action_frame.setObjectName("decisionAction")
        action_layout = QVBoxLayout(action_frame)
        action_layout.setContentsMargins(10, 8, 10, 8)
        action_layout.setSpacing(3)
        action_label = QLabel("CREW / MISSION ACTION")
        action_label.setObjectName("decisionActionLabel")
        self.action = QLabel("Review the available evidence before making a mission decision.")
        self.action.setWordWrap(True)
        self.action.setObjectName("decisionActionText")
        action_layout.addWidget(action_label)
        action_layout.addWidget(self.action)
        root.addWidget(action_frame)

        self.detail_button = QPushButton("＋  VIEW SCIENTIFIC EVIDENCE & LIMITATIONS")
        self.detail_button.setObjectName("evidenceButton")
        self.detail_button.setCheckable(True)
        root.addWidget(self.detail_button)

        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setObjectName("decisionDetails")
        self.details.hide()
        root.addWidget(self.details)
        self.detail_button.toggled.connect(self._toggle_details)
        self.set_content("CHECKING", "Checking project data…", "Wait for the initial data scan.", "", "info")

    def _toggle_details(self, expanded):
        self.details.setVisible(expanded)
        self.detail_button.setText(
            "−  HIDE SCIENTIFIC EVIDENCE" if expanded
            else "＋  VIEW SCIENTIFIC EVIDENCE & LIMITATIONS"
        )

    def set_content(self, status, summary, action, details, tone="info"):
        self.status.setText(str(status).upper())
        self.summary.setText(str(summary))
        self.action.setText(str(action))
        self.details.setText(str(details))
        self._tone = tone if tone in self.TONES else "info"
        accent, tint, border = self.TONES[self._tone]
        self.status.setStyleSheet(
            f"color:{accent}; background:{tint}; border:1px solid {border};"
            "border-radius:8px; padding:5px 8px; font-size:8px; font-weight:900;"
        )
        self.setStyleSheet(
            "QFrame#decisionCard {"
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #182b40, stop:0.58 #111d2d, stop:1 #101827);"
            f"border:1px solid {border}; border-radius:13px; }}"
        )


class SurvivalGuidePage(QWidget):
    """Evidence-first survival decision dashboard; never presents a safety certification."""

    def __init__(self, selection=None, parent=None):
        super().__init__(parent)
        self.selection = selection
        self.metrics = {}
        self.decision_cards = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(3, 3, 3, 3)
        root.setSpacing(10)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("SURVIVAL INTELLIGENCE")
        title.setObjectName("pageTitle")
        subtitle = QLabel("EVIDENCE → ASSESSMENT → CREW ACTION → MISSION PREPARATION")
        subtitle.setObjectName("muted")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)
        self.refresh_btn = QPushButton("⟳  REFRESH DATA")
        self.refresh_btn.setObjectName("primaryButton")
        header.addWidget(self.refresh_btn)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(2, 2, 7, 8)
        page.setSpacing(12)

        # Hero: readiness of the evidence, deliberately not a safety score.
        hero = QFrame()
        hero.setObjectName("survivalHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(20, 17, 20, 17)
        hero_layout.setSpacing(9)
        hero_top = QHBoxLayout()
        hero_title = QLabel("MISSION EVIDENCE STATUS")
        hero_title.setObjectName("heroTitle")
        self.readiness_badge = QLabel("SCANNING SCIENTIFIC DATA")
        self.readiness_badge.setObjectName("heroBadge")
        hero_top.addWidget(hero_title, 1)
        hero_top.addWidget(self.readiness_badge)
        hero_layout.addLayout(hero_top)
        self.readiness_text = QLabel(
            "The system is checking which environmental and terrain datasets are actually present."
        )
        self.readiness_text.setWordWrap(True)
        self.readiness_text.setObjectName("heroText")
        hero_layout.addWidget(self.readiness_text)
        progress_row = QHBoxLayout()
        self.coverage_bar = QProgressBar()
        self.coverage_bar.setRange(0, 100)
        self.coverage_bar.setValue(0)
        self.coverage_bar.setFormat("SCIENTIFIC DATA COVERAGE  |  %p%")
        self.coverage_label = QLabel("0 / 6 domains")
        self.coverage_label.setObjectName("heroCount")
        progress_row.addWidget(self.coverage_bar, 1)
        progress_row.addWidget(self.coverage_label)
        hero_layout.addLayout(progress_row)
        note = QLabel(
            "IMPORTANT: this is a data-coverage indicator, NOT a human-safety score, habitability rating or GO/NO-GO decision."
        )
        note.setObjectName("heroWarning")
        note.setWordWrap(True)
        hero_layout.addWidget(note)
        page.addWidget(hero)

        # Concise top-level mission decision.
        summary_row = QGridLayout()
        summary_row.setSpacing(9)
        for col, (key, label, value, detail) in enumerate([
            ("surface", "GLOBAL SURFACE CONTEXT", "VIKING MDIM", "Surface imagery only"),
            ("topography", "GLOBAL TOPOGRAPHY", "MOLA MEGDR", "Elevation / terrain context"),
            ("regional", "LOCAL TERRAIN", "JEZERO", "Regional arrays only"),
        ]):
            frame, val, desc = metric_card(label, value, detail)
            frame.setObjectName("metricCard")
            frame.setMinimumHeight(82)
            self.metrics[key] = (val, desc)
            summary_row.addWidget(frame, 0, col)
        page.addLayout(summary_row)

        # Six evidence domains.
        evidence_panel, evidence_layout = add_panel(
            "01  SCIENTIFIC EVIDENCE BOARD",
            "Every card separates what the loaded data can support from what it cannot establish. Expand a card for source and limitations."
        )
        grid = QGridLayout()
        grid.setSpacing(10)
        definitions = [
            ("viking", "Global surface imagery"),
            ("mola", "Global elevation / topography"),
            ("terrain", "Jezero regional terrain"),
            ("atmosphere", "Atmosphere / weather"),
            ("radiation", "Radiation environment"),
            ("water", "Water / subsurface ice"),
        ]
        for index, (key, label) in enumerate(definitions):
            card = SurvivalDecisionCard(label)
            self.decision_cards[key] = card
            grid.addWidget(card, index // 2, index % 2)
        evidence_layout.addLayout(grid)
        page.addWidget(evidence_panel)

        # Actionable decisions, not generic blocks of text.
        action_panel, action_layout = add_panel(
            "02  MISSION DECISION QUEUE",
            "Priorities that follow from the current evidence gaps. These are preparation tasks, not autonomous safety instructions."
        )
        action_grid = QGridLayout()
        action_grid.setSpacing(9)
        priorities = [
            ("01  ENVIRONMENTAL ENVELOPE", "Obtain location- and season-specific pressure, temperature, wind, dust and atmospheric-composition evidence before evaluating EVA or habitat conditions.", "DATA GAP", "amber"),
            ("02  RADIATION BASELINE", "Add a cited radiation dataset or model with time coverage, units, uncertainty and instrument/source metadata. Do not infer dose from surface colour or elevation.", "DATA GAP", "rose"),
            ("03  RESOURCE VERIFICATION", "Use appropriate radar, neutron, spectral or other dedicated observations before claiming water, ice or mineral resources at a selected site.", "DATA GAP", "violet"),
            ("04  TERRAIN VALIDATION", "Use local high-resolution elevation, slope and roughness data for a specific route. Jezero arrays must not be extrapolated to the rest of Mars.", "REGIONAL ONLY", "cyan"),
        ]
        for i, (heading, body, tag, tone) in enumerate(priorities):
            tile = QFrame()
            tile.setObjectName("priorityTile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(13, 11, 13, 11)
            tile_layout.setSpacing(6)
            top = QHBoxLayout()
            h = QLabel(heading)
            h.setObjectName("priorityHeading")
            t = QLabel(tag)
            t.setObjectName("priorityTag")
            t.setProperty("tone", tone)
            top.addWidget(h, 1)
            top.addWidget(t)
            tile_layout.addLayout(top)
            b = QLabel(body)
            b.setWordWrap(True)
            b.setObjectName("priorityBody")
            tile_layout.addWidget(b)
            action_grid.addWidget(tile, i // 2, i % 2)
        action_layout.addLayout(action_grid)
        page.addWidget(action_panel)

        # Location-aware context: coordinates are not environmental measurements.
        context_panel, context_layout = add_panel(
            "03  SELECTED MISSION LOCATION",
            "Shared location context from the Mars globe and route planner."
        )
        self.location_context = QLabel(
            "No location selected. Select a point on the Mars globe to display its coordinates here."
        )
        self.location_context.setObjectName("locationInfo")
        self.location_context.setWordWrap(True)
        context_layout.addWidget(self.location_context)
        page.addWidget(context_panel)

        # Honest simulation boundary.
        simulation_panel, simulation_layout = add_panel(
            "04  SIMULATION BOUNDARY",
            "The Mission Twin, EVA dashboard, habitat status and rover telemetry are demonstrations."
        )
        sim_text = QLabel(
            "MISSION TWIN: resource consumption rates are editable demonstration assumptions, not verified life-support engineering values.\n"
            "ASTRONAUT EVA: oxygen, battery, CO₂, pressure, temperature, humidity and heart-rate values are simulated; no suit or crew sensors are connected.\n"
            "HABITAT SYSTEMS: subsystem panels are conceptual and are not connected to pressure, scrubber, thermal or power hardware.\n"
            "ROVER OPERATIONS: route traversal and telemetry are simulated; the Jezero route candidates are not certified rover commands."
        )
        sim_text.setWordWrap(True)
        sim_text.setObjectName("simulationBoundary")
        simulation_layout.addWidget(sim_text)
        page.addWidget(simulation_panel)

        limits_panel, limits_layout = add_panel(
            "05  SCIENTIFIC LIMITATIONS & INTERPRETATION",
            "Unknown values remain unknown. Missing data must never be treated as evidence of safe conditions."
        )
        self.limits_text = QLabel(
            "GLOBAL FOUNDATION: Viking MDIM provides surface imagery; MOLA MEGDR provides elevation/topography.\n\n"
            "NOT ESTABLISHED BY THOSE TWO LAYERS: surface pressure, atmospheric composition, temperature, wind, weather, radiation dose, water/ice abundance, mineral composition or human habitability.\n\n"
            "REGIONAL SCOPE: Jezero slope, roughness and terrain-cost arrays describe the Jezero data region only. Relative terrain cost is experimental and is not a validated mobility or hazard score.\n\n"
            "This prototype supports exploration and evidence review. It is not a certified crew-safety system, landing-site approval tool or operational mission planner."
        )
        self.limits_text.setObjectName("limitsText")
        self.limits_text.setWordWrap(True)
        limits_layout.addWidget(self.limits_text)
        page.addWidget(limits_panel)

        page.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        self.refresh_btn.clicked.connect(self.refresh_intelligence)
        if self.selection is not None:
            self.selection.selectionChanged.connect(self.refresh_location_context)
        self.refresh_intelligence()
        self.refresh_location_context()

    @staticmethod
    def _sample_stats(path):
        """Read a bounded sample from a 2-D NumPy grid without loading all cells."""
        if path is None or not Path(path).is_file():
            return None
        try:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2:
                return {"error": "Expected a two-dimensional grid."}
            stride = max(1, math.ceil(max(array.shape) / 350))
            sample = np.asarray(array[::stride, ::stride], dtype=np.float32)
            sample = sample[np.isfinite(sample)]
            if sample.size == 0:
                return None
            return {
                "shape": tuple(array.shape),
                "min": float(np.min(sample)),
                "mean": float(np.mean(sample)),
                "p95": float(np.percentile(sample, 95)),
                "max": float(np.max(sample)),
                "samples": int(sample.size),
                "stride": stride,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def refresh_location_context(self):
        if self.selection is None or self.selection.selected_point is None:
            self.location_context.setText(
                "No location selected. Select a point on the Mars globe to display its coordinates here. "
                "Coordinates alone do not provide local pressure, temperature, radiation or resource measurements."
            )
            return
        point = self.selection.selected_point
        name = self.selection.selected_name
        self.location_context.setText(
            f"SELECTED: {name}\n"
            f"Latitude: {point[0]:.3f}°  |  East longitude: {point[1] % 360:.3f}°\n\n"
            "The selected coordinates are a geographic reference only. Site-specific environmental or resource values are not inferred unless a suitable scientific layer is loaded."
        )

    def _set_card(self, key, status, summary, action, details, tone):
        self.decision_cards[key].set_content(status, summary, action, details, tone)

    def refresh_intelligence(self):
        self.refresh_btn.setEnabled(False)
        self.readiness_badge.setText("SCANNING LOCAL DATASETS")
        self.readiness_text.setText("Checking global imagery, elevation and regional Jezero arrays…")
        try:
            # Viking MDIM
            viking_ok = IMAGE_PATH.is_file()
            viking_detail = ""
            if viking_ok:
                viking_detail = (
                    f"File: {IMAGE_PATH}\nSize: {IMAGE_PATH.stat().st_size / (1024 * 1024):.1f} MB.\n"
                    "Use: global surface appearance and geographic context."
                )
                if rasterio is not None:
                    try:
                        with rasterio.open(IMAGE_PATH) as src:
                            viking_detail += f"\nRaster: {src.width:,} × {src.height:,} pixels; {src.count} bands."
                    except Exception:
                        pass
            self._set_card(
                "viking",
                "AVAILABLE" if viking_ok else "MISSING",
                "Global surface imagery is present." if viking_ok else "The Viking global mosaic was not found in the expected project folder.",
                "Use it for broad surface context. Check local-resolution imagery before interpreting a small landing or traverse area.",
                (viking_detail or f"Expected file: {IMAGE_PATH}") + "\n\nLIMIT: imagery does not directly measure weather, pressure, radiation, water/ice abundance or mineral composition.",
                "available" if viking_ok else "missing",
            )

            # MOLA: prefer the global 4-pixel/degree grid, then fall back to the globe grid.
            mola_path = GLOBAL_ELEVATION_PATH if GLOBAL_ELEVATION_PATH.is_file() else (
                GLOBE_ELEVATION_PATH if GLOBE_ELEVATION_PATH.is_file() else None
            )
            mola_ok = mola_path is not None
            mola_stats = self._sample_stats(mola_path) if mola_ok else None
            mola_detail = f"File: {mola_path}" if mola_path else f"Expected either: {GLOBAL_ELEVATION_PATH} or {GLOBE_ELEVATION_PATH}"
            mola_summary = "Global elevation/topography is present." if mola_ok else "No MOLA elevation array was found."
            if mola_stats and "mean" in mola_stats:
                mola_summary += f" Sampled values: {mola_stats['min']:.0f} to {mola_stats['max']:.0f} m."
                mola_detail += (
                    f"\nGrid: {mola_stats['shape']}\nSampled elevation range: "
                    f"{mola_stats['min']:.1f} to {mola_stats['max']:.1f} m; "
                    f"sample mean {mola_stats['mean']:.1f} m."
                )
            elif mola_stats and "error" in mola_stats:
                mola_detail += f"\nRead warning: {mola_stats['error']}"
            self._set_card(
                "mola",
                "AVAILABLE" if mola_ok else "MISSING",
                mola_summary,
                "Use for elevation and broad topographic context. Use local-resolution terrain products for route-scale decisions.",
                mola_detail + "\n\nLIMIT: elevation alone does not establish surface strength, trafficability, weather, radiation or human safety.",
                "available" if mola_ok else "missing",
            )

            # Jezero regional data — explicitly keep its geographic scope local.
            slope_path = PROCESSED_DIR / "jezero_slope_degrees.npy"
            roughness_path = PROCESSED_DIR / "jezero_roughness.npy"
            cost_path = PROCESSED_DIR / "jezero_terrain_cost.npy"
            slope = self._sample_stats(slope_path)
            roughness = self._sample_stats(roughness_path)
            cost = self._sample_stats(cost_path)
            slope_ok = slope is not None and "mean" in slope
            roughness_ok = roughness is not None and "mean" in roughness
            cost_ok = cost is not None and "mean" in cost
            terrain_ok = slope_ok and roughness_ok
            terrain_partial = (slope_ok or roughness_ok or cost_ok) and not terrain_ok
            terrain_tone = "available" if terrain_ok else "partial" if terrain_partial else "missing"
            terrain_status = "AVAILABLE" if terrain_ok else "PARTIAL" if terrain_partial else "NOT LOADED"
            terrain_summary = (
                "Jezero slope and roughness arrays are both readable."
                if terrain_ok else
                "Some Jezero terrain data is present, but one or more core layers are missing."
                if terrain_partial else
                "No readable Jezero terrain arrays were found."
            )
            terrain_lines = [
                f"Slope array: {'readable' if slope_ok else 'missing / unreadable'} ({slope_path})",
                f"Roughness array: {'readable' if roughness_ok else 'missing / unreadable'} ({roughness_path})",
                f"Relative cost array: {'readable' if cost_ok else 'missing / unreadable'} ({cost_path})",
            ]
            if slope_ok:
                terrain_lines.append(
                    f"Slope sample: mean {slope['mean']:.2f}°, 95th percentile {slope['p95']:.2f}°, sample max {slope['max']:.2f}°."
                )
            if roughness_ok:
                terrain_lines.append(
                    f"Roughness sample: mean {roughness['mean']:.3f} m, 95th percentile {roughness['p95']:.3f} m, sample max {roughness['max']:.3f} m."
                )
            if cost_ok:
                terrain_lines.append(
                    f"Experimental relative cost: mean {cost['mean']:.3f}, 95th percentile {cost['p95']:.3f}, sample max {cost['max']:.3f}."
                )
            dem_path = BASE_DIR / "data" / "regional" / "jezero" / "M20_JezeroCrater_CTXDEM_20m.tif"
            if dem_path.is_file() and rasterio is not None:
                try:
                    with rasterio.open(dem_path) as dem:
                        terrain_lines.append(f"CTX DEM: {dem.width:,} × {dem.height:,} pixels; {dem.res[0]:g} m/pixel.")
                except Exception:
                    terrain_lines.append("CTX DEM file exists; metadata could not be read.")
            self._set_card(
                "terrain", terrain_status, terrain_summary,
                "Use Jezero arrays only within their covered region. Review the actual slope and roughness layers; do not treat relative cost as a certified hazard score.",
                "\n".join(terrain_lines) + "\n\nLIMIT: these are regional Jezero products, not a global Mars terrain layer. A* output is an exploratory candidate, not a certified safe route.",
                terrain_tone,
            )

            missing_cards = [
                ("atmosphere", "ATMOSPHERE / WEATHER", "No direct local atmospheric dataset is currently loaded.",
                 "Obtain site- and season-specific pressure, temperature, wind, dust and composition evidence before evaluating surface or EVA conditions.",
                 "Viking MDIM imagery and MOLA elevation do not directly measure atmospheric state. An archived instrument record must be identified by instrument, location, time, units and calibration status."),
                ("radiation", "RADIATION ENVIRONMENT", "No radiation measurement layer is currently loaded.",
                 "Add a dedicated radiation dataset or documented model with time coverage and uncertainty before discussing dose or shielding requirements.",
                 "Radiation dose cannot be inferred from surface imagery or elevation. Missing radiation data is an unknown, not a zero-dose condition."),
                ("water", "WATER / SUBSURFACE ICE", "No regional water or ice evidence layer is currently loaded.",
                 "Use dedicated radar, neutron, spectral or other appropriate observations before identifying a resource prospect.",
                 "Colour, morphology and elevation alone do not prove water/ice abundance. Resource presence and extractability are separate questions."),
            ]
            for key, label, summary, action, detail in missing_cards:
                self._set_card(key, "NOT LOADED", summary, action, detail, "missing")

            available_count = int(viking_ok) + int(mola_ok) + int(terrain_ok)
            partial_count = int(terrain_partial)
            coverage_percent = int(round(available_count / 6 * 100))
            self.coverage_bar.setValue(coverage_percent)
            self.coverage_label.setText(f"{available_count} / 6 complete domains" + ("  •  1 partial" if partial_count else ""))

            if viking_ok and mola_ok and terrain_ok:
                badge = "CORE MAP DATA PRESENT · ENVIRONMENTAL GAPS REMAIN"
                tone = "#ffc46e"
                headline = "The global mapping foundation and Jezero terrain arrays are available, but key human-survival variables are not loaded."
            elif viking_ok or mola_ok or terrain_partial:
                badge = "PARTIAL SCIENTIFIC COVERAGE"
                tone = "#ffc46e"
                headline = "Some mapping evidence is available. The current project cannot assess complete local environmental or crew-safety conditions."
            else:
                badge = "DATA FOUNDATION INCOMPLETE"
                tone = "#ff8d91"
                headline = "Required global or regional datasets are missing. Review the evidence cards before using any map-derived result."
            self.readiness_badge.setText(badge)
            self.readiness_badge.setStyleSheet(
                f"color:{tone}; background:rgba(10,18,30,170); border:1px solid {tone};"
                "border-radius:9px; padding:7px 10px; font-size:8px; font-weight:900;"
            )
            self.readiness_text.setText(headline)

            terrain_summary_lines = []
            if slope_ok:
                terrain_summary_lines.append(
                    f"Slope: mean {slope['mean']:.2f}° | P95 {slope['p95']:.2f}° | sample max {slope['max']:.2f}°."
                )
            if roughness_ok:
                terrain_summary_lines.append(
                    f"Roughness: mean {roughness['mean']:.3f} m | P95 {roughness['p95']:.3f} m | sample max {roughness['max']:.3f} m."
                )
            self.metrics["surface"][0].setText("AVAILABLE" if viking_ok else "MISSING")
            self.metrics["surface"][1].setText("Viking MDIM global imagery")
            self.metrics["topography"][0].setText("AVAILABLE" if mola_ok else "MISSING")
            self.metrics["topography"][1].setText("MOLA MEGDR elevation grid")
            self.metrics["regional"][0].setText("AVAILABLE" if terrain_ok else "PARTIAL" if terrain_partial else "MISSING")
            self.metrics["regional"][1].setText("Jezero region only")

        except Exception as exc:
            self.readiness_badge.setText("ASSESSMENT INCOMPLETE")
            self.readiness_badge.setStyleSheet(
                "color:#ff8d91; background:#3b222b; border:1px solid #8b4650; border-radius:8px; padding:7px;"
            )
            self.readiness_text.setText(f"The scientific data scan could not complete: {exc}")
            self.coverage_label.setText("Coverage not calculated")
        finally:
            self.refresh_btn.setEnabled(True)


# ============================================================
# MISSION TWIN — CONFIGURABLE RESOURCE BUDGET SIMULATION
# ============================================================
class MissionTwinPage(QWidget):
    def __init__(self, model: MissionModel, parent=None):
        super().__init__(parent)
        self.model = model
        root = QVBoxLayout(self)
        title = QLabel("MISSION TWIN | MISSION CONFIGURATION")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        note = QLabel("SIMULATION ONLY • Editable demonstration assumptions • Not a certified human-mission planning model")
        note.setObjectName("warningText")
        root.addWidget(note)

        body = QHBoxLayout()
        config, config_layout = add_panel("Mission configuration", "Set crew, duration and available reserves.")
        form = QFormLayout()
        self.crew = QSpinBox(); self.crew.setRange(1, 20); self.crew.setValue(4)
        self.sols = QSpinBox(); self.sols.setRange(1, 2000); self.sols.setValue(180)
        self.oxygen = self._spin(750, 0, 100000, 1, " kg")
        self.water = self._spin(4500, 0, 1000000, 10, " L")
        self.food = self._spin(1400, 0, 100000, 1, " kg")
        self.energy = self._spin(3600, 0, 1000000, 10, " kWh")
        form.addRow("Crew size", self.crew)
        form.addRow("Mission duration (sols)", self.sols)
        form.addRow("Oxygen reserve", self.oxygen)
        form.addRow("Water reserve", self.water)
        form.addRow("Food reserve", self.food)
        form.addRow("Energy reserve", self.energy)
        config_layout.addLayout(form)
        assumptions, assumptions_layout = add_panel("Editable demo consumption rates", "These are placeholders for demonstrating resource budgeting; replace with mission-approved engineering values before any real use.")
        rates_form = QFormLayout()
        self.o2_rate = self._spin(0.84, 0, 100, 0.01, " kg / person-sol")
        self.water_rate = self._spin(5.0, 0, 1000, 0.1, " L / person-sol")
        self.food_rate = self._spin(1.5, 0, 100, 0.1, " kg / person-sol")
        self.energy_rate = self._spin(4.0, 0, 1000, 0.1, " kWh / person-sol")
        rates_form.addRow("Oxygen", self.o2_rate)
        rates_form.addRow("Water", self.water_rate)
        rates_form.addRow("Food", self.food_rate)
        rates_form.addRow("Energy", self.energy_rate)
        assumptions_layout.addLayout(rates_form)
        self.run_btn = QPushButton("RUN MISSION SIMULATION")
        self.run_btn.clicked.connect(self.run_simulation)
        assumptions_layout.addWidget(self.run_btn)
        left_col = QVBoxLayout(); left_col.addWidget(config); left_col.addWidget(assumptions); left_col.addStretch()
        body.addLayout(left_col, 4)

        result_panel, result_layout = add_panel("Mission twin results", "Projected reserves after the configured mission duration.")
        self.result_status = QLabel("CONFIGURATION NOT YET SIMULATED")
        self.result_status.setObjectName("statusPill")
        result_layout.addWidget(self.result_status)
        self.result_cards = QGridLayout()
        self.result_values = {}
        for i, (name, unit) in enumerate((("Oxygen", "kg"), ("Water", "L"), ("Food", "kg"), ("Energy", "kWh"))):
            card, value, detail = metric_card(f"{name} remaining", "—", f"{unit} after mission")
            self.result_values[name] = (value, detail)
            self.result_cards.addWidget(card, i // 2, i % 2)
        result_layout.addLayout(self.result_cards)
        self.result_summary = QLabel("Configure the mission and run the simulation to calculate resource requirements.")
        self.result_summary.setWordWrap(True)
        self.result_summary.setObjectName("muted")
        result_layout.addWidget(self.result_summary)
        body.addWidget(result_panel, 6)
        root.addLayout(body, 1)

    @staticmethod
    def _spin(value, minimum, maximum, step, suffix):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    def run_simulation(self):
        supplies = {
            "Oxygen": self.oxygen.value(), "Water": self.water.value(),
            "Food": self.food.value(), "Energy": self.energy.value(),
        }
        rates = {
            "Oxygen": self.o2_rate.value(), "Water": self.water_rate.value(),
            "Food": self.food_rate.value(), "Energy": self.energy_rate.value(),
        }
        report = self.model.simulate(self.crew.value(), self.sols.value(), supplies, rates)
        units = {"Oxygen": "kg", "Water": "L", "Food": "kg", "Energy": "kWh"}
        for key in report["remaining"]:
            value, detail = self.result_values[key]
            remaining = report["remaining"][key]
            value.setText(f"{remaining:,.1f} {units[key]}" if remaining >= 0 else f"SHORT {abs(remaining):,.1f} {units[key]}")
            detail.setText(f"Required: {report['required'][key]:,.1f} {units[key]} • Reserve coverage: {report['coverage'][key]:.1f}%")
        if report["feasible_under_assumptions"]:
            self.result_status.setText("RESERVES COVER THIS DEMO SCENARIO")
            self.result_status.setProperty("state", "good")
            self.result_summary.setText("All four reserves cover the configured mission under the editable demonstration rates. This does not establish real mission feasibility or crew safety.")
        else:
            short = [key for key, enough in report["adequate"].items() if not enough]
            self.result_status.setText("RESOURCE SHORTAGE — REDESIGN REQUIRED")
            self.result_status.setProperty("state", "warning")
            self.result_summary.setText("Insufficient simulated reserves: " + ", ".join(short) + ". Increase reserves, reduce duration/crew, or adjust approved assumptions. This is a planning demonstration only.")
        self.result_status.style().unpolish(self.result_status)
        self.result_status.style().polish(self.result_status)


# ============================================================
# HABITAT SYSTEMS — SHARED MISSION RESOURCE STATUS
# ============================================================
class HabitatSystemsPage(QWidget):
    def __init__(self, model: MissionModel, parent=None):
        super().__init__(parent)
        self.model = model
        root = QVBoxLayout(self)
        title = QLabel("HABITAT SYSTEMS | LIFE-SUPPORT RESOURCE BOARD")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        warning = QLabel("SIMULATED SYSTEM STATUS • Connect to Mission Twin configuration to update resource budgets")
        warning.setObjectName("warningText")
        root.addWidget(warning)
        self.cards = {}
        grid = QGridLayout(); grid.setSpacing(10)
        for idx, key in enumerate(("Oxygen", "Water", "Food", "Energy")):
            frame, layout = add_panel(key + " reserve")
            amount = QLabel("No mission simulation yet")
            amount.setObjectName("metricValue")
            bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(0)
            status = QLabel("Waiting for Mission Twin")
            status.setObjectName("muted")
            layout.addWidget(amount); layout.addWidget(bar); layout.addWidget(status)
            self.cards[key] = (amount, bar, status)
            grid.addWidget(frame, idx // 2, idx % 2)
        root.addLayout(grid)
        system, system_layout = add_panel("Habitat subsystem overview", "Conceptual status panels for the mission digital twin.")
        systems_grid = QGridLayout()
        for i, (name, detail) in enumerate([
            ("Pressure Control", "Simulation placeholder — no pressure sensor connected"),
            ("CO₂ Removal", "Simulation placeholder — dedicated scrubber model required"),
            ("Thermal Control", "Simulation placeholder — temperature data not loaded"),
            ("Power Distribution", "Budget represented by Mission Twin energy reserve"),
        ]):
            cell, cell_layout = add_panel(name, detail)
            state = QLabel("NOT TELEMETRY-CONNECTED")
            state.setObjectName("warningText")
            cell_layout.addWidget(state)
            systems_grid.addWidget(cell, i // 2, i % 2)
        system_layout.addLayout(systems_grid)
        root.addWidget(system, 1)
        self.model.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        report = self.model.report
        units = {"Oxygen": "kg", "Water": "L", "Food": "kg", "Energy": "kWh"}
        for key, (amount, bar, status) in self.cards.items():
            if not report:
                amount.setText("—")
                bar.setValue(0)
                status.setText("Run a mission in Mission Twin first")
                continue
            reserve = report["supplies"][key]
            remaining = max(0.0, report["remaining"][key])
            percent = int(np.clip(remaining / reserve * 100 if reserve > 0 else 0, 0, 100))
            amount.setText(f"{remaining:,.1f} {units[key]} remaining")
            bar.setValue(percent)
            status.setText("Budget covers mission" if report["adequate"][key] else "SHORTAGE in configured scenario")


class SuitVisualWidget(QWidget):
    """Stylised, non-photorealistic spacesuit diagram drawn with Qt primitives."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 390)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#0b1722"))
        painter.setPen(QPen(QColor("#1d3c50"), 1))
        for y in range(18, h, 28):
            painter.drawLine(12, y, w - 12, y)
        cx = w * 0.50
        scale = min(w / 360.0, h / 470.0)
        def r(x, y, ww, hh):
            return QRectF(cx + (x - 180) * scale, (y - 18) * scale, ww * scale, hh * scale)
        shell = QLinearGradient(0, 0, w, h)
        shell.setColorAt(0, QColor("#d6e2e8"))
        shell.setColorAt(0.5, QColor("#8fa8b7"))
        shell.setColorAt(1, QColor("#435e70"))
        painter.setPen(QPen(QColor("#d8eff4"), max(1.2, 2.0 * scale)))
        painter.setBrush(QBrush(shell))
        # Backpack behind the torso.
        painter.drawRoundedRect(r(116, 168, 128, 175), 18 * scale, 18 * scale)
        painter.setBrush(QColor("#243c4c"))
        painter.drawRoundedRect(r(130, 184, 100, 135), 12 * scale, 12 * scale)
        # Legs and boots.
        painter.setBrush(QBrush(shell))
        painter.drawRoundedRect(r(132, 305, 43, 112), 14 * scale, 14 * scale)
        painter.drawRoundedRect(r(185, 305, 43, 112), 14 * scale, 14 * scale)
        painter.setBrush(QColor("#253f50"))
        painter.drawRoundedRect(r(125, 398, 56, 27), 7 * scale, 7 * scale)
        painter.drawRoundedRect(r(179, 398, 56, 27), 7 * scale, 7 * scale)
        # Arms.
        painter.setBrush(QBrush(shell))
        painter.drawRoundedRect(r(76, 174, 48, 145), 19 * scale, 19 * scale)
        painter.drawRoundedRect(r(236, 174, 48, 145), 19 * scale, 19 * scale)
        painter.setBrush(QColor("#263f50"))
        painter.drawEllipse(r(78, 300, 46, 43))
        painter.drawEllipse(r(236, 300, 46, 43))
        # Torso.
        painter.setBrush(QBrush(shell))
        painter.drawRoundedRect(r(119, 151, 122, 178), 26 * scale, 26 * scale)
        # Chest control panel.
        painter.setBrush(QColor("#142c3d"))
        painter.drawRoundedRect(r(145, 204, 70, 75), 9 * scale, 9 * scale)
        painter.setPen(QPen(QColor("#6ce5de"), max(1, 2 * scale)))
        painter.drawRoundedRect(r(153, 214, 54, 27), 5 * scale, 5 * scale)
        painter.setPen(QPen(QColor("#f3b36e"), max(1, 2 * scale)))
        painter.drawEllipse(r(157, 250, 10, 10)); painter.drawEllipse(r(177, 250, 10, 10)); painter.drawEllipse(r(197, 250, 10, 10))
        # Neck ring and helmet.
        painter.setPen(QPen(QColor("#e7f5f7"), max(2, 4 * scale)))
        painter.setBrush(QColor("#718b9b"))
        painter.drawRoundedRect(r(151, 129, 58, 28), 8 * scale, 8 * scale)
        painter.setBrush(QColor("#c2d1d8"))
        painter.drawEllipse(r(116, 38, 128, 112))
        painter.setBrush(QColor("#0d2536"))
        painter.setPen(QPen(QColor("#68dce5"), max(2, 3 * scale)))
        painter.drawEllipse(r(129, 52, 102, 82))
        painter.setBrush(QColor(72, 151, 174, 80))
        painter.setPen(Qt.NoPen)
        painter.drawPie(r(137, 59, 86, 68), 20 * 16, 115 * 16)
        # Visual callouts.
        painter.setPen(QPen(QColor("#79e4df"), 1))
        painter.drawLine(r(230, 84, 0, 0).topLeft(), r(300, 70, 0, 0).topLeft())
        painter.drawLine(r(232, 225, 0, 0).topLeft(), r(300, 220, 0, 0).topLeft())
        painter.drawLine(r(128, 225, 0, 0).topLeft(), r(55, 225, 0, 0).topLeft())
        painter.setPen(QPen(QColor("#c5d6e0"), 1))
        painter.drawText(r(292, 61, 65, 20), Qt.AlignLeft | Qt.AlignVCenter, "HELMET")
        painter.drawText(r(292, 210, 65, 24), Qt.AlignLeft | Qt.AlignVCenter, "PLSS")
        painter.drawText(r(4, 215, 65, 24), Qt.AlignLeft | Qt.AlignVCenter, "O₂ CORE")
        painter.setPen(QPen(QColor("#8299aa"), 1))
        painter.drawText(r(100, 444, 160, 20), Qt.AlignCenter, "CONCEPTUAL SUIT DIAGRAM")
        painter.end()


# ============================================================
# ASTRONAUT EVA — SPACESUIT VISUAL DASHBOARD (SIMULATED)
# ============================================================
class AstronautEVAPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.elapsed = 0
        self.oxygen = 100.0
        self.battery = 100.0
        self.co2_ppm = 450.0
        self.temperature = 22.0
        self.humidity = 45.0
        self.heart_rate = 78
        root = QVBoxLayout(self)
        title = QLabel("ASTRONAUT EVA | SPACESUIT DIGITAL TWIN")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        note = QLabel("SIMULATED TELEMETRY • No real astronaut, suit, sensor or spacecraft connection")
        note.setObjectName("warningText")
        root.addWidget(note)
        top = QHBoxLayout()
        self.status = QLabel("SUIT STATUS: STANDBY")
        self.status.setObjectName("statusPill")
        self.clock = QLabel("EVA ELAPSED: 00:00:00")
        self.clock.setObjectName("metricValue")
        top.addWidget(self.status); top.addStretch(); top.addWidget(self.clock)
        root.addLayout(top)
        body = QHBoxLayout(); body.setSpacing(10)
        suit, suit_layout = add_panel("Suit visualisation", "Conceptual suit silhouette — telemetry values are simulated for this demo.")
        suit_art = SuitVisualWidget()
        suit_layout.addWidget(suit_art, 1)
        suit_layout.addWidget(QLabel("HELMET  •  PLSS PACK  •  POWER  •  COMMUNICATION"))
        body.addWidget(suit, 4)

        telemetry, telemetry_layout = add_panel("Suit telemetry", "Illustrative demo readings; not certified suit operating limits.")
        self.telemetry_cards = {}
        rows = [
            ("Oxygen supply", "oxygen", "%", 0, 100),
            ("Suit battery", "battery", "%", 0, 100),
            ("CO₂ concentration", "co2", "ppm", 0, 5000),
            ("Suit pressure", "pressure", "kPa", None, None),
            ("Temperature", "temperature", "°C", None, None),
            ("Humidity", "humidity", "%", None, None),
            ("Crew heart rate", "heart", "BPM", None, None),
        ]
        for name, key, unit, low, high in rows:
            row = QHBoxLayout()
            label = QLabel(name); label.setObjectName("muted")
            value = QLabel("—"); value.setObjectName("metricValue")
            row.addWidget(label); row.addStretch(); row.addWidget(value)
            telemetry_layout.addLayout(row)
            bar = None
            if low is not None:
                bar = QProgressBar(); bar.setRange(int(low), int(high)); telemetry_layout.addWidget(bar)
            self.telemetry_cards[key] = (value, bar)
        controls = QHBoxLayout()
        self.start_btn = QPushButton("START EVA SIMULATION")
        self.pause_btn = QPushButton("PAUSE")
        self.return_btn = QPushButton("RETURN TO HABITAT")
        self.warning_btn = QPushButton("TEST WARNING")
        controls.addWidget(self.start_btn); controls.addWidget(self.pause_btn)
        controls.addWidget(self.return_btn); controls.addWidget(self.warning_btn)
        telemetry_layout.addLayout(controls)
        self.alert = QLabel("No active alerts.")
        self.alert.setWordWrap(True)
        self.alert.setObjectName("muted")
        telemetry_layout.addWidget(self.alert)
        body.addWidget(telemetry, 6)
        root.addLayout(body, 1)
        self.start_btn.clicked.connect(self.start_eva)
        self.pause_btn.clicked.connect(self.pause_eva)
        self.return_btn.clicked.connect(self.end_eva)
        self.warning_btn.clicked.connect(self.inject_warning)
        self.refresh()

    def start_eva(self):
        self.timer.start()
        self.status.setText("SUIT STATUS: EVA SIMULATION ACTIVE")

    def pause_eva(self):
        self.timer.stop()
        self.status.setText("SUIT STATUS: PAUSED")

    def end_eva(self):
        self.timer.stop()
        self.status.setText("SUIT STATUS: RETURNED / EVA ENDED")
        self.alert.setText("EVA simulation ended. Telemetry values remain as the final demo state.")

    def inject_warning(self):
        self.oxygen = min(self.oxygen, 18.0)
        self.battery = min(self.battery, 22.0)
        self.co2_ppm = max(self.co2_ppm, 1800.0)
        self.refresh()

    def tick(self):
        self.elapsed += 1
        self.oxygen = max(0.0, self.oxygen - 0.025)
        self.battery = max(0.0, self.battery - 0.012)
        self.co2_ppm = min(5000.0, self.co2_ppm + 1.4)
        self.temperature = 22.0 + 1.4 * math.sin(self.elapsed / 18.0)
        self.humidity = 45.0 + 3.0 * math.sin(self.elapsed / 13.0)
        self.heart_rate = int(78 + 6 * math.sin(self.elapsed / 8.0))
        self.refresh()

    def refresh(self):
        hh, rem = divmod(self.elapsed, 3600); mm, ss = divmod(rem, 60)
        self.clock.setText(f"EVA ELAPSED: {hh:02d}:{mm:02d}:{ss:02d}")
        values = {
            "oxygen": (f"{self.oxygen:.1f} %", self.oxygen),
            "battery": (f"{self.battery:.1f} %", self.battery),
            "co2": (f"{self.co2_ppm:.0f} ppm", self.co2_ppm),
            "pressure": ("29.6 kPa (demo)", None),
            "temperature": (f"{self.temperature:.1f} °C", None),
            "humidity": (f"{self.humidity:.1f} %", None),
            "heart": (f"{self.heart_rate} BPM", None),
        }
        for key, (text, number) in values.items():
            label, bar = self.telemetry_cards[key]
            label.setText(text)
            if bar is not None:
                bar.setValue(int(np.clip(number, 0, 100)))
        alerts = []
        if self.oxygen < 25: alerts.append("OXYGEN RESERVE LOW")
        if self.battery < 25: alerts.append("SUIT BATTERY LOW")
        if self.co2_ppm > 1500: alerts.append("CO₂ LEVEL HIGH")
        if alerts:
            self.status.setText("SUIT STATUS: WARNING")
            self.alert.setText("⚠  " + "  •  ".join(alerts) + "  |  Simulated alert only.")
            self.alert.setObjectName("warningText")
        else:
            self.alert.setText("No simulated warnings. Values are illustrative, not actual suit telemetry.")
            self.alert.setObjectName("muted")
        self.alert.style().unpolish(self.alert); self.alert.style().polish(self.alert)


# ============================================================
# ROVER OPERATIONS — SIMULATED TELEMETRY
# ============================================================

class RoverOperationsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)

        self.running = False
        self.elapsed_minutes = 0.0
        self.distance = 0.0
        self.battery = 100.0
        self.route_points = []
        self.route_index = 0
        self.route_distance = 0.0
        self.cell_size_m = 20.0

        self.slope = self._load_array("jezero_slope_degrees.npy")
        self.roughness = self._load_array("jezero_roughness.npy")
        self.cost = self._load_array("jezero_terrain_cost.npy")

        self.source_shape = (
            self.slope.shape if self.slope is not None
            else self.roughness.shape if self.roughness is not None
            else self.cost.shape if self.cost is not None
            else None
        )

        root = QVBoxLayout(self)

        title = QLabel("ROVER OPERATIONS | EXPLORATION DIGITAL TWIN")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        note = QLabel(
            "SIMULATED ROVER TELEMETRY • Jezero regional terrain data • "
            "No live rover connection"
        )
        note.setObjectName("warningText")
        root.addWidget(note)

        # Telemetry cards
        stats_panel, stats_layout = add_panel(
            "Rover telemetry",
            "Live simulation values. These are not measurements from a physical rover."
        )

        grid = QGridLayout()
        self.values = {}

        names = [
            ("Rover speed", "0.00 km/h"),
            ("Battery", "100.0 %"),
            ("Pitch", "0.0°"),
            ("Roll", "0.0°"),
            ("Heading", "090°"),
            ("Distance travelled", "0.000 km"),
        ]

        for i, (name, value) in enumerate(names):
            card, val, detail = metric_card(name, value, "SIMULATED")
            self.values[name] = val
            grid.addWidget(card, i // 3, i % 3)

        stats_layout.addLayout(grid)
        root.addWidget(stats_panel)

        # Route selection
        route_panel, route_layout = add_panel(
            "Mission route controller",
            "Select an existing Jezero route candidate and simulate rover traversal."
        )

        controls = QHBoxLayout()

        self.route_combo = QComboBox()
        self.route_combo.addItem(
            "ALPHA | Shortest-distance candidate",
            str(PROCESSED_DIR / "jezero_shortest_route.csv")
        )
        self.route_combo.addItem(
            "BETA | Balanced terrain candidate",
            str(PROCESSED_DIR / "jezero_balanced_route.csv")
        )
        self.route_combo.addItem(
            "GAMMA | Terrain-cost-avoiding candidate",
            str(PROCESSED_DIR / "jezero_terrain_avoiding_route.csv")
        )

        self.speed_combo = QComboBox()
        self.speed_combo.addItem("CAUTIOUS | 0.15 km/h", 0.15)
        self.speed_combo.addItem("NOMINAL | 0.29 km/h", 0.29)
        self.speed_combo.addItem("TRAVERSE | 0.50 km/h", 0.50)
        self.speed_combo.setCurrentIndex(1)

        controls.addWidget(QLabel("ROUTE"))
        controls.addWidget(self.route_combo, 3)
        controls.addWidget(QLabel("SPEED"))
        controls.addWidget(self.speed_combo, 2)

        route_layout.addLayout(controls)

        self.route_status = QLabel("Loading route data...")
        self.route_status.setWordWrap(True)
        self.route_status.setObjectName("muted")
        route_layout.addWidget(self.route_status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("MISSION PROGRESS | %p%")
        route_layout.addWidget(self.progress)

        self.position_label = QLabel("ROVER POSITION: NOT STARTED")
        self.position_label.setObjectName("muted")
        route_layout.addWidget(self.position_label)

        buttons = QHBoxLayout()

        self.start_button = QPushButton("START ROVER")
        self.stop_button = QPushButton("PAUSE ROVER")
        self.reset_button = QPushButton("RESET MISSION")

        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.reset_button)

        route_layout.addLayout(buttons)
        root.addWidget(route_panel)

        # Terrain telemetry
        terrain_panel, terrain_layout = add_panel(
            "Live terrain intelligence",
            "Terrain information is sampled from the regional Jezero arrays "
            "at the simulated rover position."
        )

        self.terrain_status = QLabel("WAITING FOR ROVER MOVEMENT")
        self.terrain_status.setWordWrap(True)
        terrain_layout.addWidget(self.terrain_status)

        self.terrain_warning = QLabel(
            "Terrain cost is a relative experimental index, not a validated safety score."
        )
        self.terrain_warning.setObjectName("warningText")
        self.terrain_warning.setWordWrap(True)
        terrain_layout.addWidget(self.terrain_warning)

        root.addWidget(terrain_panel)
        root.addStretch()

        # Connections
        self.route_combo.currentIndexChanged.connect(self.load_route)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.reset_button.clicked.connect(self.reset)

        self.load_route()

    @staticmethod
    def _load_array(filename):
        path = PROCESSED_DIR / filename
        if not path.is_file():
            return None
        try:
            return np.load(path, mmap_mode="r")
        except Exception:
            return None

    def load_route(self):
        self.stop()

        self.route_points = []
        self.route_index = 0
        self.distance = 0.0
        self.elapsed_minutes = 0.0
        self.battery = 100.0
        self.route_distance = 0.0

        path = Path(self.route_combo.currentData())

        if not path.is_file():
            self.route_status.setText(
                "Route file not found: " + path.name
            )
            self.refresh()
            return

        if self.source_shape is None:
            self.route_status.setText(
                "Regional terrain arrays are unavailable. "
                "Cannot validate route coordinates against the terrain grid."
            )
            self.refresh()
            return

        try:
            with path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                reader = csv.DictReader(file)
                headers = {
                    name.strip().lower(): name
                    for name in (reader.fieldnames or [])
                }

                row_key = headers.get("row") or headers.get("grid_row")
                col_key = headers.get("column") or headers.get("grid_column")

                if not row_key or not col_key:
                    raise ValueError("Route CSV has no row/column fields.")

                for item in reader:
                    row = int(float(item[row_key]))
                    col = int(float(item[col_key]))
                    self.route_points.append((row, col))

            if len(self.route_points) < 2:
                raise ValueError("Route contains fewer than two points.")

            height, width = self.source_shape

            # The existing Terrain Intelligence page downsamples
            # regional arrays to a maximum side of 560 cells.
            downsample_stride = max(
                1, math.ceil(max(height, width) / 560)
            )

            small_height = math.ceil(height / downsample_stride)
            small_width = math.ceil(width / downsample_stride)

            max_row = max(p[0] for p in self.route_points)
            max_col = max(p[1] for p in self.route_points)

            if max_row < small_height and max_col < small_width:
                self.grid_stride = downsample_stride
            elif max_row < height and max_col < width:
                self.grid_stride = 1
            else:
                raise ValueError(
                    "Route coordinates do not match the Jezero terrain grid."
                )

            self.cell_size_m = 20.0 * self.grid_stride

            for (r0, c0), (r1, c1) in zip(
                self.route_points[:-1], self.route_points[1:]
            ):
                self.route_distance += (
                    math.hypot(r1 - r0, c1 - c0)
                    * self.cell_size_m / 1000.0
                )

            self.route_status.setText(
                f"ROUTE LOADED: {path.name}\n"
                f"Route points: {len(self.route_points):,} | "
                f"Estimated raster distance: {self.route_distance:.3f} km | "
                f"Grid step: {self.cell_size_m:.0f} m"
            )

            self.route_status.setObjectName("muted")

        except Exception as exc:
            self.route_points = []
            self.route_status.setText(
                f"ROUTE LOADING ERROR: {exc}"
            )

        self.refresh()

    def start(self):
        if len(self.route_points) < 2:
            self.route_status.setText(
                "Cannot start: select a valid route with at least two points."
            )
            return

        if self.route_index >= len(self.route_points) - 1:
            self.reset()

        self.running = True
        self.timer.start()
        self.status_text("ROVER SIMULATION ACTIVE")

    def status_text(self, message):
        self.position_label.setText(message)

    def stop(self):
        self.running = False
        self.timer.stop()

    def reset(self):
        self.stop()
        self.route_index = 0
        self.distance = 0.0
        self.elapsed_minutes = 0.0
        self.battery = 100.0
        self.progress.setValue(0)
        self.refresh()

    def tick(self):
        if not self.route_points:
            self.stop()
            return

        # Accelerated simulation: advance a fraction of the route
        # on each timer tick.
        increment = max(
            1, math.ceil(len(self.route_points) / 150)
        )

        previous_index = self.route_index
        target_index = min(
            previous_index + increment,
            len(self.route_points) - 1
        )

        step_distance = 0.0

        for i in range(previous_index + 1, target_index + 1):
            r0, c0 = self.route_points[i - 1]
            r1, c1 = self.route_points[i]
            step_distance += (
                math.hypot(r1 - r0, c1 - c0)
                * self.cell_size_m / 1000.0
            )

        speed = float(self.speed_combo.currentData() or 0.29)

        self.distance += step_distance
        self.elapsed_minutes += step_distance / speed * 60.0

        # Illustrative battery model, not physical rover engineering.
        self.battery = max(
            0.0, self.battery - step_distance * 0.15
        )

        self.route_index = target_index

        self.progress.setValue(
            int(
                100 * self.route_index
                / max(1, len(self.route_points) - 1)
            )
        )

        self.refresh()

        if self.route_index >= len(self.route_points) - 1:
            self.stop()
            self.position_label.setText(
                "ROUTE COMPLETE | Simulated traverse finished"
            )

    def refresh(self):
        speed = float(self.speed_combo.currentData() or 0.29)

        current_speed = speed if self.running else 0.0

        self.values["Rover speed"].setText(
            f"{current_speed:.2f} km/h"
        )
        self.values["Battery"].setText(
            f"{self.battery:.1f} %"
        )
        self.values["Pitch"].setText(
            f"{2.0 * math.sin(self.elapsed_minutes / 8):.1f}°"
        )
        self.values["Roll"].setText(
            f"{1.5 * math.cos(self.elapsed_minutes / 10):.1f}°"
        )

        heading = 90

        if self.route_index > 0 and self.route_points:
            r0, c0 = self.route_points[self.route_index - 1]
            r1, c1 = self.route_points[self.route_index]
            heading = int(
                math.degrees(math.atan2(c1 - c0, r0 - r1)) % 360
            )

        self.values["Heading"].setText(f"{heading:03d}°")
        self.values["Distance travelled"].setText(
            f"{self.distance:.3f} km"
        )

        if self.route_points:
            row, col = self.route_points[self.route_index]

            self.status_text(
                f"ROVER GRID POSITION | Row {row} | Column {col}"
            )

            self._refresh_terrain(row, col)

        else:
            self.position_label.setText(
                "ROVER POSITION: NO VALID ROUTE"
            )
            self.terrain_status.setText(
                "No route position available for terrain sampling."
            )

        self.terrain_warning.setText(
            f"Simulation clock: {self.elapsed_minutes:.1f} mission minutes "
            f"| Battery draw is illustrative. "
            f"Grid movement is based on the route CSV."
        )

    def _refresh_terrain(self, row, col):
        source_row = row * self.grid_stride
        source_col = col * self.grid_stride

        details = []

        for name, array, unit in [
            ("Slope", self.slope, "°"),
            ("Roughness", self.roughness, "m"),
            ("Relative terrain cost", self.cost, ""),
        ]:
            if array is None:
                details.append(f"{name}: unavailable")
                continue

            if (
                source_row >= array.shape[0]
                or source_col >= array.shape[1]
            ):
                details.append(f"{name}: coordinate outside grid")
                continue

            try:
                value = float(array[source_row, source_col])

                if not math.isfinite(value):
                    details.append(f"{name}: no-data cell")
                else:
                    details.append(
                        f"{name}: {value:.3f}{unit}"
                    )
            except Exception:
                details.append(f"{name}: unavailable")

        if self.cost is not None:
            try:
                value = float(
                    self.cost[source_row, source_col]
                )
                if math.isfinite(value) and value >= 0.75:
                    details.append(
                        "TERRAIN FLAG: higher relative-cost cell "
                        "(not a validated hazard threshold)"
                    )
            except Exception:
                pass

        self.terrain_status.setText(
            "CURRENT ROVER TERRAIN SAMPLE\n\n"
            + "\n".join(details)
        )



# ============================================================
# GLOBAL SCIENTIFIC MAP — FLAT MAP WITH ALL PLACE LABELS
# ============================================================
class GlobalMapPage(QWidget):
    def __init__(self, selection: MissionSelection, parent=None):
        super().__init__(parent)
        self.selection = selection
        self.zoom_factor = 1.0
        self.cache = {}
        root = QVBoxLayout(self)
        title = QLabel("MARS ENVIRONMENT | GLOBAL SCIENTIFIC LAYERS")
        title.setObjectName("pageTitle"); root.addWidget(title)
        note = QLabel("All named sites are labelled on this 2D view. Click the map to inspect a location or pick a route point when a selection mode is active.")
        note.setObjectName("muted"); root.addWidget(note)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("MAP LAYER"))
        self.selector = QComboBox(); self.selector.addItems(["Global imagery", "Global elevation", "Global slope"])
        self.selector.setMinimumWidth(220); controls.addWidget(self.selector); controls.addStretch()
        root.addLayout(controls)
        self.figure = Figure(figsize=(10, 6), facecolor="#0b1420")
        self.canvas = FigureCanvas(self.figure)
        root.addWidget(self.canvas, 1)
        self.status = QLabel("Loading global layer..."); self.status.setObjectName("muted"); root.addWidget(self.status)
        self.selector.currentTextChanged.connect(self.show_layer)
        self.canvas.mpl_connect("button_press_event", self.on_map_click)
        self.show_layer("Global imagery")
        

    def load_layer(self, name):
        if name in self.cache:
            return self.cache[name]
        if name == "Global imagery":
            if rasterio is None or not IMAGE_PATH.exists():
                raise FileNotFoundError("Viking global mosaic not found or Rasterio unavailable.")
            with rasterio.open(IMAGE_PATH) as src:
                if src.count < 3:
                    raise ValueError("Viking mosaic has fewer than three bands.")
                bands = src.read([1, 2, 3], out_shape=(3, 720, 1440), resampling=Resampling.average)
            data = np.transpose(bands, (1, 2, 0)).astype(np.float32)
            for band in range(3):
                p2, p98 = np.percentile(data[:, :, band], [2, 98])
                data[:, :, band] = np.clip((data[:, :, band] - p2) / max(p98 - p2, 1), 0, 1)
            data = np.roll(data, -(data.shape[1] // 2), axis=1)
        elif name == "Global elevation":
            if not GLOBAL_ELEVATION_PATH.exists():
                raise FileNotFoundError(f"Global elevation array not found: {GLOBAL_ELEVATION_PATH}")
            data = np.load(GLOBAL_ELEVATION_PATH)
        elif name == "Global slope":
            if not GLOBAL_SLOPE_PATH.exists():
                raise FileNotFoundError(f"Global slope array not found: {GLOBAL_SLOPE_PATH}")
            data = np.load(GLOBAL_SLOPE_PATH)
        else:
            raise ValueError("Unknown layer")
        self.cache[name] = data
        return data

    def show_layer(self, name):
        self.figure.clear(); self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#111820")
        try:
            data = self.load_layer(name)
            if name == "Global imagery":
                self.ax.imshow(data, extent=[0, 360, -90, 90], origin="upper", aspect="auto")
                self.status.setText("Viking Orbiter Global Mosaic (MDIM) • global surface imagery")
            elif name == "Global elevation":
                lo, hi = np.nanpercentile(data, [2, 98])
                im = self.ax.imshow(data, extent=[0, 360, -90, 90], origin="upper", aspect="auto", cmap="terrain", vmin=lo, vmax=hi)
                self.figure.colorbar(im, ax=self.ax, label="Elevation (m)", shrink=0.82)
                self.status.setText("MOLA MEGDR-derived elevation • 4 pixels per degree")
            else:
                hi = max(float(np.nanpercentile(data, 98)), 0.01)
                im = self.ax.imshow(data, extent=[0, 360, -90, 90], origin="upper", aspect="auto", cmap="inferno", vmin=0, vmax=hi)
                self.figure.colorbar(im, ax=self.ax, label="Slope (degrees)", shrink=0.82)
                self.status.setText("Slope is calculated from MOLA elevation; this is coarse global terrain context.")
            # Plot every named site on the 2D map at once.
            for place in MARS_PLACES:
                colour = "#ffe18b" if "landing" in place["kind"].lower() else "#70e4e2"
                self.ax.scatter(place["lon"], place["lat"], s=14, c=colour, edgecolors="#15212b", linewidths=0.4, zorder=8)
                self.ax.annotate(place["name"], (place["lon"], place["lat"]), xytext=(3, 3), textcoords="offset points",
                                 color="#fff1dc", fontsize=6.5, zorder=9)
            self.ax.set_xlim(0, 360); self.ax.set_ylim(-90, 90)
            self.ax.set_xlabel("East longitude (degrees)", color="#c9d5df")
            self.ax.set_ylabel("Planetocentric latitude (degrees)", color="#c9d5df")
            self.ax.tick_params(colors="#b5c3cf")
            self.ax.set_title(name, color="#eef3f7")
            self.figure.tight_layout()
        except Exception as error:
            self.ax.text(0.5, 0.55, "LAYER UNAVAILABLE", transform=self.ax.transAxes,
                         ha="center", color="#f0a06b", fontsize=13, weight="bold")
            self.ax.text(0.5, 0.42, str(error), transform=self.ax.transAxes,
                         ha="center", va="center", color="#d4dde6", wrap=True, fontsize=8)
            self.ax.set_axis_off()
            self.status.setText("This layer could not be loaded. Other application modules remain available.")
        self.canvas.draw_idle()

    def on_map_click(self, event):
        if event.inaxes != getattr(self, "ax", None) or event.xdata is None or event.ydata is None:
            return
        lat = float(event.ydata); lon = float(event.xdata) % 360
        if -90 <= lat <= 90:
            self.selection.accept_globe_click(lat, lon)


# ============================================================
# JEZERO TERRAIN INTELLIGENCE — SLOPE / ROUGHNESS / A* DEMO
# ============================================================
class TerrainIntelligencePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = load_jezero_layers(build_cost=True)
        self.mode = None
        self.start_xy = None
        self.goal_xy = None
        self.path = None
        self.cost_small = None
        self.stride = 1
        root = QVBoxLayout(self)
        title = QLabel("TERRAIN INTELLIGENCE | JEZERO REGIONAL ANALYSIS")
        title.setObjectName("pageTitle"); root.addWidget(title)
        note = QLabel("Regional proof-of-concept • select start and destination on the heatmap to run A*. The relative cost index is experimental and is NOT a validated safety score.")
        note.setObjectName("warningText"); note.setWordWrap(True); root.addWidget(note)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("DISPLAY"))
        self.selector = QComboBox(); self.selector.addItems(["Relative terrain cost", "Slope", "Roughness"])
        controls.addWidget(self.selector)
        self.pick_start = QPushButton("PICK TERRAIN START")
        self.pick_goal = QPushButton("PICK TERRAIN DESTINATION")
        self.run_route = QPushButton("CALCULATE A* CANDIDATE")
        self.clear = QPushButton("CLEAR")
        for b in (self.pick_start, self.pick_goal, self.run_route, self.clear): controls.addWidget(b)
        root.addLayout(controls)
        self.figure = Figure(figsize=(10, 6), facecolor="#0b1420")
        self.canvas = FigureCanvas(self.figure); root.addWidget(self.canvas, 1)
        self.status = QLabel("Loading optional Jezero terrain arrays..."); self.status.setObjectName("muted"); root.addWidget(self.status)
        self.selector.currentTextChanged.connect(self.show_layer)
        self.pick_start.clicked.connect(lambda: self.set_mode("start"))
        self.pick_goal.clicked.connect(lambda: self.set_mode("goal"))
        self.run_route.clicked.connect(self.calculate_route)
        self.clear.clicked.connect(self.clear_route)
        self.canvas.mpl_connect("button_press_event", self.on_map_click)
        self._prepare_cost()
        self.show_layer("Relative terrain cost")

    def _prepare_cost(self):
        cost = self.layers.get("cost")
        if cost is not None:
            self.cost_small, self.stride = downsample_grid(cost, max_side=560)

    def _current_array(self):
        key = {"Relative terrain cost": "cost", "Slope": "slope", "Roughness": "roughness"}[self.selector.currentText()]
        return self.layers.get(key)

    def show_layer(self, _name):
        array = self._current_array()
        self.figure.clear(); self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#111820")
        if array is None:
            self.ax.text(0.5, 0.55, "JEZERO TERRAIN DATA NOT FOUND", transform=self.ax.transAxes,
                         ha="center", color="#f0a06b", fontsize=13, weight="bold")
            self.ax.text(0.5, 0.42, "Expected files in data/processed/: jezero_slope_degrees.npy, jezero_roughness.npy, and/or jezero_terrain_cost.npy",
                         transform=self.ax.transAxes, ha="center", va="center", color="#d4dde6", wrap=True, fontsize=8)
            self.ax.set_axis_off()
            self.status.setText("Optional regional layers missing. Global Mars features still work.")
            self.canvas.draw_idle(); return
        display, stride = downsample_grid(array, max_side=560)
        if self.selector.currentText() == "Relative terrain cost":
            im = self.ax.imshow(display, origin="upper", cmap="RdYlGn_r", vmin=0, vmax=1)
            self.figure.colorbar(im, ax=self.ax, label="Relative terrain cost (0–1)", shrink=0.8)
        elif self.selector.currentText() == "Slope":
            hi = max(float(np.nanpercentile(display, 98)), 0.1)
            im = self.ax.imshow(display, origin="upper", cmap="inferno", vmin=0, vmax=hi)
            self.figure.colorbar(im, ax=self.ax, label="Slope (degrees)", shrink=0.8)
        else:
            hi = max(float(np.nanpercentile(display, 98)), 0.1)
            im = self.ax.imshow(display, origin="upper", cmap="viridis", vmin=0, vmax=hi)
            self.figure.colorbar(im, ax=self.ax, label="Local roughness (m; source-derived)", shrink=0.8)
        self.ax.set_title("Jezero regional terrain layer", color="#edf2f7")
        self.ax.set_xlabel(f"Raster column (downsample stride {stride})", color="#c9d5df")
        self.ax.set_ylabel("Raster row", color="#c9d5df")
        self.ax.tick_params(colors="#b5c3cf")
        self._draw_route_markers()
        self.figure.tight_layout()
        self.status.setText("Select start/destination using the buttons, then click the map. A* grid is downsampled for responsiveness.")
        self.canvas.draw_idle()

    def set_mode(self, mode):
        self.mode = mode
        self.status.setText(f"Click the Jezero raster to choose TERRAIN {mode.upper()}.")

    def on_map_click(self, event):
        if self.mode not in ("start", "goal") or event.inaxes != getattr(self, "ax", None):
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        array = self._current_array()
        if array is None:
            return
        display, _ = downsample_grid(array, max_side=560)
        if not (0 <= y < display.shape[0] and 0 <= x < display.shape[1]):
            return
        if self.mode == "start": self.start_xy = (x, y)
        else: self.goal_xy = (x, y)
        self.mode = None
        self._draw_route_markers(); self.canvas.draw_idle()
        self.status.setText(f"Terrain points set • Start: {self.start_xy} • Destination: {self.goal_xy}")

    def _draw_route_markers(self):
        if not hasattr(self, "ax"):
            return
        if self.start_xy:
            self.ax.scatter([self.start_xy[0]], [self.start_xy[1]], c="#53ef9a", s=65, edgecolors="white", zorder=12, label="Start")
        if self.goal_xy:
            self.ax.scatter([self.goal_xy[0]], [self.goal_xy[1]], c="#ff6969", s=65, edgecolors="white", zorder=12, label="Destination")
        if self.path:
            xs = [p[1] for p in self.path]; ys = [p[0] for p in self.path]
            self.ax.plot(xs, ys, color="#4fe5ff", linewidth=2, label="A* candidate")

    def calculate_route(self):
        if self.cost_small is None:
            self.status.setText("A* requires Jezero terrain-cost data. Build it from slope and roughness arrays first.")
            return
        if self.start_xy is None or self.goal_xy is None:
            self.status.setText("Choose both terrain start and destination first.")
            return
        try:
            self.path, score = astar_grid(self.cost_small, self.start_xy, self.goal_xy)
            distance = path_length_km(self.path, pixel_size_m=20.0, stride=self.stride)
            self.show_layer(self.selector.currentText())
            self.status.setText(
                f"A* candidate found • Approx. path length {distance:.2f} km • relative path cost {score:.1f} • ~{20*self.stride:.0f} m grid. NOT a certified safe route."
            )
        except Exception as error:
            self.path = None
            self.status.setText(f"A* could not produce a route: {error}")

    def clear_route(self):
        self.start_xy = self.goal_xy = self.path = None
        self.mode = None
        self.show_layer(self.selector.currentText())


# ============================================================
# MARS NAVIGATION — SEPARATE DISTANCE AND ROUTE PANEL
# ============================================================
class MarsNavigationPage(QWidget):
    def __init__(self, selection: MissionSelection, pick_callback, parent=None):
        super().__init__(parent)
        self.selection = selection
        self.pick_callback = pick_callback
        root = QVBoxLayout(self)
        title = QLabel("MARS NAV | DISTANCE & ROUTE PLANNING")
        title.setObjectName("pageTitle"); root.addWidget(title)
        note = QLabel("Great-circle distance uses Mars mean radius (3,389.5 km). It is a geometric estimate, not a terrain-safe travel route or travel-time prediction.")
        note.setObjectName("muted"); note.setWordWrap(True); root.addWidget(note)
        body = QHBoxLayout(); body.setSpacing(10)
        planner, planner_layout = add_panel("Route endpoints", "Choose known Mars sites or pick arbitrary coordinates on the 3D globe.")
        self.start_combo = QComboBox(); self.dest_combo = QComboBox()
        for place in MARS_PLACES:
            label = f"{place['name']} — {place['lat']:.2f}°, {place['lon']:.2f}°E"
            self.start_combo.addItem(label, (place["lat"], place["lon"], place["name"]))
            self.dest_combo.addItem(label, (place["lat"], place["lon"], place["name"]))
        self.start_combo.setCurrentIndex(0)
        self.dest_combo.setCurrentIndex(1)
        form = QFormLayout(); form.addRow("START LOCATION", self.start_combo); form.addRow("DESTINATION", self.dest_combo)
        planner_layout.addLayout(form)
        pick_row = QHBoxLayout()
        pick_start = QPushButton("PICK START ON GLOBE")
        pick_dest = QPushButton("PICK DESTINATION ON GLOBE")
        pick_row.addWidget(pick_start); pick_row.addWidget(pick_dest)
        planner_layout.addLayout(pick_row)
        calculate = QPushButton("CALCULATE GREAT-CIRCLE DISTANCE")
        planner_layout.addWidget(calculate)
        self.endpoint_status = QLabel("Route not calculated yet."); self.endpoint_status.setObjectName("muted"); self.endpoint_status.setWordWrap(True)
        planner_layout.addWidget(self.endpoint_status)
        clear = QPushButton("CLEAR ROUTE"); planner_layout.addWidget(clear)
        body.addWidget(planner, 5)

        result, result_layout = add_panel("Distance intelligence", "A separate result panel keeps geometric distance distinct from regional terrain path planning.")
        self.distance_value = QLabel("— km")
        self.distance_value.setObjectName("bigDistance")
        result_layout.addWidget(self.distance_value)
        self.angle_value = QLabel("Central angle: —")
        self.angle_value.setObjectName("metricValue"); result_layout.addWidget(self.angle_value)
        self.method_value = QLabel("Method: spherical great-circle / haversine")
        self.method_value.setObjectName("muted"); result_layout.addWidget(self.method_value)
        self.limit_value = QLabel("Terrain-aware path: use Terrain Intelligence → Jezero A* candidate. Global slope arrays do not certify route safety.")
        self.limit_value.setWordWrap(True); self.limit_value.setObjectName("warningText"); result_layout.addWidget(self.limit_value)
        body.addWidget(result, 5)
        root.addLayout(body, 1)

        pick_start.clicked.connect(lambda: self.pick_callback("start"))
        pick_dest.clicked.connect(lambda: self.pick_callback("destination"))
        calculate.clicked.connect(self.calculate)
        clear.clicked.connect(self.clear)
        self.selection.routeChanged.connect(self._refresh_picked_status)

    def calculate(self):
        start_data = self.start_combo.currentData()
        dest_data = self.dest_combo.currentData()
        if not start_data or not dest_data:
            return
        start = (start_data[0], start_data[1]); dest = (dest_data[0], dest_data[1])
        self.selection.set_route_point("start", *start, start_data[2])
        self.selection.set_route_point("destination", *dest, dest_data[2])
        distance = great_circle_distance_km(start, dest)
        angle = distance / MARS_MEAN_RADIUS_KM
        self.distance_value.setText(f"{distance:,.2f} km")
        self.angle_value.setText(f"Central angle: {np.degrees(angle):.3f}°")
        self.endpoint_status.setText(f"START: {start_data[2]}\nDESTINATION: {dest_data[2]}\nRoute drawn on the 3D globe when visible.")

    def _refresh_picked_status(self):
        if self.selection.start_point and self.selection.destination_point:
            distance = great_circle_distance_km(self.selection.start_point, self.selection.destination_point)
            self.endpoint_status.setText(
                f"Globe-selected endpoints received.\nStart: {coordinates_text(self.selection.start_point)}\n"
                f"Destination: {coordinates_text(self.selection.destination_point)}\nGreat-circle distance: {distance:.2f} km"
            )
            self.distance_value.setText(f"{distance:,.2f} km")

    def clear(self):
        self.selection.clear_route()
        self.distance_value.setText("— km")
        self.angle_value.setText("Central angle: —")
        self.endpoint_status.setText("Route cleared.")


# ============================================================
# RESOURCE INTELLIGENCE AND UNKNOWN FRONTIER
# ============================================================

def _science_array_summary(path):
    # Sample a 2-D NumPy array without loading the full high-resolution grid.
    path = Path(path)
    if not path.is_file():
        return None
    try:
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2:
            return None
        step = max(1, int(np.sqrt(array.size / 200000)))
        sample = np.asarray(array[::step, ::step], dtype=np.float64)
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return None
        return {
            "shape": tuple(array.shape),
            "min": float(np.min(sample)),
            "max": float(np.max(sample)),
            "mean": float(np.mean(sample)),
            "median": float(np.median(sample)),
            "p95": float(np.percentile(sample, 95)),
            "sample_count": int(sample.size),
        }
    except Exception:
        return None


def _meda_record_count(path):
    # Count archived CSV rows without requiring pandas.
    path = Path(path)
    if not path.is_file():
        return 0
    try:
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            return sum(1 for row in csv.DictReader(handle) if row)
    except Exception:
        return 0


class ResourceIntelligencePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setSpacing(12)

        title = QLabel("RESOURCE INTELLIGENCE")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        intro = QLabel(
            "Evidence-first resource screening. This page separates mapped terrain "
            "evidence from resource measurements that are not currently loaded."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        root.addWidget(intro)

        meda_dir = BASE_DIR / "data" / "meda"
        meda_files = list(meda_dir.glob("*.csv")) if meda_dir.exists() else []
        meda_records = sum(_meda_record_count(p) for p in meda_files)
        slope = _science_array_summary(GLOBAL_SLOPE_PATH)
        elevation = _science_array_summary(GLOBAL_ELEVATION_PATH)
        jezero_cost = _science_array_summary(PROCESSED_DIR / "jezero_terrain_cost.npy")

        items = [
            (
                "GLOBAL TERRAIN EVIDENCE", "AVAILABLE" if elevation and slope else "PARTIAL",
                (f"MOLA elevation and derived global slope are present. Sampled elevation mean: "
                 f"{elevation['mean']:.1f} m; sampled mean slope: {slope['mean']:.3f} degrees."
                 if elevation and slope else "One or more global terrain arrays are missing."),
                "Terrain describes topography. It does not prove that a resource exists."
            ),
            (
                "WATER / SUBSURFACE ICE", "NOT MEASURED",
                "No dedicated neutron, radar or spectral ice-evidence layer is connected.",
                "Do not interpret low elevation, image colour or terrain shape as proof of water or ice."
            ),
            (
                "MINERAL COMPOSITION", "NOT LOADED",
                "No calibrated compositional spectroscopy dataset is connected to this page.",
                "Mineral identification requires appropriate spectral observations and provenance."
            ),
            (
                "REGIONAL SITE SCREENING", "PRELIMINARY" if jezero_cost else "NOT AVAILABLE",
                (f"Jezero terrain-cost grid is present; sampled mean relative cost: {jezero_cost['mean']:.3f}."
                 if jezero_cost else "No Jezero terrain-cost grid was found."),
                "Relative terrain cost is an exploratory index, not a resource score or safety certification."
            ),
            (
                "ARCHIVED MEDA OBSERVATIONS", "RAW DATA PRESENT" if meda_records else "NOT LOADED",
                f"{len(meda_files)} local CSV file(s); {meda_records:,} observation records detected.",
                "Available ATS channels are raw engineering counts, not calibrated weather or temperature values."
            ),
            (
                "EVIDENCE CONFIDENCE", "SOURCE-AWARE",
                "Global imagery, elevation, derived terrain and archived instrument records are identified separately.",
                "Unknown or unmeasured values remain unknown; no resource abundance is inferred."
            ),
        ]

        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (heading, status, body, limitation) in enumerate(items):
            frame, layout = add_panel(heading, body)
            tag = QLabel(status)
            tag.setObjectName("warningText")
            tag.setWordWrap(True)
            layout.addWidget(tag)
            detail = QLabel(limitation)
            detail.setObjectName("muted")
            detail.setWordWrap(True)
            layout.addWidget(detail)
            grid.addWidget(frame, i // 2, i % 2)
        root.addLayout(grid, 1)


class UnknownFrontierPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("UNKNOWN FRONTIER | SCIENTIFIC DATA GAPS")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        intro = QLabel(
            "A responsible survival system distinguishes unavailable measurements "
            "from available evidence. These are the current limits of the local project data."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        root.addWidget(intro)

        meda_dir = BASE_DIR / "data" / "meda"
        meda_files = list(meda_dir.glob("*.csv")) if meda_dir.exists() else []
        meda_records = sum(_meda_record_count(p) for p in meda_files)
        gaps = [
            ("RADIATION DOSE", "NOT AVAILABLE", "No local radiation instrument dataset or validated radiation model is connected."),
            ("ATMOSPHERIC PRESSURE", "NOT CALIBRATED", "MEDA ATS raw engineering-count files are present, but they do not provide a calibrated pressure product."),
            ("TEMPERATURE & WEATHER", "NOT CALIBRATED", "Archived MEDA ATS records are available, but calibrated temperature and weather products are not connected."),
            ("WATER / SUBSURFACE ICE", "NOT AVAILABLE", "The current Viking imagery and MOLA elevation layers do not measure subsurface ice abundance."),
            ("MINERAL COMPOSITION", "NOT AVAILABLE", "No compositional spectroscopy product is currently loaded into this application."),
            ("HUMAN HABITABILITY", "NOT CERTIFIED", "This research prototype is not a validated human-safety assessment or mission certification system."),
            ("ARCHIVED MEDA ATS", "RAW RECORDS FOUND" if meda_records else "NOT FOUND", f"{len(meda_files)} local CSV file(s), {meda_records:,} records. Raw counts only; calibration is required for physical interpretation."),
        ]

        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (name, status, detail_text) in enumerate(gaps):
            frame, layout = add_panel(name, detail_text)
            tag = QLabel(status)
            tag.setObjectName("warningText")
            tag.setWordWrap(True)
            layout.addWidget(tag)
            grid.addWidget(frame, i // 3, i % 3)
        root.addLayout(grid, 1)


class ScienceLabPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("MARTIAN SCIENCE LAB | LIVE LOCAL DATA")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        intro = QLabel(
            "Dataset inventory and numerical summaries calculated from files currently stored "
            "in this project's data folders. Grid statistics below are sampled."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        root.addWidget(intro)

        elevation = _science_array_summary(GLOBAL_ELEVATION_PATH)
        globe_elevation = _science_array_summary(GLOBE_ELEVATION_PATH)
        slope = _science_array_summary(GLOBAL_SLOPE_PATH)
        jezero_slope = _science_array_summary(PROCESSED_DIR / "jezero_slope_degrees.npy")
        jezero_rough = _science_array_summary(PROCESSED_DIR / "jezero_roughness.npy")
        jezero_cost = _science_array_summary(PROCESSED_DIR / "jezero_terrain_cost.npy")

        meda_dir = BASE_DIR / "data" / "meda"
        meda_files = list(meda_dir.glob("*.csv")) if meda_dir.exists() else []
        meda_records = sum(_meda_record_count(p) for p in meda_files)

        def stat_text(label, stats, unit=""):
            if not stats:
                return f"{label}: NOT FOUND"
            return (
                f"{label}: {stats['shape'][0]:,} x {stats['shape'][1]:,} grid | "
                f"sampled min {stats['min']:.2f}{unit} | sampled max {stats['max']:.2f}{unit} | "
                f"sampled mean {stats['mean']:.2f}{unit}"
            )

        try:
            import rasterio
            if IMAGE_PATH.is_file():
                with rasterio.open(IMAGE_PATH) as raster:
                    image_info = (
                        f"Viking MDIM: PRESENT | {raster.width:,} x {raster.height:,} pixels | "
                        f"{raster.count} band(s) | {IMAGE_PATH.stat().st_size / (1024**2):.1f} MB"
                    )
            else:
                image_info = "Viking MDIM: NOT FOUND"
        except Exception as error:
            image_info = f"Viking MDIM: file present, raster details unavailable ({error})"

        cards = [
            ("VIKING ORBITER GLOBAL MOSAIC", "GLOBAL IMAGE DATASET", image_info),
            ("MOLA MEGDR | GLOBAL ELEVATION", "GLOBAL TOPOGRAPHY",
             stat_text("4 ppd elevation", elevation, " m") + "\n" + stat_text("360 x 180 globe array", globe_elevation, " m")),
            ("DERIVED GLOBAL SLOPE", "COMPUTED FROM MOLA ELEVATION", stat_text("Global slope", slope, " degrees")),
            ("JEZERO REGIONAL TERRAIN", "LOCAL HIGH-RESOLUTION ANALYSIS",
             stat_text("Slope", jezero_slope, " degrees") + "\n" + stat_text("Roughness", jezero_rough) + "\n" + stat_text("Relative terrain cost", jezero_cost)),
            ("MEDA ARCHIVED OBSERVATIONS", "PERSEVERANCE ROVER | PDS CSV",
             f"{len(meda_files)} CSV file(s) detected | {meda_records:,} observation records. ATS values are raw engineering counts, not calibrated temperatures."),
            ("DATA PROVENANCE", "MEASURED / DERIVED / UNAVAILABLE",
             "Viking MDIM = imagery; MOLA MEGDR = elevation; slope and Jezero cost = derived arrays; MEDA ATS = archived raw instrument records."),
        ]

        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (heading, value, detail_text) in enumerate(cards):
            frame, layout = add_panel(heading, value)
            detail = QLabel(detail_text)
            detail.setObjectName("muted")
            detail.setWordWrap(True)
            layout.addWidget(detail)
            grid.addWidget(frame, i // 2, i % 2)
        root.addLayout(grid)

        chart_panel, chart_layout = add_panel(
            "SCIENTIFIC DATA DISTRIBUTIONS",
            "Sampled grid-value distributions; these plots are not global hazard probabilities."
        )
        self.figure = Figure(figsize=(10, 3.5), dpi=100, facecolor="#0b1420")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(self.canvas)

        ax1, ax2 = self.figure.subplots(1, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor("#101c2a")
            ax.tick_params(colors="#d6e2ef", labelsize=8)
            ax.xaxis.label.set_color("#d6e2ef")
            ax.yaxis.label.set_color("#d6e2ef")
            ax.title.set_color("#f3f6fb")
            for spine in ax.spines.values():
                spine.set_color("#40536b")
            ax.grid(True, color="#40536b", alpha=0.3)

        if GLOBAL_SLOPE_PATH.is_file():
            values = np.load(GLOBAL_SLOPE_PATH, mmap_mode="r")
            sample = np.asarray(values[::2, ::2], dtype=float).ravel()
            sample = sample[np.isfinite(sample)]
            if sample.size:
                ax1.hist(sample, bins=40)
                ax1.set_title("Global MOLA-derived slope")
                ax1.set_xlabel("Slope (degrees)")
                ax1.set_ylabel("Sample count")
            else:
                ax1.text(0.5, 0.5, "No valid slope values", ha="center", va="center", transform=ax1.transAxes, color="#d6e2ef")
        else:
            ax1.text(0.5, 0.5, "Global slope array not found", ha="center", va="center", transform=ax1.transAxes, color="#d6e2ef")

        jezero_cost_path = PROCESSED_DIR / "jezero_terrain_cost.npy"
        if jezero_cost_path.is_file():
            values = np.load(jezero_cost_path, mmap_mode="r")
            sample = np.asarray(values[::8, ::8], dtype=float).ravel()
            sample = sample[np.isfinite(sample)]
            if sample.size:
                ax2.hist(sample, bins=40)
                ax2.set_title("Jezero relative terrain cost")
                ax2.set_xlabel("Relative cost (0 to 1)")
                ax2.set_ylabel("Sample count")
            else:
                ax2.text(0.5, 0.5, "No valid Jezero values", ha="center", va="center", transform=ax2.transAxes, color="#d6e2ef")
        else:
            ax2.text(0.5, 0.5, "Jezero terrain-cost array not found", ha="center", va="center", transform=ax2.transAxes, color="#d6e2ef")

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()
        root.addWidget(chart_panel, 1)

        note, note_layout = add_panel("SCIENTIFIC INTERPRETATION & LIMITATIONS")
        note_text = QLabel(
            "GLOBAL FOUNDATION: Viking MDIM supplies global surface imagery; MOLA MEGDR supplies global elevation. "
            "The derived slope grid is calculated from elevation.\n\n"
            "JEZERO: slope, roughness and relative terrain cost are regional derived products. "
            "The terrain-cost index is an exploratory planning layer, not a certified rover or astronaut safety score.\n\n"
            "MEDA: the local ATS CSV files contain archived raw engineering counts. Do not convert them to degrees Celsius "
            "without the correct calibration products and instrument documentation.\n\n"
            "NOT ESTABLISHED BY THESE DATA: water or ice abundance, mineral composition, radiation dose, "
            "weather conditions, or human habitability."
        )
        note_text.setObjectName("warningText")
        note_text.setWordWrap(True)
        note_layout.addWidget(note_text)
        root.addWidget(note)

# MAIN WINDOW — SEPARATE NAVIGATION PANELS FOR EVERY FEATURE
# ============================================================
class MainWindow(QMainWindow):
    NAV_GROUPS = [
        ("OVERVIEW", [("01  COMMAND CENTRE", "command")]),
        ("SURVIVAL SYSTEMS", [
            ("02  SURVIVAL GUIDE", "survival"),
            ("03  MISSION TWIN", "mission"),
            ("04  HABITAT SYSTEMS", "habitat"),
            ("05  ASTRONAUT EVA", "eva"),
        ]),
        ("FIELD OPERATIONS", [
            ("06  ROVER OPERATIONS", "rover"),
            ("07  MARS ENVIRONMENT", "environment"),
            ("08  TERRAIN INTELLIGENCE", "terrain"),
            ("09  MARS NAV", "navigation"),
        ]),
        ("ORBITAL OPERATIONS", [
            ("10  ORBITAL ASSETS", "orbitals"),
        ]),
        ("SCIENCE & DISCOVERY", [
            ("11  RESOURCE INTELLIGENCE", "resources"),
            ("12  UNKNOWN FRONTIER", "frontier"),
            ("13  MARTIAN SCIENCE LAB", "science"),
            ("14  MEDA OBSERVATIONS", "meda"),
        ]),
    ]
    NAV_ITEMS = [
        ("01  COMMAND CENTRE", "command"),
        ("02  SURVIVAL GUIDE", "survival"),
        ("03  MISSION TWIN", "mission"),
        ("04  HABITAT SYSTEMS", "habitat"),
        ("05  ASTRONAUT EVA", "eva"),
        ("06  ROVER OPERATIONS", "rover"),
        ("07  MARS ENVIRONMENT", "environment"),
        ("08  TERRAIN INTELLIGENCE", "terrain"),
        ("09  MARS NAV", "navigation"),
        ("10  ORBITAL ASSETS", "orbitals"),
        ("11  RESOURCE INTELLIGENCE", "resources"),
        ("12  UNKNOWN FRONTIER", "frontier"),
        ("13  MARTIAN SCIENCE LAB", "science"),
        ("14  MEDA OBSERVATIONS", "meda"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MARTIAN MAP | Interplanetary Survival & Exploration Intelligence")
        self.resize(1600, 960)
        self.setMinimumSize(1180, 760)
        self.selection = MissionSelection()
        self.mission_model = MissionModel()
        self.nav_buttons = []
        self.page_indices = {}

        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # Left navigation rail, grouped by task instead of one long button list.
        sidebar = QFrame()
        self.sidebar = sidebar
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(258)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 20, 16, 16)
        side.setSpacing(7)

        brand = QLabel("MARTIAN <span style='color:#e58a48'>MAP</span>")
        brand.setObjectName("logo")
        brand.setTextFormat(Qt.RichText)
        side.addWidget(brand)
        sub = QLabel("INTERPLANETARY SURVIVAL GUIDE")
        sub.setObjectName("sidebarSub")
        sub.setWordWrap(True)
        side.addWidget(sub)
        side.addSpacing(10)

        for group_name, items in self.NAV_GROUPS:
            group_label = QLabel(group_name)
            group_label.setObjectName("navGroup")
            side.addWidget(group_label)
            for label, _key in items:
                index = len(self.nav_buttons)
                button = QPushButton(label)
                button.setObjectName("navButton")
                button.setProperty("active", False)
                button.setCursor(Qt.PointingHandCursor)
                button.setMinimumHeight(36)
                button.clicked.connect(lambda checked=False, i=index: self.change_page(i))
                side.addWidget(button)
                self.nav_buttons.append(button)

        side.addStretch(1)
        foot = QLabel("GLOBAL BASE  ·  VIKING MDIM + MOLA MEGDR\nSIMULATION / SCIENTIFIC PROTOTYPE")
        foot.setObjectName("sidebarFoot")
        foot.setWordWrap(True)
        side.addWidget(foot)
        main.addWidget(sidebar)

        # Main workspace.
        content = QWidget()
        content.setObjectName("workspace")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 15, 20, 12)
        content_layout.setSpacing(11)

        self.topbar_widget = QWidget()
        topbar = QHBoxLayout(self.topbar_widget)
        self.top_title = QLabel("PLANETARY COMMAND CENTRE")
        self.top_title.setObjectName("topTitle")
        self.top_context = QLabel("MARS EXPLORATION OPERATIONS")
        self.top_context.setObjectName("topContext")
        topbar.addWidget(self.top_title)
        topbar.addSpacing(12)
        topbar.addWidget(self.top_context)
        topbar.addStretch()
        self.system_state = QLabel("● SYSTEM READY")
        self.system_state.setObjectName("online")
        topbar.addWidget(self.system_state)
        content_layout.addWidget(self.topbar_widget)

        self.pages = QStackedWidget()
        self.pages.setObjectName("pageStack")
        content_layout.addWidget(self.pages, 1)

        statusbar = QFrame()
        self.statusbar = statusbar
        statusbar.setObjectName("statusBar")
        status_layout = QHBoxLayout(statusbar)
        status_layout.setContentsMargins(12, 7, 12, 7)
        self.data_status = QLabel("DATA FOUNDATION  ·  VIKING MDIM + MOLA MEGDR")
        self.data_status.setObjectName("statusText")
        self.sim_status = QLabel("SIMULATED SYSTEMS  ·  NOT LIVE TELEMETRY")
        self.sim_status.setObjectName("statusTextMuted")
        status_layout.addWidget(self.data_status)
        status_layout.addStretch()
        status_layout.addWidget(self.sim_status)
        content_layout.addWidget(statusbar)
        main.addWidget(content, 1)

        self.command_page = CommandCentrePage(self.selection)
        self._add_page(self.command_page, "command")
        self._add_page(SurvivalGuidePage(self.selection), "survival")
        self._add_page(MissionTwinPage(self.mission_model), "mission")
        self._add_page(HabitatSystemsPage(self.mission_model), "habitat")
        self._add_page(AstronautEVAPage(), "eva")
        self._add_page(RoverOperationsPage(), "rover")
        self._add_page(GlobalMapPage(self.selection), "environment")
        self._add_page(TerrainIntelligencePage(), "terrain")
        self._add_page(MarsNavigationPage(self.selection, self.activate_globe_pick), "navigation")
        self._add_page(OrbitalAssetsPage(), "orbitals")
        self._add_page(ResourceIntelligencePage(), "resources")
        self._add_page(UnknownFrontierPage(), "frontier")
        self._add_page(ScienceLabPage(), "science")
        self._add_page(MEDAObservationsPage(), "meda")
        self.immersive = False
        self._install_immersive_controls()
        self._apply_style()
        self.change_page(0)
        self.selection.activeChanged.connect(self.update_pick_status)
        self.selection.selectionChanged.connect(self.update_selected_status)
        self.selection.routeChanged.connect(self.update_route_status)

    def _add_page(self, widget, key):
        """Register a page and remember its navigation index."""
        self.page_indices[key] = self.pages.addWidget(widget)

    def _install_immersive_controls(self):
        """Install compact controls inside the existing globe panel."""
        outer = self.command_page.layout()
        self._command_header_layout = outer.itemAt(0).layout()
        self._command_cards_layout = outer.itemAt(1).layout()
        body_layout = outer.itemAt(2).layout()
        self._globe_panel = body_layout.itemAt(0).widget()
        self._command_side_panel = body_layout.itemAt(1).widget()
        panel_layout = self._globe_panel.layout()
        self._globe_panel_heading = panel_layout.itemAt(0).widget()
        self._globe_panel_subtitle = panel_layout.itemAt(1).widget()

        self.normal_immersive_button = QPushButton("⛶  IMMERSIVE MARS")
        self.normal_immersive_button.setObjectName("primaryButton")
        self.normal_immersive_button.setToolTip("Open the Mars globe in a full-screen mission view (F11).")
        self.normal_immersive_button.clicked.connect(self.enter_immersive_mode)
        self._command_header_layout.addWidget(self.normal_immersive_button)

        self.immersive_bar = QFrame()
        self.immersive_bar.setObjectName("immersiveBar")
        bar_layout = QHBoxLayout(self.immersive_bar)
        bar_layout.setContentsMargins(12, 8, 12, 8)
        bar_layout.setSpacing(8)
        brand = QLabel("MARS  /  EXPLORATION INTELLIGENCE")
        brand.setObjectName("immersiveBrand")
        bar_layout.addWidget(brand, 1)

        self.pick_start_immersive = QPushButton("＋ PICK START")
        self.pick_dest_immersive = QPushButton("＋ PICK DESTINATION")
        self.clear_route_immersive = QPushButton("CLEAR ROUTE")
        self.survival_immersive = QPushButton("SURVIVAL INTELLIGENCE")
        self.exit_immersive = QPushButton("EXIT  ✕")
        for button in (self.pick_start_immersive, self.pick_dest_immersive,
                       self.clear_route_immersive, self.survival_immersive, self.exit_immersive):
            button.setCursor(Qt.PointingHandCursor)
            bar_layout.addWidget(button)

        self.pick_start_immersive.clicked.connect(lambda: self._arm_globe_pick("start"))
        self.pick_dest_immersive.clicked.connect(lambda: self._arm_globe_pick("destination"))
        self.clear_route_immersive.clicked.connect(self.selection.clear_route)
        self.survival_immersive.clicked.connect(lambda: self.change_page(self.page_indices["survival"]))
        self.exit_immersive.clicked.connect(self.exit_immersive_mode)
        panel_layout.insertWidget(2, self.immersive_bar)
        self.immersive_bar.hide()

    def _arm_globe_pick(self, target):
        self.selection.arm(target)
        self.system_state.setText(f"● PICK {target.upper()} ON THE MARS GLOBE")

    @staticmethod
    def _set_nested_layout_visible(layout, visible):
        if layout is None:
            return
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget() is not None:
                item.widget().setVisible(visible)
            elif item.layout() is not None:
                MainWindow._set_nested_layout_visible(item.layout(), visible)

    def enter_immersive_mode(self):
        """Show the command globe as a full-screen mission view."""
        if self.immersive:
            return
        command_index = self.page_indices["command"]
        self.pages.setCurrentIndex(command_index)
        self.immersive = True

        self.sidebar.hide()
        self.topbar_widget.hide()
        self.statusbar.hide()
        self._set_nested_layout_visible(self._command_header_layout, False)
        self._set_nested_layout_visible(self._command_cards_layout, False)
        self._command_side_panel.hide()
        self._globe_panel_heading.hide()
        self._globe_panel_subtitle.hide()
        self.immersive_bar.show()
        self.command_page.layout().setContentsMargins(0, 0, 0, 0)
        self.command_page.layout().setSpacing(0)
        self._globe_panel.layout().setContentsMargins(7, 7, 7, 7)
        self._globe_panel.layout().setSpacing(5)
        self.command_page.globe.setMinimumHeight(0)
        self.normal_immersive_button.hide()
        self.showFullScreen()

    def exit_immersive_mode(self):
        """Restore the standard application layout from full-screen mode."""
        if not self.immersive:
            return
        self.immersive = False
        self.immersive_bar.hide()
        self._set_nested_layout_visible(self._command_header_layout, True)
        self._set_nested_layout_visible(self._command_cards_layout, True)
        self._command_side_panel.show()
        self._globe_panel_heading.show()
        self._globe_panel_subtitle.show()
        self.sidebar.show()
        self.topbar_widget.show()
        self.statusbar.show()
        self.command_page.layout().setContentsMargins(2, 2, 2, 2)
        self.command_page.layout().setSpacing(10)
        self._globe_panel.layout().setContentsMargins(14, 12, 14, 12)
        self._globe_panel.layout().setSpacing(7)
        self.normal_immersive_button.show()
        self.showMaximized()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.immersive:
            self.exit_immersive_mode()
            event.accept()
            return
        if event.key() == Qt.Key_F11:
            if self.immersive:
                self.exit_immersive_mode()
            else:
                self.enter_immersive_mode()
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_style(self):
        self.setStyleSheet("""
        QMainWindow {
                background: qradialgradient(cx:0.78, cy:0.05, radius:1.25, fx:0.78, fy:0.05,
                    stop:0 #243a54, stop:0.42 #101b2d, stop:1 #070c16);
            }
            QFrame#sidebar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #192d43, stop:0.48 #101e30, stop:1 #0a1422);
                border-right: 1px solid #36516a;
            }
            QWidget#workspace {
                background: qradialgradient(cx:0.84, cy:0.05, radius:1.25, fx:0.84, fy:0.05,
                    stop:0 #1a2d43, stop:0.45 #0c1728, stop:1 #070c15);
            }
            QFrame#panel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a2d42, stop:0.52 #111e30, stop:1 #0d1725);
                border: 1px solid #344d65;
                border-radius: 12px;
            }
            QFrame#metricCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #263b52, stop:0.48 #172a40, stop:1 #111d30);
                border: 1px solid #3b5872;
                border-radius: 11px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #29445d, stop:1 #17283d);
                color: #e3edf6;
                border: 1px solid #45627c;
                border-radius: 8px;
                padding: 8px 11px;
                font-size: 10px;
                font-weight: 800;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3d607b, stop:1 #243c55);
                border: 1px solid #f0a16d;
                color: #fff1e4;
            }
            QPushButton:pressed { background: #263b50; }
            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ef9b62, stop:0.5 #c76e49, stop:1 #85445a);
                border: 1px solid #ffc18d;
                color: #fff8f1;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffb77a, stop:1 #b95d56);
            }
            QFrame#survivalHero {
                background: qradialgradient(cx:0.88, cy:0.12, radius:1.2, fx:0.88, fy:0.12,
                    stop:0 #354b68, stop:0.45 #1a2c45, stop:1 #111a2b);
                border: 1px solid #55718d;
                border-radius: 16px;
            }
            QFrame#decisionCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1b3048, stop:1 #101a2a);
                border: 1px solid #3b5872;
                border-radius: 13px;
            }
            QFrame#decisionAction {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e3a48, stop:1 #18283d);
                border: 1px solid #34586b;
                border-radius: 8px;
            }
            QFrame#priorityTile {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1b3048, stop:1 #111b2b);
                border: 1px solid #3a536e;
                border-radius: 11px;
            }
            QFrame#immersiveBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(12,25,42,245), stop:0.58 rgba(23,43,63,238), stop:1 rgba(54,40,48,235));
                border: 1px solid #58738a;
                border-radius: 11px;
            }
            QLabel#heroTitle { color: #ffe1c6; font-size: 15px; font-weight: 900; letter-spacing: 1.2px; }
            QLabel#heroText { color: #d4e4f0; font-size: 11px; }
            QLabel#heroWarning { color: #ffc778; font-size: 9px; font-weight: 800; }
            QLabel#heroCount { color: #8de4e0; font-family: Consolas; font-size: 10px; font-weight: 900; }
            QLabel#decisionHeading { color: #e8f1f8; font-size: 10px; font-weight: 900; letter-spacing: 0.6px; }
            QLabel#decisionSummary { color: #c5d5e3; font-size: 10px; }
            QLabel#decisionActionLabel { color: #7fded8; font-size: 8px; font-weight: 900; letter-spacing: 0.8px; }
            QLabel#decisionActionText { color: #e5edf4; font-size: 9px; }
            QLabel#decisionDetails { color: #9fb3c5; background: rgba(6,13,23,100); border: 1px solid #2d455b; border-radius: 7px; padding: 9px; font-size: 9px; }
            QPushButton#evidenceButton { background: transparent; color: #8fd9e1; border: 1px solid #34596b; padding: 6px 8px; text-align: left; }
            QLabel#priorityHeading { color: #f0d8c5; font-size: 9px; font-weight: 900; }
            QLabel#priorityBody { color: #b8c9d8; font-size: 9px; }
            QLabel#priorityTag { color: #ffc777; font-size: 8px; font-weight: 900; border: 1px solid #715a3d; border-radius: 6px; padding: 4px 6px; }
            QLabel#simulationBoundary { color: #b8c9d8; background: rgba(8,17,29,100); border: 1px solid #30485e; border-radius: 8px; padding: 11px; }
            QLabel#limitsText { color: #b8c9d8; background: rgba(8,17,29,100); border: 1px solid #30485e; border-radius: 8px; padding: 11px; }
            QLabel#immersiveBrand { color: #ffd7b7; font-size: 10px; font-weight: 900; letter-spacing: 1px; }
            
            /* MARTIAN READABILITY OVERRIDES */
            QLabel { color: #EDF4FC; font-size: 11px; }
            QLabel#muted { color: #C6D5E4; font-size: 11px; }
            QLabel#pageTitle { color: #FFC095; font-size: 20px; font-weight: 900; }
            QLabel#panelTitle { color: #FFD0AD; font-size: 11px; }
            QLabel#metricTitle { color: #C0D0E0; font-size: 9px; }
            QLabel#metricValue { color: #FFFFFF; font-size: 19px; }
            QLabel#topContext, QLabel#sidebarSub, QLabel#navGroup, QLabel#sidebarFoot { color: #BBCDDD; }
            QLabel#statusText, QLabel#statusTextMuted { color: #D0DFEC; font-size: 9px; }
            QLabel#locationInfo { color: #EFF5FB; background: #142940; border: 1px solid #526C86; }
            QLabel#routeSummary { color: #A5F4EA; background: #123138; }
            QLabel#heroText, QLabel#decisionSummary, QLabel#decisionActionText, QLabel#priorityBody { color: #E6EEF6; }
            QLabel#decisionHeading, QLabel#priorityHeading { color: #FFFFFF; }
            QLabel#decisionDetails, QLabel#simulationBoundary, QLabel#limitsText { color: #E3EDF6; background: #101F32; border: 1px solid #526A82; }
            QLabel#warningText, QLabel#heroWarning { color: #FFD18D; }
            QPushButton { color: #F5F8FC; background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #29445D,stop:1 #172A40); border: 1px solid #56728C; padding: 9px 12px; }
            QPushButton:hover { color: #FFFFFF; background: #385B78; border: 1px solid #8BE3E8; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { color: #FFFFFF; background: #192D43; border: 1px solid #607B95; }
            QComboBox QAbstractItemView { color: #FFFFFF; background: #172B40; selection-background-color: #385B79; selection-color: #FFFFFF; }
            QListWidget { color: #F0F5FA; background: #101F32; border: 1px solid #526A82; }
            QListWidget::item { color: #E8F0F7; padding: 10px; }
            QListWidget::item:selected { color: #FFFFFF; background: #345977; }
            QTableWidget { color: #F0F5FA; background: #101F32; alternate-background-color: #1C3046; gridline-color: #526A82; }
            QHeaderView::section { color: #FFFFFF; background: #29445D; padding: 8px; }
            QProgressBar { color: #FFFFFF; background: #101F32; border: 1px solid #607B95; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #D97948,stop:1 #58D9D2); }
            QToolTip { color: #FFFFFF; background: #1C3048; border: 1px solid #7BA4C5; }

        """)


    def change_page(self, index):
        """Switch to a navigation page and keep the application chrome in sync."""
        if not isinstance(index, int) or not 0 <= index < self.pages.count():
            return

        command_index = self.page_indices["command"]
        if self.immersive and index != command_index:
            self.exit_immersive_mode()

        self.pages.setCurrentIndex(index)
        label = self.NAV_ITEMS[index][0]
        self.top_title.setText(label)
        for i, button in enumerate(self.nav_buttons):
            button.setProperty("active", i == index)
            button.style().unpolish(button)
            button.style().polish(button)
        if index in (1, 2, 3, 4):
            group_context = "SURVIVAL OPERATIONS"
        elif index in (5, 6, 7, 8, 9):
            group_context = "FIELD & ORBITAL OPERATIONS"
        elif index in (10, 11, 12, 13):
            group_context = "SCIENCE & DISCOVERY"
        else:
            group_context = "MISSION OVERVIEW"
        self.top_context.setText(group_context)

    def activate_globe_pick(self, target):
        self.selection.arm(target)
        self.change_page(self.page_indices["command"])
        self.system_state.setText(f"● PICK {target.upper()} ON GLOBE")

    def update_pick_status(self):
        if self.selection.active_target:
            self.system_state.setText(f"● PICK {self.selection.active_target.upper()} ON GLOBE")
        else:
            self.update_route_status()

    def update_selected_status(self):
        if self.selection.selected_point:
            self.system_state.setText(f"● SELECTED: {self.selection.selected_name.upper()}")

    def update_route_status(self):
        if self.selection.start_point and self.selection.destination_point:
            d = great_circle_distance_km(self.selection.start_point, self.selection.destination_point)
            self.system_state.setText(f"● ROUTE READY  |  {d:.1f} KM GREAT-CIRCLE")
        elif not self.selection.active_target:
            self.system_state.setText("● SYSTEM READY")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MARTIAN MAP")
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.enter_immersive_mode)
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
