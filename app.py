from pathlib import Path
import csv
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# MARTIAN MAP — Browser edition of the original 14-module app.
# The original PySide6 desktop application remains unchanged in main.py.
# ============================================================
st.set_page_config(
    page_title="MARTIAN MAP | Interplanetary Survival Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
GLOBAL = DATA / "global"
PROC = DATA / "processed"
MEDA = DATA / "meda"
MARS_RADIUS_KM = 3389.5

NAV = {
    "OVERVIEW": [("01  COMMAND CENTRE", "Command Centre")],
    "SURVIVAL SYSTEMS": [
        ("02  SURVIVAL GUIDE", "Survival Guide"),
        ("03  MISSION TWIN", "Mission Twin"),
        ("04  HABITAT SYSTEMS", "Habitat Systems"),
        ("05  ASTRONAUT EVA", "Astronaut EVA"),
    ],
    "FIELD OPERATIONS": [
        ("06  ROVER OPERATIONS", "Rover Operations"),
        ("07  MARS ENVIRONMENT", "Mars Environment"),
        ("08  TERRAIN INTELLIGENCE", "Terrain Intelligence"),
        ("09  MARS NAV", "Mars Nav"),
    ],
    "ORBITAL OPERATIONS": [("10  ORBITAL ASSETS", "Orbital Assets")],
    "SCIENCE & DISCOVERY": [
        ("11  RESOURCE INTELLIGENCE", "Resource Intelligence"),
        ("12  UNKNOWN FRONTIER", "Unknown Frontier"),
        ("13  MARTIAN SCIENCE LAB", "Martian Science Lab"),
        ("14  MEDA OBSERVATIONS", "MEDA Observations"),
    ],
}
PAGES = [page for group in NAV.values() for _, page in group]

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

ORBITAL_ASSETS = [
    {"name": "Mars Odyssey", "short": "ODYSSEY", "operator": "NASA", "role": "Science mapping and relay support", "color": "#70c7ff", "radius": 1.30, "inc": 32, "raan": 8, "phase": 0.2, "rate": 0.90},
    {"name": "Mars Reconnaissance Orbiter", "short": "MRO", "operator": "NASA", "role": "High-resolution imaging, climate science and relay support", "color": "#65e6d1", "radius": 1.34, "inc": 18, "raan": 62, "phase": 1.1, "rate": 1.10},
    {"name": "Mars Express", "short": "MARS EXPRESS", "operator": "ESA", "role": "Atmospheric, surface and subsurface science", "color": "#ffca78", "radius": 1.39, "inc": 42, "raan": 115, "phase": 2.0, "rate": 0.78},
    {"name": "ExoMars Trace Gas Orbiter", "short": "TGO", "operator": "ESA", "role": "Trace-gas science and communications relay", "color": "#b9a2ff", "radius": 1.43, "inc": 24, "raan": 175, "phase": 0.8, "rate": 1.02},
    {"name": "Emirates Mars Mission (Hope)", "short": "HOPE / EMM", "operator": "UAE Space Agency", "role": "Upper-atmosphere and climate science", "color": "#ff9c83", "radius": 1.48, "inc": 12, "raan": 220, "phase": 2.7, "rate": 0.67},
    {"name": "Tianwen-1 Orbiter", "short": "TIANWEN-1", "operator": "CNSA", "role": "Remote sensing, imaging and relay support", "color": "#d8e67a", "radius": 1.52, "inc": 28, "raan": 285, "phase": 1.7, "rate": 0.95},
]
NATURAL_SATELLITES = [
    {"name": "Phobos", "short": "PHOBOS", "operator": "Natural moon", "role": "Inner natural satellite; schematic orbit", "color": "#d6d0c5", "radius": 1.24, "inc": 4, "raan": 35, "phase": 1.4, "rate": 1.85},
    {"name": "Deimos", "short": "DEIMOS", "operator": "Natural moon", "role": "Outer natural satellite; schematic orbit", "color": "#aaaeb7", "radius": 1.59, "inc": 2, "raan": 92, "phase": 0.3, "rate": 0.72},
]
ORBITAL_OBJECTS = ORBITAL_ASSETS + NATURAL_SATELLITES
MEDA_FILES = {
    "Sol 1": MEDA / "WE__0001___________ESD_ATS_________________J02.csv",
    "Sol 1740": MEDA / "WE__1740___________ESD_ATS_________________J01.csv",
}
MEDA_CHANNELS = [f"ATS_ATS{i}" for i in range(1, 6)]

# -------------------- Design system --------------------
st.markdown("""
<style>
.stApp {background:linear-gradient(145deg,#070d17 0%,#0b1625 58%,#101d2e 100%);color:#e9f1fa}
[data-testid="stHeader"] {background:rgba(7,13,23,.96)}
[data-testid="stSidebar"] {background:#0e1b2b;border-right:1px solid #304962}
[data-testid="stSidebar"] * {color:#e6eef8!important}
[data-testid="stSidebar"] button {border:1px solid #3a5873!important;border-radius:8px!important;min-height:34px}
[data-testid="stSidebar"] button[kind="primary"] {background:linear-gradient(110deg,#d87542,#8b5365)!important;border-color:#e7a06e!important}
h1,h2,h3 {color:#f1f5fb!important}
p,li,label {color:#dce7f2}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#1a3048,#122338);border:1px solid #395572;padding:14px 16px;border-radius:12px;min-height:100px}
div[data-testid="stMetricLabel"] {color:#a9c1d5!important}
div[data-testid="stMetricValue"] {color:#fff1e4!important}
.stButton>button {border-radius:8px;border:1px solid #4b6881;color:#edf4fb;background:#1b344d;font-weight:650}
.stButton>button:hover {border-color:#f2a06a;color:#fff}
div[data-testid="stExpander"] {border:1px solid #36516b;border-radius:10px;background:#102035}
hr {border-color:#38516a}
.mm-sub {color:#a9bfd2;font-size:13px;margin-bottom:15px}
.mm-panel {background:linear-gradient(145deg,#172d43,#101f31);border:1px solid #36536e;border-radius:12px;padding:12px 15px}
.mm-label {color:#f2a06a;font-weight:850;letter-spacing:.08em;font-size:12px}
.mm-foot {display:flex;justify-content:space-between;gap:10px;color:#8799aa;font-size:10px;border-top:1px solid #30475f;padding-top:12px;margin-top:20px}
</style>
""", unsafe_allow_html=True)

# -------------------- Shared helpers --------------------
@st.cache_data(show_spinner=False)
def load_array(path_text):
    path = Path(path_text)
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")

def array_stats(path):
    arr = load_array(str(path))
    if arr is None or arr.ndim != 2:
        return None
    stride = max(1, int(math.ceil(max(arr.shape) / 350)))
    sample = np.asarray(arr[::stride, ::stride], dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        return None
    return {"shape": arr.shape, "min": float(sample.min()), "max": float(sample.max()),
            "mean": float(sample.mean()), "median": float(np.median(sample)),
            "p95": float(np.percentile(sample, 95)), "samples": int(sample.size)}

def great_circle_km(a, b):
    lat1, lat2 = np.radians(a[0]), np.radians(b[0])
    dlon = np.radians(b[1] - a[1])
    h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return float(2*MARS_RADIUS_KM*np.arctan2(np.sqrt(np.clip(h,0,1)), np.sqrt(1-np.clip(h,0,1))))

def xyz(lat, lon, radius=1.06):
    la, lo = np.radians(lat), np.radians(lon)
    return radius*np.cos(la)*np.cos(lo), radius*np.cos(la)*np.sin(lo), radius*np.sin(la)

def orbit_position(asset, angle):
    radius = asset["radius"]
    inc, raan = np.radians(asset["inc"]), np.radians(asset["raan"])
    x0, y0, z0 = radius*np.cos(angle), radius*np.sin(angle)*np.cos(inc), radius*np.sin(angle)*np.sin(inc)
    return x0*np.cos(raan)-y0*np.sin(raan), x0*np.sin(raan)+y0*np.cos(raan), z0

def page_header(title, subtitle):
    st.markdown(f"<div style='font-size:27px;font-weight:900;color:#f2a06a'>{title}</div><div class='mm-sub'>{subtitle}</div>", unsafe_allow_html=True)

def panel(title, subtitle=""):
    st.markdown(f"<div class='mm-panel'><div class='mm-label'>{title}</div><div style='color:#a9bfd2;font-size:12px'>{subtitle}</div></div>", unsafe_allow_html=True)

def footer():
    st.markdown("<div class='mm-foot'><span>DATA FOUNDATION · VIKING MDIM + MOLA MEGDR</span><span>SIMULATED SYSTEMS · NOT LIVE TELEMETRY</span></div>", unsafe_allow_html=True)

def sampled_grid(array, max_rows=110, max_cols=250):
    if array is None:
        return None
    step_r = max(1, int(math.ceil(array.shape[0]/max_rows)))
    step_c = max(1, int(math.ceil(array.shape[1]/max_cols)))
    return np.asarray(array[::step_r, ::step_c], dtype=float)

def global_layer_figure(path, title, colorscale, colorbar_title):
    arr = load_array(str(path))
    if arr is None:
        return None
    z = sampled_grid(arr)
    lats = np.linspace(90, -90, z.shape[0])
    lons = np.linspace(0, 360, z.shape[1])
    fig = go.Figure(go.Heatmap(z=z, x=lons, y=lats, colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        hovertemplate="East longitude %{x:.1f}°<br>Latitude %{y:.1f}°<br>Value %{z:.2f}<extra></extra>"))
    fig.update_layout(title=title, height=470, margin=dict(l=20,r=20,t=45,b=25),
        paper_bgcolor="#0d1928", plot_bgcolor="#0d1928", font=dict(color="#dce7f2"),
        xaxis_title="East longitude (degrees)", yaxis_title="Planetocentric latitude (degrees)")
    return fig

def globe_figure(start_name=None, dest_name=None, show_orbits=True):
    elevation = load_array(str(PROC/"mars_mola_globe_360x180.npy"))
    if elevation is None:
        elevation = load_array(str(PROC/"mars_global_elevation_4ppd.npy"))
    if elevation is None:
        return None
    elevation = np.asarray(elevation, dtype=np.float32)
    row_step = max(1, int(math.ceil(elevation.shape[0]/90)))
    col_step = max(1, int(math.ceil(elevation.shape[1]/180)))
    e = elevation[::row_step, ::col_step]
    lat = np.linspace(90, -90, e.shape[0])
    lon = np.linspace(0, 360, e.shape[1], endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    radius = 1.0 + np.clip(e, -10000, 22000)/3396000.0*10.0
    la, lo = np.radians(lat_grid), np.radians(lon_grid)
    x, y, z = radius*np.cos(la)*np.cos(lo), radius*np.cos(la)*np.sin(lo), radius*np.sin(la)
    fig = go.Figure(go.Surface(x=x,y=y,z=z,surfacecolor=e,
        colorscale=[[0,"#261e24"],[0.2,"#693522"],[0.48,"#a9502d"],[0.75,"#d88142"],[1,"#f3c68e"]],
        cmin=float(np.nanpercentile(e,2)), cmax=float(np.nanpercentile(e,98)),
        showscale=False, hoverinfo="skip", name="MOLA MEGDR elevation relief",
        lighting=dict(ambient=0.62,diffuse=0.8,specular=0.08,roughness=0.9)))
    xs,ys,zs,texts,colors = [],[],[],[],[]
    for place in MARS_PLACES:
        px,py,pz = xyz(place["lat"],place["lon"],1.075)
        xs.append(px); ys.append(py); zs.append(pz); texts.append(place["name"])
        colors.append("#ffd27a" if "landing" in place["kind"].lower() else "#63e7e0")
    fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode="markers+text",text=texts,textposition="top center",
        textfont=dict(size=8,color="#f4e7d1"),marker=dict(size=3.5,color=colors,line=dict(color="#15212b",width=1)),
        name="Named Mars locations",hovertemplate="%{text}<extra></extra>"))
    if show_orbits:
        theta = np.linspace(0,2*np.pi,130)
        for asset in ORBITAL_OBJECTS:
            points = np.array([orbit_position(asset,t) for t in theta])
            fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode="lines",
                line=dict(color=asset["color"],width=1.3,dash="dash" if asset in ORBITAL_ASSETS else "dot"),
                opacity=0.45,showlegend=False,hoverinfo="skip"))
            pos = orbit_position(asset,asset["phase"])
            fig.add_trace(go.Scatter3d(x=[pos[0]],y=[pos[1]],z=[pos[2]],mode="markers+text",
                text=[asset["short"]],textposition="top center",textfont=dict(size=8,color=asset["color"]),
                marker=dict(size=4,color=asset["color"]),showlegend=False,
                hovertemplate=asset["name"]+"<br>SCHEMATIC POSITION<extra></extra>"))
    if start_name in PLACE_BY_NAME and dest_name in PLACE_BY_NAME:
        a = np.array(xyz(PLACE_BY_NAME[start_name]["lat"],PLACE_BY_NAME[start_name]["lon"],1.07))
        b = np.array(xyz(PLACE_BY_NAME[dest_name]["lat"],PLACE_BY_NAME[dest_name]["lon"],1.07))
        a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
        angle = float(np.arccos(np.clip(np.dot(a,b),-1,1)))
        t = np.linspace(0,1,100)
        if angle < 1e-8:
            vectors = np.tile(a,(len(t),1))
        elif abs(np.sin(angle)) < 1e-8:
            vectors = np.tile(a,(len(t),1))
        else:
            vectors = (np.sin((1-t)*angle)[:,None]*a + np.sin(t*angle)[:,None]*b)/np.sin(angle)
        vectors = vectors/np.linalg.norm(vectors,axis=1)[:,None]*1.085
        fig.add_trace(go.Scatter3d(x=vectors[:,0],y=vectors[:,1],z=vectors[:,2],mode="lines",
            line=dict(color="#ff59b7",width=5),name="Geometric route"))
    fig.update_layout(height=620,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="#0b1422",
        font=dict(color="#dce8f3"),showlegend=False,
        scene=dict(bgcolor="#0b1422",aspectmode="data",
          xaxis=dict(visible=False,range=[-1.7,1.7]),yaxis=dict(visible=False,range=[-1.7,1.7]),
          zaxis=dict(visible=False,range=[-1.7,1.7]),camera=dict(eye=dict(x=1.55,y=1.55,z=0.8))))
    return fig

def read_route_csv(filename):
    path = PROC/filename
    if not path.exists():
        return []
    with path.open("r",newline="",encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

def route_figure(rows,title):
    points=[]
    for row in rows:
        try:
            points.append((float(row["map_x_m"]),float(row["map_y_m"])))
        except (KeyError,ValueError,TypeError):
            continue
    if not points:
        return None
    fig=go.Figure(go.Scatter(x=[p[0] for p in points],y=[p[1] for p in points],mode="lines",
        line=dict(color="#63d8d0",width=3),name=title))
    fig.update_layout(title=title,height=400,paper_bgcolor="#0d1928",plot_bgcolor="#0d1928",
        font=dict(color="#dce7f2"),xaxis_title="Map X (m)",yaxis_title="Map Y (m)")
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    return fig

@st.cache_data(show_spinner=False)
def load_meda_csv(path_text):
    path=Path(path_text)
    if not path.exists():
        return None
    times=[]; channels={c:[] for c in MEDA_CHANNELS}; preview=[]; headers=[]
    with path.open("r",newline="",encoding="utf-8-sig",errors="replace") as handle:
        reader=csv.DictReader(handle)
        headers=reader.fieldnames or []
        for row in reader:
            if not row:
                continue
            clock=str(row.get("LTST","") or "").strip()
            try:
                hh,mm,ss=clock.split()[-1].split(":")
                t=int(hh)+int(mm)/60+float(ss)/3600
            except Exception:
                t=float(len(times))
            times.append(t)
            for channel in MEDA_CHANNELS:
                try:
                    channels[channel].append(float(row.get(channel,"")))
                except (TypeError,ValueError):
                    channels[channel].append(np.nan)
            if len(preview)<40:
                preview.append({key:row.get(key,"") for key in headers[:10]})
    return {"times":np.asarray(times),"channels":{k:np.asarray(v) for k,v in channels.items()},
            "preview":preview,"headers":headers,"count":len(times)}

# -------------------- Session state --------------------
defaults = {
    "current_page":"Command Centre", "selected_place":"Jezero Crater",
    "start_site":"Jezero Crater", "dest_site":"Gale Crater", "mission_report":None,
    "eva":{"status":"STANDBY","oxygen":100.0,"battery":100.0,"co2":450.0,"temperature":22.0,
           "humidity":45.0,"heart_rate":78,"elapsed":0},
    "rover":{"progress":0,"battery":100.0,"status":"STANDBY"},
}
for key,value in defaults.items():
    if key not in st.session_state:
        st.session_state[key]=value

# -------------------- Sidebar: same 5 groups / 14 modules --------------------
st.sidebar.markdown("<div style='font-size:25px;font-weight:900;color:#f1a06a'>🔴 MARTIAN MAP</div><div style='font-size:10px;letter-spacing:.12em;color:#a9bed1;margin:3px 0 17px'>INTERPLANETARY SURVIVAL GUIDE</div>",unsafe_allow_html=True)
for group,items in NAV.items():
    st.sidebar.markdown(f"<div style='font-size:10px;font-weight:850;letter-spacing:.12em;color:#f0a06a;margin:14px 0 5px'>{group}</div>",unsafe_allow_html=True)
    for label,page_name in items:
        if st.sidebar.button(label,key="nav_"+page_name.replace(" ","_"),
            use_container_width=True,type="primary" if st.session_state.current_page==page_name else "secondary"):
            st.session_state.current_page=page_name
            st.rerun()
st.sidebar.markdown("<hr><div style='font-size:10px;color:#9db2c5;line-height:1.7'>GLOBAL BASE · VIKING MDIM + MOLA MEGDR<br>SIMULATION / SCIENTIFIC PROTOTYPE</div>",unsafe_allow_html=True)

page = st.session_state.current_page
left_top,right_top=st.columns([5,1])
with left_top:
    st.markdown(f"<div style='border-bottom:1px solid #30475f;padding:6px 0 13px'><span style='font-size:13px;color:#f4a878;font-weight:850;letter-spacing:.12em'>{PAGES.index(page)+1:02d}  {page.upper()}</span><span style='color:#9eb2c4;font-size:11px;margin-left:16px'>MARS EXPLORATION OPERATIONS</span></div>",unsafe_allow_html=True)
with right_top:
    st.markdown("<div style='text-align:right;color:#75dfc8;font-size:11px;font-weight:800;padding-top:9px'>● SYSTEM READY</div>",unsafe_allow_html=True)

# ============================================================
# 01 COMMAND CENTRE
# ============================================================
if page=="Command Centre":
    page_header("PLANETARY COMMAND CENTRE","Explore Mars · inspect known sites · prepare mission endpoints")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("GLOBAL FOUNDATION","VIKING + MOLA","Imagery and elevation")
    m2.metric("NAMED LOCATIONS",str(len(MARS_PLACES)),"Landing sites and landforms")
    m3.metric("ORBITAL ASSETS","6 ACTIVE","2 natural moons · schematic orbits")
    m4.metric("DATA INTEGRITY","SOURCE-AWARE","Unavailable data stays marked")
    left,right=st.columns([7,3],gap="medium")
    with left:
        panel("MARS GLOBAL VIEW","Interactive 3D MOLA elevation relief · drag to rotate · scroll to zoom · orbit paths are schematic")
        figure=globe_figure(st.session_state.start_site,st.session_state.dest_site,True)
        if figure is not None:
            st.plotly_chart(figure,use_container_width=True,config={"displayModeBar":True,"scrollZoom":True})
        else:
            st.error("Global MOLA grid is not available in the deployed bundle.")
        if not (PROC/"mars_viking_texture_360x180.png").exists():
            st.caption("Globe geometry and relief use MOLA MEGDR. The reduced Viking MDIM browser texture has not yet been added; the original 761 MB TIFF remains local.")
    with right:
        panel("KNOWN MARS LOCATIONS","Select a landing site, major landform or region of interest.")
        names=[p["name"] for p in MARS_PLACES]
        chosen=st.selectbox("Search / select location",names,index=names.index(st.session_state.selected_place),key="command_place")
        st.session_state.selected_place=chosen
        p=PLACE_BY_NAME[chosen]
        st.markdown(f"<div class='mm-panel'><b>{p['kind'].upper()}</b><br>{p['mission']}<br>Latitude: {p['lat']:.3f}°<br>East longitude: {p['lon']:.3f}°</div>",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        if c1.button("SET START",use_container_width=True):
            st.session_state.start_site=chosen
            st.rerun()
        if c2.button("SET DESTINATION",use_container_width=True):
            st.session_state.dest_site=chosen
            st.rerun()
        st.markdown("**ROUTE ENDPOINTS**")
        start=st.selectbox("Start location",names,index=names.index(st.session_state.start_site),key="command_start")
        dest=st.selectbox("Destination",names,index=names.index(st.session_state.dest_site),key="command_dest")
        st.session_state.start_site=start
        st.session_state.dest_site=dest
        st.metric("GREAT-CIRCLE DISTANCE",f"{great_circle_km((PLACE_BY_NAME[start]['lat'],PLACE_BY_NAME[start]['lon']),(PLACE_BY_NAME[dest]['lat'],PLACE_BY_NAME[dest]['lon'])):,.1f} km")
    st.info("GLOBAL FOUNDATION: Viking Orbiter Global Mosaic (MDIM) is the agreed imagery source; MOLA MEGDR provides elevation/topography. This deployed globe currently renders MOLA elevation relief; Viking texture is a pending web asset.")

# ============================================================
# 02 SURVIVAL GUIDE
# ============================================================
elif page=="Survival Guide":
    page_header("SURVIVAL INTELLIGENCE","Evidence → assessment → crew action → mission preparation")
    mola_ok=(PROC/"mars_global_elevation_4ppd.npy").exists()
    slope_ok=(PROC/"mars_global_slope_4ppd.npy").exists()
    jezero_preview=(PROC/"jezero_terrain_cost_preview.png").exists()
    meda_ok=all(path.exists() for path in MEDA_FILES.values())
    coverage=sum([mola_ok,slope_ok,jezero_preview,meda_ok,False,False])
    st.metric("SCIENTIFIC DATA COVERAGE",f"{coverage}/6 domains","Coverage indicator only · not a safety score")
    st.progress(coverage/6)
    st.warning("This is a data-coverage indicator, NOT a human-safety score, habitability rating or GO/NO-GO decision.")
    cols=st.columns(2)
    evidence=[
      ("GLOBAL SURFACE IMAGERY","WEB TEXTURE PENDING","Viking MDIM is the agreed global image foundation. The original TIFF is not in the deployed bundle.","Imagery does not directly measure weather, pressure, radiation, water or minerals."),
      ("GLOBAL TOPOGRAPHY","AVAILABLE" if mola_ok else "MISSING","MOLA MEGDR-derived elevation grid is included." if mola_ok else "Global elevation grid was not found.","Elevation supports terrain context, not direct resource or safety conclusions."),
      ("JEZERO REGIONAL TERRAIN","PREVIEW AVAILABLE" if jezero_preview else "NOT AVAILABLE","Regional slope, roughness and route preview images are included.","Preview images are regional and do not replace high-resolution numeric arrays."),
      ("ATMOSPHERE / WEATHER","NOT CALIBRATED","No calibrated pressure, temperature, wind or weather product is connected.","MEDA ATS channels here are raw engineering counts, not calibrated physical values."),
      ("RADIATION ENVIRONMENT","NOT AVAILABLE","No local radiation dataset or validated radiation model is connected.","Do not infer radiation dose from imagery or elevation."),
      ("WATER / SUBSURFACE ICE","NOT MEASURED","No dedicated neutron, radar or spectral ice-evidence layer is connected.","Low elevation or image colour is not proof of water or ice.")]
    for i,(title,status,body,limit) in enumerate(evidence):
        with cols[i%2]:
            with st.expander(f"{title} · {status}",expanded=True):
                st.write(body); st.caption("LIMITATION: "+limit)
    panel("MISSION DECISION QUEUE","Preparation tasks implied by the current evidence gaps")
    st.markdown("""
    - **Environmental envelope:** obtain site- and season-specific pressure, temperature, wind, dust and atmospheric-composition evidence.
    - **Radiation baseline:** add a cited radiation dataset or validated model with units, time coverage and uncertainty.
    - **Resource verification:** use suitable radar, neutron or spectral evidence before making water, ice or mineral claims.
    - **Terrain validation:** use local high-resolution elevation, slope and roughness for a specific traverse.
    """)
    selected=PLACE_BY_NAME[st.session_state.selected_place]
    st.info(f"SELECTED LOCATION: {selected['name']} · {selected['lat']:.3f}°, {selected['lon']:.3f}°E. Coordinates are geographic references only.")
    st.error("SIMULATION BOUNDARY: Mission Twin, EVA, habitat and rover values are demonstrations. No physical hardware or sensors are connected.")

# ============================================================
# 03 MISSION TWIN
# ============================================================
elif page=="Mission Twin":
    page_header("MISSION TWIN | MISSION CONFIGURATION","Editable resource-budget demonstration · not a certified mission-planning model")
    st.warning("SIMULATION ONLY · Default consumption rates are placeholders for demonstrating reserve arithmetic.")
    left,right=st.columns([1,1],gap="large")
    with left:
        panel("MISSION CONFIGURATION","Set crew, duration and available reserves.")
        crew=st.number_input("Crew size",1,20,4,key="mission_crew")
        sols=st.number_input("Mission duration (sols)",1,2000,180,key="mission_sols")
        supplies={}
        for name,default,unit in [("Oxygen",750,"kg"),("Water",4500,"L"),("Food",1400,"kg"),("Energy",3600,"kWh")]:
            supplies[name]=st.number_input(f"{name} reserve ({unit})",0.0,1000000.0,float(default),step=10.0,key="reserve_"+name)
        panel("EDITABLE DEMO CONSUMPTION RATES","Replace with mission-approved engineering values before any real use.")
        rates={}
        for name,default,unit in [("Oxygen",.84,"kg/person-sol"),("Water",5.0,"L/person-sol"),("Food",1.5,"kg/person-sol"),("Energy",4.0,"kWh/person-sol")]:
            rates[name]=st.number_input(f"{name} rate ({unit})",0.0,1000.0,float(default),step=.1,key="rate_"+name)
        run=st.button("RUN MISSION SIMULATION",type="primary",use_container_width=True)
    if run:
        required={k:float(crew)*float(sols)*rates[k] for k in supplies}
        remaining={k:supplies[k]-required[k] for k in supplies}
        coverage={k:(supplies[k]/required[k]*100 if required[k]>0 else 9999) for k in supplies}
        st.session_state.mission_report={"crew":crew,"sols":sols,"supplies":supplies,"rates":rates,
            "required":required,"remaining":remaining,"coverage":coverage,
            "adequate":{k:remaining[k]>=0 for k in supplies}}
    with right:
        panel("MISSION TWIN RESULTS","Projected reserves under configured demonstration assumptions.")
        report=st.session_state.mission_report
        if report:
            if all(report["adequate"].values()): st.success("RESERVES COVER THIS DEMO SCENARIO")
            else: st.error("RESOURCE SHORTAGE — REDESIGN REQUIRED")
            cols=st.columns(2)
            for i,(key,unit) in enumerate([("Oxygen","kg"),("Water","L"),("Food","kg"),("Energy","kWh")]):
                with cols[i%2]:
                    value=report["remaining"][key]
                    st.metric(key+" remaining",f"{value:,.1f} {unit}",f"Required {report['required'][key]:,.1f} {unit}")
                    st.caption(f"Reserve coverage: {report['coverage'][key]:.1f}%")
            st.write(f"Scenario: {report['crew']} crew · {report['sols']} sols")
            st.caption("This does not establish real mission feasibility or crew safety.")
        else:
            st.info("Configure the mission and run the simulation to calculate resource requirements.")

# ============================================================
# 04 HABITAT SYSTEMS
# ============================================================
elif page=="Habitat Systems":
    page_header("HABITAT SYSTEMS | LIFE-SUPPORT RESOURCE BOARD","Conceptual habitat subsystems linked to the Mission Twin demonstration")
    st.warning("SIMULATED SYSTEM STATUS · No pressure, scrubber, thermal or power hardware is connected.")
    report=st.session_state.mission_report
    if not report:
        st.info("Run a scenario in Mission Twin to populate the shared reserve board.")
    else:
        cols=st.columns(4); units={"Oxygen":"kg","Water":"L","Food":"kg","Energy":"kWh"}
        for i,key in enumerate(units):
            remain=max(0.0,report["remaining"][key]); reserve=report["supplies"][key]
            percent=float(np.clip(remain/reserve*100 if reserve>0 else 0,0,100))
            with cols[i]:
                st.metric(key+" remaining",f"{remain:,.1f} {units[key]}")
                st.progress(percent/100)
                st.caption("Budget covers scenario" if report["adequate"][key] else "SHORTAGE in configured scenario")
    panel("HABITAT SUBSYSTEM OVERVIEW","Conceptual status panels; not connected to sensors or hardware.")
    cols=st.columns(2)
    systems=[("PRESSURE CONTROL","Simulation placeholder — no pressure sensor connected."),
      ("CO₂ REMOVAL","Simulation placeholder — dedicated scrubber model required."),
      ("THERMAL CONTROL","Simulation placeholder — calibrated temperature product not loaded."),
      ("POWER DISTRIBUTION","Budget represented by Mission Twin energy reserve.")]
    for i,(name,detail) in enumerate(systems):
        with cols[i%2]:
            with st.container(border=True):
                st.subheader(name); st.write(detail); st.caption("NOT TELEMETRY-CONNECTED")

# ============================================================
# 05 ASTRONAUT EVA
# ============================================================
elif page=="Astronaut EVA":
    page_header("ASTRONAUT EVA | SPACESUIT DIGITAL TWIN","Synthetic EVA training panel · no real astronaut, suit or sensor connection")
    st.warning("SIMULATED TELEMETRY ONLY · Controls change a demonstration state; they do not control a physical spacesuit.")
    eva=st.session_state.eva
    st.markdown(f"**SUIT STATUS:** {eva['status']}  ·  **EVA ELAPSED:** {eva['elapsed']:02d} simulated minutes")
    c=st.columns(4)
    c[0].metric("OXYGEN",f"{eva['oxygen']:.1f}%"); c[1].metric("BATTERY",f"{eva['battery']:.1f}%")
    c[2].metric("CO₂",f"{eva['co2']:.0f} ppm"); c[3].metric("SUIT TEMP",f"{eva['temperature']:.1f} °C")
    c=st.columns(3); c[0].metric("HUMIDITY",f"{eva['humidity']:.1f}%"); c[1].metric("HEART RATE",f"{eva['heart_rate']} bpm"); c[2].metric("ELAPSED",f"{eva['elapsed']} min")
    a,b,d=st.columns(3)
    if a.button("START EVA",use_container_width=True): eva["status"]="RUNNING"
    if b.button("ADVANCE 1 SIM MIN",use_container_width=True) and eva["status"]=="RUNNING":
        eva["elapsed"]+=1; eva["oxygen"]=max(0,eva["oxygen"]-.35); eva["battery"]=max(0,eva["battery"]-.28)
        eva["co2"]+=4.5; eva["temperature"]+=.03; eva["humidity"]+=.08; eva["heart_rate"]=min(155,eva["heart_rate"]+1)
    if d.button("END / RESET EVA",use_container_width=True):
        st.session_state.eva={"status":"STANDBY","oxygen":100.0,"battery":100.0,"co2":450.0,"temperature":22.0,"humidity":45.0,"heart_rate":78,"elapsed":0}
        st.rerun()
    st.progress(eva["oxygen"]/100,text="SIMULATED OXYGEN RESERVE")
    st.progress(eva["battery"]/100,text="SIMULATED BATTERY")
    st.info("One click on ADVANCE 1 SIM MIN advances one synthetic minute. Values are illustrative, not validated suit dynamics.")

# ============================================================
# 06 ROVER OPERATIONS
# ============================================================
elif page=="Rover Operations":
    page_header("ROVER OPERATIONS | EXPLORATION DIGITAL TWIN","Jezero regional route candidates · simulated traversal only")
    st.warning("SIMULATED ROVER TELEMETRY · No live rover, sensor or spacecraft connection.")
    route_choice=st.selectbox("MISSION ROUTE",["ALPHA · Shortest-distance candidate","BETA · Balanced terrain candidate","GAMMA · Terrain-cost-avoiding candidate"])
    filenames={"ALPHA · Shortest-distance candidate":"jezero_shortest_route.csv",
      "BETA · Balanced terrain candidate":"jezero_balanced_route.csv",
      "GAMMA · Terrain-cost-avoiding candidate":"jezero_terrain_avoiding_route.csv"}
    rows=read_route_csv(filenames[route_choice])
    rover=st.session_state.rover
    c=st.columns(6)
    c[0].metric("ROVER SPEED","0.29 km/h","Nominal demo setting")
    c[1].metric("BATTERY",f"{rover['battery']:.1f}%","Simulated")
    c[2].metric("PITCH","0.0°","Not sensor measured")
    c[3].metric("ROLL","0.0°","Not sensor measured")
    c[4].metric("HEADING","090°","Illustrative")
    c[5].metric("ROUTE POINTS",f"{len(rows):,}","CSV candidate")
    if rows:
        st.success(f"Loaded {route_choice}: {len(rows):,} route points.")
        fig=route_figure(rows,route_choice)
        if fig is not None: st.plotly_chart(fig,use_container_width=True)
        st.progress(rover["progress"]/100,text=f"SIMULATED MISSION PROGRESS · {rover['progress']}%")
        st.write("ROVER STATUS:",rover["status"])
        a,b,c=st.columns(3)
        if a.button("START / ADVANCE ROVER",use_container_width=True):
            rover["status"]="RUNNING"; rover["progress"]=min(100,rover["progress"]+5); rover["battery"]=max(0,rover["battery"]-0.6); st.rerun()
        if b.button("PAUSE ROVER",use_container_width=True): rover["status"]="PAUSED"; st.rerun()
        if c.button("RESET MISSION",use_container_width=True):
            st.session_state.rover={"progress":0,"battery":100.0,"status":"STANDBY"}; st.rerun()
        if rover["progress"]>=100: st.success("Demo route traversal complete.")
    else:
        st.error("The selected route CSV is not present in the deployed data bundle.")
    st.caption("Route candidates are precomputed CSV paths. Progress and battery are synthetic UI demonstrations; the full regional DEM, slope, roughness and terrain-cost arrays are not included in this web bundle.")
    image=PROC/"jezero_route_comparison.png"
    if image.exists(): st.image(str(image),caption="Precomputed Jezero route comparison")

# ============================================================
# 07 MARS ENVIRONMENT / GLOBAL SCIENTIFIC LAYERS
# ============================================================
elif page=="Mars Environment":
    page_header("MARS ENVIRONMENT | GLOBAL SCIENTIFIC LAYERS","Planet-wide mapped context · named locations · source-aware layer selection")
    choice=st.selectbox("MAP LAYER",["Global elevation","Global slope","MOLA global image preview","Viking MDIM imagery"])
    if choice=="Global elevation":
        fig=global_layer_figure(PROC/"mars_global_elevation_4ppd.npy","MOLA MEGDR-derived elevation","Earth","Elevation (m)")
        if fig is not None: st.plotly_chart(fig,use_container_width=True); st.caption("MOLA MEGDR-derived elevation · 4 pixels per degree.")
        else: st.warning("Global elevation grid is unavailable.")
    elif choice=="Global slope":
        fig=global_layer_figure(PROC/"mars_global_slope_4ppd.npy","MOLA-derived global slope","Inferno","Slope (degrees)")
        if fig is not None: st.plotly_chart(fig,use_container_width=True); st.caption("Slope is derived from MOLA elevation; coarse global context only.")
        else: st.warning("Global slope grid is unavailable.")
    elif choice=="MOLA global image preview":
        img=GLOBAL/"mars_mola_global.jpg"
        if img.exists(): st.image(str(img),caption="MOLA global elevation visualization — not Viking imagery",use_container_width=True)
        else: st.warning("MOLA global image preview is unavailable.")
    else:
        st.warning("The original Viking MDIM GeoTIFF is 761 MB and was intentionally excluded from this GitHub web deployment. The original source remains in the desktop project folder. A reduced web texture must be generated and added to the repository to enable this layer.")
    st.subheader("NAMED MARS LOCATIONS")
    st.dataframe([{"Location":p["name"],"Mission / feature":p["mission"],"Latitude":p["lat"],"East longitude":p["lon"],"Type":p["kind"]} for p in MARS_PLACES],use_container_width=True,hide_index=True)

# ============================================================
# 08 TERRAIN INTELLIGENCE — JEZERO REGIONAL ANALYSIS
# ============================================================
elif page=="Terrain Intelligence":
    page_header("TERRAIN INTELLIGENCE | JEZERO REGIONAL ANALYSIS","Regional terrain evidence and precomputed A* candidate routes")
    st.warning("Jezero relative terrain cost is experimental. It is not a validated rover mobility score or certified safe route.")
    layer=st.selectbox("DISPLAY LAYER",["Route comparison","Elevation preview","Slope preview","Roughness preview","Relative terrain-cost preview"])
    previews={"Route comparison":"jezero_route_comparison.png","Elevation preview":"jezero_elevation_preview.png",
      "Slope preview":"jezero_slope_preview.png","Roughness preview":"jezero_roughness_preview.png",
      "Relative terrain-cost preview":"jezero_terrain_cost_preview.png"}
    image=PROC/previews[layer]
    if image.exists(): st.image(str(image),caption=layer+" · regional Jezero project preview",use_container_width=True)
    else: st.warning("This preview is not included in the deployed repository.")
    st.subheader("PRECOMPUTED ROUTE CANDIDATES")
    route_name=st.selectbox("Candidate route",["Shortest-distance","Balanced terrain","Terrain-cost-avoiding"])
    csv_name={"Shortest-distance":"jezero_shortest_route.csv","Balanced terrain":"jezero_balanced_route.csv",
      "Terrain-cost-avoiding":"jezero_terrain_avoiding_route.csv"}[route_name]
    rows=read_route_csv(csv_name)
    fig=route_figure(rows,route_name+" route candidate")
    if fig is not None:
        st.plotly_chart(fig,use_container_width=True)
        st.metric("ROUTE WAYPOINTS",f"{len(rows):,}","Precomputed CSV points")
        with st.expander("View route waypoint preview"):
            st.dataframe(rows[:80],use_container_width=True,hide_index=True)
    else:
        st.warning("Selected route candidate CSV is not available.")
    st.info("Interactive A* recalculation requires the full Jezero terrain-cost raster (about 86 MB), intentionally excluded from this public repository to keep deployment small. These paths are precomputed project outputs.")

# ============================================================
# 09 MARS NAV
# ============================================================
elif page=="Mars Nav":
    page_header("MARS NAV | DISTANCE & ROUTE PLANNING","Great-circle distance between named Mars sites · geometric estimate only")
    names=[p["name"] for p in MARS_PLACES]
    left,right=st.columns(2)
    with left:
        panel("ROUTE ENDPOINTS","Choose known Mars sites.")
        start=st.selectbox("START LOCATION",names,index=names.index(st.session_state.start_site),key="nav_start")
        dest=st.selectbox("DESTINATION",names,index=names.index(st.session_state.dest_site),key="nav_dest")
        st.session_state.start_site=start; st.session_state.dest_site=dest
        st.write("START:",start); st.write("DESTINATION:",dest)
        if st.button("CALCULATE GREAT-CIRCLE DISTANCE",type="primary"):
            st.session_state.nav_calculated=True
        if st.button("CLEAR ROUTE"):
            st.session_state.start_site=names[0]; st.session_state.dest_site=names[1]
            st.session_state.nav_calculated=False; st.rerun()
    with right:
        panel("DISTANCE INTELLIGENCE","Spherical great-circle / haversine method.")
        a=PLACE_BY_NAME[start]; b=PLACE_BY_NAME[dest]
        km=great_circle_km((a["lat"],a["lon"]),(b["lat"],b["lon"]))
        st.metric("GREAT-CIRCLE DISTANCE",f"{km:,.2f} km")
        st.metric("CENTRAL ANGLE",f"{math.degrees(km/MARS_RADIUS_KM):.3f}°")
        st.caption("This is not a terrain-aware path, rover travel-time prediction or navigation command.")
    fig=globe_figure(start,dest,False)
    if fig is not None: st.plotly_chart(fig,use_container_width=True)

# ============================================================
# 10 ORBITAL ASSETS
# ============================================================
elif page=="Orbital Assets":
    page_header("ORBITAL ASSETS & RELAY NETWORK","Six named spacecraft and two natural satellites · schematic orbital geometry")
    st.warning("ORBIT PATHS AND MOVING MARKERS ARE SCHEMATIC, NOT LIVE EPHEMERIS OR SPACECRAFT NAVIGATION DATA.")
    cols=st.columns(3)
    for i,asset in enumerate(ORBITAL_ASSETS):
        with cols[i%3]:
            with st.container(border=True):
                st.subheader(asset["short"]); st.caption(asset["name"])
                st.write(asset["operator"]); st.write(asset["role"]); st.markdown("**ACTIVE · STATUS LABEL FROM ORIGINAL PROJECT**")
    st.subheader("NATURAL SATELLITES")
    a,b=st.columns(2)
    with a: st.metric("PHOBOS","INNER MOON"); st.write("Schematic orbit shown around Mars.")
    with b: st.metric("DEIMOS","OUTER MOON"); st.write("Schematic orbit shown around Mars.")
    fig=globe_figure(None,None,True)
    if fig is not None: st.plotly_chart(fig,use_container_width=True)
    st.caption("Illustrative radii, inclinations, phases and speeds. Use NAIF SPICE kernels or verified ephemeris for precise positions.")

# ============================================================
# 11 RESOURCE INTELLIGENCE
# ============================================================
elif page=="Resource Intelligence":
    page_header("RESOURCE INTELLIGENCE","Evidence-first resource screening · mapped terrain is not a resource measurement")
    elev=array_stats(PROC/"mars_global_elevation_4ppd.npy")
    slope=array_stats(PROC/"mars_global_slope_4ppd.npy")
    meda_paths=[p for p in MEDA.glob("*.csv")]
    c=st.columns(3)
    c[0].metric("GLOBAL ELEVATION","AVAILABLE" if elev else "MISSING","MOLA MEGDR-derived")
    c[1].metric("GLOBAL SLOPE","AVAILABLE" if slope else "MISSING","Derived from MOLA")
    c[2].metric("MEDA ARCHIVE FILES",str(len(meda_paths)),"Raw ATS CSV files")
    items=[
      ("GLOBAL TERRAIN EVIDENCE","AVAILABLE" if elev and slope else "PARTIAL",
       f"Sampled elevation mean: {elev['mean']:.1f} m; sampled mean slope: {slope['mean']:.3f}°." if elev and slope else "One or more global terrain grids are missing.",
       "Terrain describes topography; it does not prove that a resource exists."),
      ("WATER / SUBSURFACE ICE","NOT MEASURED","No dedicated neutron, radar or spectral ice-evidence layer is connected.",
       "Do not interpret low elevation, image colour or terrain shape as proof of water or ice."),
      ("MINERAL COMPOSITION","NOT LOADED","No calibrated compositional spectroscopy dataset is connected.",
       "Mineral identification requires appropriate spectral observations and provenance."),
      ("REGIONAL SITE SCREENING","PREVIEW ONLY","Jezero route and terrain preview products are included.",
       "Relative terrain cost is exploratory, not a resource score or safety certification."),
      ("ARCHIVED MEDA OBSERVATIONS","RAW DATA PRESENT",f"{len(meda_paths)} CSV file(s) are bundled.",
       "ATS channels in these files are raw engineering counts, not calibrated weather or temperature values."),
      ("EVIDENCE CONFIDENCE","SOURCE-AWARE","Imagery, elevation, derived terrain and archived instrument records are distinguished.",
       "Unknown or unmeasured values remain unknown; no resource abundance is inferred.")]
    cols=st.columns(2)
    for i,(title,status,body,limit) in enumerate(items):
        with cols[i%2]:
            with st.container(border=True):
                st.subheader(title); st.markdown(f"**{status}**"); st.write(body); st.caption("LIMIT: "+limit)

# ============================================================
# 12 UNKNOWN FRONTIER
# ============================================================
elif page=="Unknown Frontier":
    page_header("UNKNOWN FRONTIER | SCIENTIFIC DATA GAPS","Unavailable measurements are identified explicitly rather than estimated from unrelated layers")
    meda_count=0
    for path in MEDA.glob("*.csv"):
        try:
            with path.open("r",encoding="utf-8-sig",errors="replace") as f:
                meda_count+=max(0,sum(1 for _ in f)-1)
        except OSError:
            pass
    gaps=[
      ("RADIATION DOSE","NOT AVAILABLE","No local radiation instrument dataset or validated radiation model is connected."),
      ("ATMOSPHERIC PRESSURE","NOT CALIBRATED","MEDA ATS raw engineering-count files are present, but calibrated pressure is not connected."),
      ("TEMPERATURE & WEATHER","NOT CALIBRATED","Archived MEDA ATS records are available, but calibrated temperature and weather products are not connected."),
      ("WATER / SUBSURFACE ICE","NOT AVAILABLE","Viking imagery and MOLA elevation do not measure subsurface ice abundance."),
      ("MINERAL COMPOSITION","NOT AVAILABLE","No compositional spectroscopy product is currently loaded."),
      ("HUMAN HABITABILITY","NOT CERTIFIED","This research prototype is not a validated human-safety assessment."),
      ("ARCHIVED MEDA ATS","RAW RECORDS FOUND" if meda_count else "NOT FOUND",f"{meda_count:,} bundled CSV rows. Raw counts only; calibration is required for physical interpretation.")]
    cols=st.columns(3)
    for i,(name,status,detail) in enumerate(gaps):
        with cols[i%3]:
            with st.container(border=True):
                st.subheader(name); st.markdown(f"**{status}**"); st.write(detail)

# ============================================================
# 13 MARTIAN SCIENCE LAB
# ============================================================
elif page=="Martian Science Lab":
    page_header("MARTIAN SCIENCE LAB | LOCAL DATA INVENTORY","Sampled numerical summaries from the scientific files included in this deployment")
    targets=[
      ("MOLA global elevation",PROC/"mars_global_elevation_4ppd.npy","m"),
      ("MOLA globe grid",PROC/"mars_mola_globe_360x180.npy","m"),
      ("Derived global slope",PROC/"mars_global_slope_4ppd.npy","degrees")]
    for title,path,unit in targets:
        summary=array_stats(path)
        with st.container(border=True):
            st.subheader(title)
            if summary:
                c=st.columns(4); c[0].metric("GRID",f"{summary['shape'][0]} × {summary['shape'][1]}")
                c[1].metric("SAMPLED MIN",f"{summary['min']:.2f} {unit}")
                c[2].metric("SAMPLED MEAN",f"{summary['mean']:.2f} {unit}")
                c[3].metric("SAMPLED MAX",f"{summary['max']:.2f} {unit}")
            else:
                st.warning("Dataset not found in this deployment.")
    st.subheader("PREVIEW PRODUCTS")
    previews=["mars_global_elevation.png","jezero_elevation_preview.png","jezero_slope_preview.png",
              "jezero_roughness_preview.png","jezero_terrain_cost_preview.png","jezero_route_comparison.png"]
    for i in range(0,len(previews),2):
        cols=st.columns(2)
        for j,name in enumerate(previews[i:i+2]):
            p=PROC/name
            with cols[j]:
                if p.exists(): st.image(str(p),caption=name,use_container_width=True)
                else: st.caption(name+" — not included")
    st.info("Provenance: MOLA MEGDR = global elevation; global slope = derived; Jezero previews = regional project products; MEDA ATS = archived raw instrument counts. The full Jezero arrays and original Viking GeoTIFF were excluded from the lightweight web bundle.")

# ============================================================
# 14 MEDA OBSERVATIONS
# ============================================================
elif page=="MEDA Observations":
    page_header("MEDA | OBSERVATION LAB","Archived Mars Environmental Dynamics Analyzer ATS records from Perseverance")
    st.info("These are archived observations, not a live or planet-wide weather feed. ATS_ATS1–ATS_ATS5 are raw engineering counts, NOT degrees Celsius.")
    available={name:path for name,path in MEDA_FILES.items() if path.exists()}
    if not available:
        st.error("No bundled MEDA ATS CSV data was found.")
    else:
        dataset=st.selectbox("DATASET",list(available.keys()))
        channel=st.selectbox("ATS CHANNEL",MEDA_CHANNELS)
        data=load_meda_csv(str(available[dataset]))
        if data:
            st.success(f"{dataset} · {data['count']:,} archived records loaded")
            st.metric("RECORD COUNT",f"{data['count']:,}","PDS CSV rows")
            fig=go.Figure(go.Scattergl(x=data["times"],y=data["channels"][channel],mode="lines",
                line=dict(color="#66d9d0",width=1),name=channel))
            fig.update_layout(title=f"{dataset} · {channel} raw engineering counts",height=400,
                paper_bgcolor="#0d1928",plot_bgcolor="#0d1928",font=dict(color="#dce7f2"),
                xaxis_title="Local True Solar Time (decimal hours)",yaxis_title="Raw engineering counts")
            st.plotly_chart(fig,use_container_width=True)
            st.subheader("OBSERVATION RECORD PREVIEW")
            st.dataframe(data["preview"],use_container_width=True,hide_index=True)
            st.caption("Source: bundled PDS MEDA ESD_ATS CSV files. Physical interpretation requires the corresponding calibration products and instrument documentation.")
        else:
            st.error("Could not parse this archived CSV file.")
    st.markdown("[Open MEDA PDS archive](https://atmos.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_1740_1859/) · [Instrument details](https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/PERSEVERANCE/meda.html)")

footer()
