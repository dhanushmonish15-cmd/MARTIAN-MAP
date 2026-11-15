
import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image

# PAGE CONFIGURATION
st.set_page_config(
    page_title="MARTIAN MAP | Mission Intelligence",
    page_icon="🔴",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GLOBAL_DIR = DATA_DIR / "global"
PROCESSED_DIR = DATA_DIR / "processed"

# CUSTOM DESIGN
st.markdown("""
<style>
.stApp {
    background: #080d18;
    color: #eaf2ff;
}
section[data-testid="stSidebar"] {
    background: #111b2b;
}
h1, h2, h3 {
    color: #f1f5ff;
}
div[data-testid="stMetric"] {
    background: #142238;
    padding: 18px;
    border: 1px solid #29415f;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🔴 MARTIAN MAP")
st.sidebar.caption("INTERPLANETARY SURVIVAL & EXPLORATION")

page = st.sidebar.radio(
    "MISSION MODULES",
    [
        "Command Centre",
        "Mars Globe",
        "Survival Guide",
        "Terrain Intelligence",
        "Jezero Route Planning",
        "Science & Discovery"
    ]
)

st.sidebar.divider()
st.sidebar.caption("SCIENTIFIC RESEARCH PROTOTYPE")
st.sidebar.caption("Data-driven | Simulation clearly labelled")

# COMMAND CENTRE
if page == "Command Centre":
    st.title("🔴 MISSION COMMAND CENTRE")
    st.write("Interplanetary Survival & Exploration Intelligence System")

    st.success("MARTIAN MAP WEB INTERFACE INITIALIZED")

    c1, c2, c3 = st.columns(3)
    c1.metric("PLANET", "MARS")
    c2.metric("MEAN RADIUS", "3,389.5 km")
    c3.metric("MISSION STATUS", "RESEARCH")

    st.subheader("MISSION OVERVIEW")
    st.write("""
    Explore Martian surface conditions, terrain characteristics,
    environmental observations and simulated mission scenarios
    through one scientific intelligence platform.
    """)

    st.info(
        "Global map foundation: Viking Orbiter Global Mosaic (MDIM) "
        "and MOLA MEGDR elevation data."
    )

# MARS GLOBE
elif page == "Mars Globe":
    st.title("🌎 GLOBAL MARS EXPLORER")

    image_path = GLOBAL_DIR / "mars_mola_global.jpg"

    if image_path.exists():
        st.image(
            str(image_path),
            caption="Mars global elevation visualization",
            use_container_width=True
        )
    else:
        st.warning("Mars global image is not available in the repository yet.")

    st.caption(
        "Global map foundation: Viking MDIM imagery and MOLA MEGDR topography."
    )

# SURVIVAL GUIDE
elif page == "Survival Guide":
    st.title("🧑‍🚀 ASTRONAUT SURVIVAL GUIDE")

    st.subheader("Major Martian Mission Challenges")

    c1, c2 = st.columns(2)
    with c1:
        st.warning("Thin carbon-dioxide atmosphere")
        st.warning("Extreme temperature variations")
    with c2:
        st.warning("Radiation exposure")
        st.warning("Dust storms and surface hazards")

    st.info(
        "These are general mission hazards, not live measurements "
        "or a real-time astronaut safety assessment."
    )

# TERRAIN INTELLIGENCE
elif page == "Terrain Intelligence":
    st.title("🗺️ TERRAIN INTELLIGENCE")

    elevation = PROCESSED_DIR / "mars_global_elevation_4ppd.npy"
    slope = PROCESSED_DIR / "mars_global_slope_4ppd.npy"

    for name, path in [
        ("Global Elevation", elevation),
        ("Global Slope", slope)
    ]:
        st.subheader(name)

        if path.exists():
            arr = np.load(path, mmap_mode="r")
            st.write("Grid dimensions:", arr.shape)
            st.write("Data type:", str(arr.dtype))
        else:
            st.warning(f"{name} dataset is not available.")

# JEZERO ROUTE PLANNING
elif page == "Jezero Route Planning":
    st.title("🚙 JEZERO CRATER ROUTE INTELLIGENCE")

    st.write(
        "Experimental terrain-cost-based route planning "
        "for simulated rover exploration."
    )

    route = PROCESSED_DIR / "jezero_route_comparison.png"

    if route.exists():
        st.image(str(route), use_container_width=True)
    else:
        st.warning("Route comparison preview is not available.")

    st.info(
        "Terrain cost is an experimental relative index, "
        "not a validated rover mobility or safety score."
    )

# SCIENCE AND DISCOVERY
elif page == "Science & Discovery":
    st.title("🔬 SCIENCE & DISCOVERY")

    st.subheader("Scientific Data Sources")

    st.markdown("""
    - **Viking MDIM:** Global Martian surface imagery.
    - **MOLA MEGDR:** Global elevation and topography.
    - **MEDA:** Archived environmental observations from NASA's Perseverance rover.
    - **Jezero terrain layers:** Regional slope, roughness and experimental terrain cost.
    """)

    st.success("Scientific data sources and derived layers are kept distinct.")

st.divider()
st.caption(
    "MARTIAN MAP | Interplanetary Survival & Exploration Intelligence | Research Prototype"
)
