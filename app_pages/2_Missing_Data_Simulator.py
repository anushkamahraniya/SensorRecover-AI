"""Missing data simulator: damage the test period in a controlled way."""

import numpy as np
import pandas as pd
import streamlit as st

from utils import visualization as viz
from utils.evaluation import PATTERNS
from utils.state import active_dataset, build_scenario, cell_states, get_scenario, sensor_unit
from utils.ui import card, chart, page_header

NOISE_KINDS = {"Gaussian noise": "gaussian", "Sudden spikes": "spike", "Stuck sensor": "stuck"}

dataset = active_dataset()
split = dataset.split

page_header(
    "Missing data simulator",
    "Hide or corrupt readings from the test period. Each damaged reading is a real "
    "value, so every reconstruction can be checked against the truth.",
)

with st.form("simulator"):
    st.subheader("Scenario")
    sensors = st.multiselect("Sensors to damage", split.columns, default=split.columns, key="sim_sensors")
    left, right = st.columns(2, gap="medium")
    with left:
        rate = st.select_slider(
            "Missing readings", options=[0.1, 0.2, 0.3, 0.4, 0.5], value=0.3,
            format_func=lambda v: f"{v:.0%}", key="sim_rate",
        )
        pattern = st.segmented_control(
            "Missing pattern", list(PATTERNS), default="Random gaps", key="sim_pattern",
            help="Random gaps are scattered single hours. Outages hide one sensor for 24 hours in a row.",
        )
        seed = st.number_input("Random seed", 0, 9999, 42, key="sim_seed", help="The same seed repeats the same damage.")
    with right:
        noise_rate = st.slider(
            "Corrupted readings", 0, 30, 0, 5, format="%d%%", key="sim_noise",
            help="Share of the remaining readings to corrupt.",
        )
        noise_label = st.selectbox("Corruption type", list(NOISE_KINDS), key="sim_kind")
        kind = NOISE_KINDS[noise_label]
        strength = st.slider("Corruption strength", 0.5, 24.0, 3.0, 0.5, key="sim_strength",
                             help="Standard deviations for noise and spikes, or lag in hours for a stuck sensor.")
    submitted = st.form_submit_button("Apply damage", type="primary", icon=":material/bolt:")

if submitted:
    if not sensors:
        st.warning("Choose at least one sensor.", icon=":material/warning:")
    elif pattern is None:
        st.warning("Choose a missing pattern.", icon=":material/warning:")
    else:
        st.session_state["scenario"] = build_scenario(
            dataset, sensors, rate, pattern, noise_rate / 100, kind, strength, int(seed)
        )
        st.session_state.pop("reconstruction", None)

scenario = get_scenario(dataset)
if scenario is None:
    st.info("Choose your settings and select **Apply damage** to create a scenario.", icon=":material/info:")
    st.stop()

hidden, corrupted, natural = scenario["hidden"], scenario["corrupted"], scenario["natural"]
cols = st.columns(4)
cols[0].metric("Test period cells", f"{hidden.size:,}")
cols[1].metric("Natural gaps", f"{int(natural.sum()):,}", help="Already empty in the real data. Never scored.")
cols[2].metric("Hidden on purpose", f"{int(hidden.sum()):,}", help="True values known. These are scored.")
cols[3].metric("Corrupted on purpose", f"{int(corrupted.sum()):,}", help="True values known. These are scored.")

with card():
    st.subheader("What was damaged")
    st.caption("First 14 days of the test period. Hover over a cell for details.")
    chart(viz.cell_map(cell_states(scenario), split.test.index, split.columns), "sim_cells")

with card():
    st.subheader("Original and damaged readings")
    sensor = st.selectbox("Sensor", scenario["sensors"], key="sim_view_sensor")
    j = split.columns.index(sensor)
    hours = slice(0, 336)
    index = split.test.index[hours]
    original = split.from_z(scenario["z_true"])[hours, j]
    damaged = split.from_z(np.nan_to_num(scenario["z_damaged"]))[hours, j]
    damaged = np.where(np.isnan(scenario["z_damaged"][hours, j]), np.nan, damaged)
    fig = viz.go.Figure()
    fig.add_scatter(x=index, y=original, name="Original", mode="lines",
                    line={"color": "#A9C6DE", "width": 2})
    fig.add_scatter(x=index, y=damaged, name="After damage", mode="lines+markers",
                    line={"color": viz.NAVY, "width": 1.4}, marker={"size": 4, "color": viz.NAVY})
    unit = sensor_unit(sensor)
    chart(viz._layout(fig, height=360, y_title=f"{sensor} ({unit})" if unit else sensor), "sim_series")
    st.caption("Breaks in the dark line are hidden or naturally missing readings.")

with st.expander("Sample of the damaged table", icon=":material/table_rows:"):
    shown = pd.DataFrame(
        split.from_z(scenario["z_damaged"]), index=split.test.index, columns=split.columns
    )
    st.dataframe(shown.head(100).round(2))

st.page_link("app_pages/3_Reconstruction.py", label="Continue to reconstruction", icon=":material/arrow_forward:")
