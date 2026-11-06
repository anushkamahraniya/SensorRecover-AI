"""Hopfield reconstruction: run the network on the current scenario."""

import numpy as np
import streamlit as st

from utils import visualization as viz
from utils.evaluation import ALL_METHODS, HOPFIELD_METHODS, HOPFIELD_WINDOW
from utils.state import (
    active_dataset,
    build_scenario,
    default_scenario,
    get_reconstruction,
    get_scenario,
    get_toolkit,
    run_reconstruction,
    sensor_unit,
)
from utils.ui import card, chart, page_header

dataset = active_dataset()
split = dataset.split

page_header(
    "Hopfield reconstruction",
    "Known readings are clamped, and the network updates the unknown neurons until "
    "the state settles in an energy valley.",
)

scenario = get_scenario(dataset) or default_scenario(dataset)

with card():
    st.subheader("Run")
    left, right = st.columns([3, 2], gap="medium", vertical_alignment="bottom")
    with left:
        rate = st.select_slider(
            "Missing readings", options=[0.1, 0.2, 0.3, 0.4, 0.5], value=scenario["rate"],
            format_func=lambda v: f"{v:.0%}", key="run_rate",
        )
    with right:
        run = st.button("Run reconstruction", type="primary", icon=":material/play_arrow:", width="stretch")
    st.caption(
        f"Scenario: {scenario['pattern'].lower()}, {len(scenario['sensors'])} sensors, "
        f"{scenario['noise_rate']:.0%} corrupted. Change the rest in the simulator."
    )

if run:
    if rate != scenario["rate"]:
        scenario = build_scenario(
            dataset, scenario["sensors"], rate, scenario["pattern"], scenario["noise_rate"],
            scenario["noise_kind"], scenario["noise_strength"], scenario["seed"],
        )
    st.session_state["scenario"] = scenario
    bar = st.progress(0.0, text="Loading the memory")
    try:
        with st.spinner("Preparing the models. The first run takes longer."):
            get_toolkit(dataset)
        st.session_state["reconstruction"] = run_reconstruction(
            dataset, scenario, on_progress=lambda share, text: bar.progress(min(share, 1.0), text=text)
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        st.error(f"Reconstruction could not finish. {error}")
        st.stop()
    bar.empty()

result = get_reconstruction(dataset)
if result is None:
    st.info("Select **Run reconstruction** to start.", icon=":material/info:")
    st.stop()

scenario = result["scenario"]
metrics = result["metrics"].set_index("method")
main = metrics.loc[HOPFIELD_WINDOW]
trace = result["traces"][HOPFIELD_WINDOW]

cols = st.columns(4)
cols[0].metric("RMSE", f"{main['RMSE']:.3f}", help="In standard deviations of each sensor.")
cols[1].metric("MAE", f"{main['MAE']:.3f}")
cols[2].metric("R²", f"{main['R2']:.3f}")
cols[3].metric("Readings scored", f"{int(main['cells']):,}")
st.caption(f"{HOPFIELD_WINDOW}, scored only on readings damaged on purpose.")

if result["precision"] is not None:
    with card():
        st.subheader("Corruption check")
        st.write(
            "No method is told which readings are corrupted. Each reading is compared with "
            "what the other sensors imply, and readings that disagree are treated as unknown."
        )
        a, b, c = st.columns(3)
        a.metric("Readings flagged", f"{int(result['flagged'].sum()):,}")
        b.metric("Precision", f"{result['precision']:.0%}", help="Of the flagged readings, the share truly corrupted.")
        c.metric("Recall", f"{result['recall']:.0%}", help="Of the corrupted readings, the share that was flagged.")

with card():
    st.subheader("Original and reconstructed readings")
    left, right = st.columns(2, gap="medium")
    sensor = left.selectbox("Sensor", split.columns, key="rec_sensor")
    method = right.selectbox("Method", ALL_METHODS, key="rec_method")
    j = split.columns.index(sensor)
    length = len(split.test)
    window = st.slider("Hours shown", 0, length, (0, min(336, length)), 24, key="rec_window")
    part = slice(*window)
    truth = split.from_z(scenario["z_true"])[part, j]
    estimate = split.from_z(result["completed"][method])[part, j]
    seen = split.from_z(scenario["z_damaged"])[part, j]
    marked = result["scored"][part, j]
    if marked.sum() == 0:
        st.info("No damaged readings for this sensor in this range.", icon=":material/info:")
    chart(
        viz.series_comparison(
            split.test.index[part], truth, seen, estimate, marked, sensor, sensor_unit(sensor)
        ),
        "rec_series",
    )

left, right = st.columns(2, gap="medium")
with left, card():
    st.subheader("Reconstruction progress")
    st.caption(
        f"The network ran {trace.iterations} update steps (the limit is 20). Falling energy "
        "means the state is moving into a memory valley."
    )
    chart(viz.trace_chart(trace.energy, "Mean energy"), "rec_energy")
with right, card():
    st.subheader("Size of each update")
    st.caption("Average change to the unknown neurons at each step, in standard deviations.")
    chart(viz.trace_chart(trace.change, "Mean change"), "rec_change")

with card():
    st.subheader("All methods on this scenario")
    st.dataframe(
        result["metrics"].rename(columns={"method": "Method", "R2": "R²", "cells": "Cells"}).round(3),
        hide_index=True,
    )
    st.caption(
        "This is one run on one random scenario. The performance page averages many runs. "
        f"{', '.join(HOPFIELD_METHODS)} are the Hopfield variants."
    )
