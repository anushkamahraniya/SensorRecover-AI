"""Visualization dashboard: all the views of one reconstruction in one place."""

import streamlit as st

from utils import visualization as viz
from utils.evaluation import ALL_METHODS, HOPFIELD_WINDOW, INTERPOLATION, KNN_SINGLE, MEAN
from utils.state import (
    active_dataset,
    cell_states,
    default_scenario,
    get_reconstruction,
    get_toolkit,
    run_reconstruction,
    sensor_metrics,
    sensor_unit,
)
from utils.ui import card, chart, page_header

dataset = active_dataset()
split = dataset.split

page_header(
    "Visualization dashboard",
    "One view of the latest reconstruction: what was damaged, what came back, and where the errors are.",
)

result = get_reconstruction(dataset)
if result is None:
    st.info("No reconstruction has been run yet in this session.", icon=":material/info:")
    if st.button("Run a sample: 30% random gaps", type="primary", icon=":material/play_arrow:"):
        bar = st.progress(0.0, text="Preparing the models")
        get_toolkit(dataset)
        scenario = default_scenario(dataset)
        st.session_state["scenario"] = scenario
        st.session_state["reconstruction"] = run_reconstruction(
            dataset, scenario, on_progress=lambda share, text: bar.progress(min(share, 1.0), text=text)
        )
        st.rerun()
    st.stop()

scenario = result["scenario"]
metrics = result["metrics"]
scored = result["scored"]
main = metrics.set_index("method").loc[HOPFIELD_WINDOW]

cols = st.columns(4)
cols[0].metric("Scenario", f"{scenario['rate']:.0%} missing", help=scenario["pattern"])
cols[1].metric("Hopfield RMSE", f"{main['RMSE']:.3f}")
cols[2].metric("Hopfield R²", f"{main['R2']:.3f}")
best = metrics.sort_values("RMSE").iloc[0]
cols[3].metric("Best on this run", best["method"].split(" (")[0], help=f"{best['method']}, RMSE {best['RMSE']:.3f}")

left, right = st.columns(2, gap="medium")
sensor = left.selectbox("Sensor", split.columns, key="dash_sensor")
method = right.selectbox("Method", ALL_METHODS, key="dash_method")
j = split.columns.index(sensor)
unit = sensor_unit(sensor)
truth = split.from_z(scenario["z_true"])
estimate = split.from_z(result["completed"][method])

with card():
    hours = slice(0, 336)
    chart(
        viz.series_comparison(
            split.test.index[hours], truth[hours, j], split.from_z(scenario["z_damaged"])[hours, j],
            estimate[hours, j], scored[hours, j], sensor, unit,
            f"Original and reconstructed {sensor}, first 14 days",
        ),
        "dash_series",
    )

left, right = st.columns(2, gap="medium")
with left, card():
    picked = scored[:, j]
    if picked.any():
        chart(viz.truth_vs_estimate(truth[picked, j], estimate[picked, j], sensor,
                                    f"True and reconstructed {sensor}"), "dash_scatter")
    else:
        st.info("This sensor has no damaged readings in this scenario.", icon=":material/info:")
with right, card():
    chart(viz.method_bars(metrics, "RMSE", "Model comparison on this run"), "dash_bars")

with card():
    chart(viz.cell_map(cell_states(scenario), split.test.index, split.columns,
                       "Missing data map, first 14 days"), "dash_cells")

with card():
    errors = {
        name: (result["completed"][name] - scenario["z_true"])[scored]
        for name in (HOPFIELD_WINDOW, KNN_SINGLE, INTERPOLATION, MEAN)
    }
    chart(viz.error_histogram(errors, "Reconstruction error"), "dash_errors")
with card():
    chart(viz.correlation_heatmap(dataset.clean, "Sensor correlation"), "dash_corr")

with card():
    z_table = sensor_metrics(dataset, result)
    z_table = z_table[z_table["sensor"].isin(scenario["sensors"])].copy()
    scale = dict(zip(split.columns, split.std))
    z_table["MAE"] = z_table["MAE"] / z_table["sensor"].map(scale)
    chart(viz.sensor_heatmap(z_table, "MAE", "Error by sensor and method (standard deviations)"), "dash_sensors")
