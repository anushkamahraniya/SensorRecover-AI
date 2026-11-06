"""Download results: reconstructed data, evaluation tables, charts and the report."""

import streamlit as st

from utils import visualization as viz
from utils.evaluation import ALL_METHODS, HOPFIELD_WINDOW
from utils.preprocessing import PROJECT_ROOT
from utils.state import (
    active_dataset,
    cell_states,
    get_reconstruction,
    load_result,
    reconstructed_table,
    sensor_metrics,
)
from utils.ui import card, page_header


def as_csv(table, index=False) -> bytes:
    return table.to_csv(index=index).encode("utf-8")


def as_html(fig) -> bytes:
    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


dataset = active_dataset()
split = dataset.split
result = get_reconstruction(dataset)

page_header("Download results", "Take the reconstructed data, the scores and the charts with you.")

with card():
    st.subheader("Latest reconstruction")
    if result is None:
        st.info("Run a reconstruction first. Its files will appear here.", icon=":material/info:")
        st.page_link("app_pages/3_Reconstruction.py", label="Go to reconstruction", icon=":material/arrow_forward:")
    else:
        scenario = result["scenario"]
        st.caption(
            f"{scenario['pattern']}, {scenario['rate']:.0%} missing, "
            f"{scenario['noise_rate']:.0%} corrupted, seed {scenario['seed']}."
        )
        method = st.selectbox("Method", ALL_METHODS, key="download_method")
        table = reconstructed_table(dataset, result, method)
        st.dataframe(table.head(50).round(3), height=240)
        st.caption(
            "Each sensor has a __source column that says whether a value is a real reading or "
            "an estimate. Filled natural gaps have no true value and could not be checked."
        )
        st.download_button(
            "Reconstructed dataset (CSV)", as_csv(table, index=True), "reconstructed_dataset.csv",
            "text/csv", type="primary", icon=":material/download:",
        )
        row = st.container(horizontal=True)
        row.download_button("Scores for this run (CSV)", as_csv(result["metrics"]),
                            "scenario_metrics.csv", "text/csv", icon=":material/download:")
        row.download_button("Scores per sensor, real units (CSV)", as_csv(sensor_metrics(dataset, result)),
                            "scenario_metrics_per_sensor.csv", "text/csv", icon=":material/download:")

with card():
    st.subheader("Benchmark tables")
    live = st.session_state.get("live_benchmark")
    row = st.container(horizontal=True)
    available = 0
    if live is not None and live["key"] == dataset.key:
        row.download_button("Live benchmark (CSV)", as_csv(live["summary"]), "live_benchmark.csv",
                            "text/csv", icon=":material/download:")
        available += 1
    if dataset.is_uci:
        files = {
            "benchmark_summary.csv": "Benchmark summary",
            "benchmark_runs.csv": "Every benchmark run",
            "per_sensor_30pct.csv": "Per-sensor errors at 30%",
            "corruption_runs.csv": "Corruption benchmark",
            "iteration_ablation.csv": "Iteration study",
        }
        for name, label in files.items():
            saved = load_result(name)
            if saved is not None:
                row.download_button(f"{label} (CSV)", as_csv(saved), name, "text/csv", icon=":material/download:")
                available += 1
    if not available:
        st.info("Run the benchmark on the performance page first.", icon=":material/info:")

with card():
    st.subheader("Charts")
    st.caption(
        "Charts download as interactive HTML, which needs internet access to open. "
        "For an image, hover over any chart in the app and select the camera icon."
    )
    row = st.container(horizontal=True)
    row.download_button(
        "Correlation heatmap (HTML)", as_html(viz.correlation_heatmap(dataset.clean, "Sensor correlation")),
        "correlation_heatmap.html", "text/html", icon=":material/download:",
    )
    if result is not None:
        row.download_button(
            "Model comparison (HTML)",
            as_html(viz.method_bars(result["metrics"], "RMSE", "Model comparison on this run")),
            "model_comparison.html", "text/html", icon=":material/download:",
        )
        row.download_button(
            "Missing data map (HTML)",
            as_html(viz.cell_map(cell_states(result["scenario"]), split.test.index, split.columns, "Missing data map")),
            "missing_data_map.html", "text/html", icon=":material/download:",
        )
        trace = result["traces"][HOPFIELD_WINDOW]
        row.download_button(
            "Energy curve (HTML)", as_html(viz.trace_chart(trace.energy, "Mean energy", "Retrieval energy")),
            "energy_curve.html", "text/html", icon=":material/download:",
        )

report = PROJECT_ROOT / "docs" / "PROJECT_REPORT.md"
if report.exists():
    with card():
        st.subheader("Project report")
        st.download_button("Academic report (Markdown)", report.read_bytes(), "PROJECT_REPORT.md",
                           "text/markdown", icon=":material/download:")
