"""Model performance: the full benchmark on the held-out test period."""

import pandas as pd
import streamlit as st

from utils import visualization as viz
from utils.evaluation import (
    ALL_METHODS,
    HOPFIELD_BINARY,
    HOPFIELD_SINGLE,
    HOPFIELD_WINDOW,
    INTERPOLATION,
    KNN_SINGLE,
    KNN_WINDOW,
    MEAN,
    run_benchmark,
    summarise,
)
from utils.state import active_dataset, get_toolkit, load_result
from utils.ui import card, chart, page_header

DEFAULT_METHODS = [HOPFIELD_WINDOW, KNN_SINGLE, INTERPOLATION, MEAN]

dataset = active_dataset()
split = dataset.split

page_header(
    "Model performance",
    "Every method was tuned on the validation period, then scored on the test period, "
    "using hidden readings whose true values are known.",
)

live = st.session_state.get("live_benchmark")
if live is not None and live["key"] != dataset.key:
    live = None
saved = load_result("benchmark_summary.csv") if dataset.is_uci else None

with st.expander("Run the benchmark live", icon=":material/play_circle:", expanded=saved is None):
    st.write(
        "This repeats the test on the active dataset, using all five missing-data levels "
        "and both patterns. It usually takes a few minutes."
    )
    runs = st.segmented_control("Runs per setting", [1, 2, 3], default=1, key="bench_runs")
    if st.button("Run benchmark", type="primary", icon=":material/play_arrow:"):
        bar = st.progress(0.0, text="Preparing the models")
        try:
            kit, _ = get_toolkit(dataset)
            results = run_benchmark(
                kit, split.to_z(split.test.to_numpy()), seeds=tuple(range(runs or 1)),
                on_step=lambda share, text: bar.progress(share, text=text),
            )
        except ValueError as error:
            st.error(f"The benchmark could not finish. {error}")
            st.stop()
        bar.empty()
        live = {"key": dataset.key, "summary": summarise(results), "runs": results}
        st.session_state["live_benchmark"] = live

if live is not None:
    summary = live["summary"]
    st.caption("Showing the live run from this session.")
elif saved is not None:
    summary = saved
    st.caption("Showing saved results from run_experiments.py: 5 runs per setting.")
else:
    st.info("Run the benchmark above to see results for this dataset.", icon=":material/info:")
    st.stop()

left, right, third = st.columns(3, gap="medium")
pattern = left.segmented_control("Missing pattern", list(summary["pattern"].unique()),
                                 default="Random gaps", key="perf_pattern") or "Random gaps"
metric = right.segmented_control("Metric", ["RMSE", "MAE", "MSE", "R2"], default="RMSE",
                                 key="perf_metric") or "RMSE"
rate = third.select_slider("Missing readings", options=sorted(summary["rate"].unique()), value=0.3,
                           format_func=lambda v: f"{v:.0%}", key="perf_rate")

current = summary[(summary["pattern"] == pattern) & (summary["rate"] == rate)]
main = current[current["method"] == HOPFIELD_WINDOW].iloc[0]
cols = st.columns(4)
for col, name in zip(cols, ["MAE", "MSE", "RMSE", "R2"]):
    col.metric(name.replace("R2", "R²"), f"{main[name]:.3f}")
st.caption(
    f"{HOPFIELD_WINDOW} on {pattern.lower()}, {rate:.0%} missing. Errors are in standard "
    "deviations, which makes sensors with different units comparable."
)

methods = st.multiselect("Methods on the chart", ALL_METHODS, default=DEFAULT_METHODS, key="perf_methods")
left, right = st.columns([3, 2], gap="medium")
with left, card():
    data = summary[summary["pattern"] == pattern]
    chart(viz.method_lines(data, metric, methods, f"{metric} as more data goes missing"), "perf_lines")
with right, card():
    chart(
        viz.method_bars(current[current["method"].isin(methods)], metric,
                        f"{metric} at {rate:.0%} missing", ascending=metric != "R2"),
        "perf_bars",
    )

with card():
    st.subheader("Comparison table")
    table = current[["method", "MAE", "MSE", "RMSE", "RMSE_std", "R2", "runs"]].sort_values("RMSE")
    table = table.rename(columns={"method": "Method", "RMSE_std": "RMSE spread", "R2": "R²", "runs": "Runs"})
    st.dataframe(table.round(3), hide_index=True)

with card():
    st.subheader("What the results mean")
    both = summary[summary["rate"].between(0.1, 0.5)]
    wide = both.pivot_table(index="method", columns="pattern", values="RMSE", aggfunc="mean")
    gaps, outages = wide["Random gaps"], wide["Sensor outages"]
    overall = wide.mean(axis=1).sort_values()
    st.markdown(
        f"- **Random gaps:** best method is {gaps.idxmin()} (mean RMSE {gaps.min():.3f}). "
        f"{HOPFIELD_WINDOW} scores {gaps[HOPFIELD_WINDOW]:.3f}. A scattered gap usually has "
        "real readings an hour either side, which suits interpolation.\n"
        f"- **Sensor outages:** best method is {outages.idxmin()} ({outages.min():.3f}). "
        f"Interpolation falls to {outages[INTERPOLATION]:.3f}: over a 24-hour gap, a straight "
        "line misses the daily cycle, while memory methods read the other sensors.\n"
        f"- **Across both patterns:** {overall.index[0]} ({overall.iloc[0]:.3f}), then "
        f"{overall.index[1]} ({overall.iloc[1]:.3f}). {HOPFIELD_WINDOW} scores "
        f"{overall[HOPFIELD_WINDOW]:.3f}.\n"
        f"- **Same window, Hopfield against KNN:** {HOPFIELD_WINDOW} {overall[HOPFIELD_WINDOW]:.3f}, "
        f"{KNN_WINDOW} {overall[KNN_WINDOW]:.3f}. Both retrieve similar stored patterns, "
        "so close results are expected.\n"
        f"- **Continuous against binary:** {HOPFIELD_SINGLE} {overall[HOPFIELD_SINGLE]:.3f}, "
        f"{HOPFIELD_BINARY} {overall[HOPFIELD_BINARY]:.3f}, both on single hours. Rounding to bits and storing "
        f"few prototypes costs accuracy, though it still beats {MEAN.lower()} ({overall[MEAN]:.3f})."
    )

if live is None and dataset.is_uci:
    sensors = load_result("per_sensor_30pct.csv")
    ablation = load_result("iteration_ablation.csv")
    corruption = load_result("corruption_runs.csv")

    if sensors is not None:
        with card():
            st.subheader("Error for each sensor, in real units")
            st.caption("One run at 30% missing. Each sensor has its own unit, so compare down a column.")
            view = sensors[sensors["pattern"] == pattern]
            wide_view = view.pivot(index="sensor", columns="method", values="MAE")[ALL_METHODS]
            st.dataframe(wide_view.loc[split.columns].round(2))

    if ablation is not None:
        with card():
            st.subheader("Do more update steps help?")
            left, right = st.columns([3, 2], gap="medium")
            with left:
                chart(viz.ablation_chart(ablation), "perf_ablation")
            with right:
                first = ablation[ablation["steps"] == 0].set_index("pattern")["RMSE"]
                last = ablation[ablation["steps"] == ablation["steps"].max()].set_index("pattern")["RMSE"]
                st.write(
                    f"RMSE moves from {first['Random gaps']:.3f} to {last['Random gaps']:.3f} on "
                    f"random gaps, and from {first['Sensor outages']:.3f} to "
                    f"{last['Sensor outages']:.3f} on outages."
                )
                st.write(
                    "The first recall gives a weighted average of memories, which is already "
                    "close to the lowest-error estimate. Later steps pull the state toward a "
                    "single memory. They guarantee a stable, energy-minimising result, not a "
                    "lower error."
                )

    if corruption is not None:
        with card():
            st.subheader("Corrupted readings")
            st.write(
                "Here no method is told which readings are wrong. The Hopfield memory flags "
                "suspicious readings, and each method then repairs the same flagged cells."
            )
            grouped = corruption.groupby(["kind", "rate", "method"], as_index=False)[
                ["RMSE", "precision", "recall"]
            ].mean()
            options = sorted(grouped["kind"].unique())
            kind = st.segmented_control("Corruption type", options, default=options[0], key="perf_kind",
                                        format_func=str.capitalize) or options[0]
            shown = grouped[grouped["kind"] == kind]
            wide_view = shown.pivot(index="method", columns="rate", values="RMSE")
            wide_view.columns = [f"RMSE at {c:.0%} corrupted" for c in wide_view.columns]
            st.dataframe(wide_view.sort_values(wide_view.columns[0]).round(3))
            detection = shown.groupby("rate")[["precision", "recall"]].mean().reset_index()
            detection["rate"] = detection["rate"].map("{:.0%}".format)
            detection = detection.rename(columns={"rate": "Corrupted", "precision": "Precision", "recall": "Recall"})
            st.dataframe(detection.round(3), hide_index=True)
