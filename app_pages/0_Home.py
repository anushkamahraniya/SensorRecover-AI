"""Home: what the project is and how it works."""

import streamlit as st

from utils.evaluation import HOPFIELD_WINDOW, INTERPOLATION, KNN_WINDOW
from utils.state import active_dataset, load_result
from utils.ui import card, chips, flow, hero

dataset = active_dataset()
summary = load_result("benchmark_summary.csv")

hero(
    "Associative memory for environmental sensing",
    "Missing Sensor Pattern Reconstruction Using Hopfield Networks",
    "SensorRecover AI learns how air-quality sensors behave together, then uses a "
    "Hopfield associative memory to restore readings that are missing or corrupted.",
)

rows = len(dataset.clean)
memory_rows = int((~dataset.split.train.isna().any(axis=1)).sum())
cols = st.columns(4)
cols[0].metric("Hourly observations", f"{rows:,}")
cols[1].metric("Sensor channels used", dataset.clean.shape[1])
cols[2].metric("Real patterns available", f"{memory_rows:,}")
if summary is not None:
    outage = summary[(summary["pattern"] == "Sensor outages") & (summary["rate"] == 0.3)]
    value = outage.loc[outage["method"] == HOPFIELD_WINDOW, "RMSE"].iloc[0]
    cols[3].metric("Outage RMSE at 30% missing", f"{value:.3f}", help="Standard deviations, on the held-out test period.")
else:
    cols[3].metric("Missing-data levels tested", "5")

st.write("")
left, right = st.columns(2, gap="medium")
with left, card(stretch=True):
    st.subheader("The problem")
    st.write(
        "Environmental sensors fail all the time: power cuts, calibration, drift and "
        "network loss all leave holes in the record. In the UCI dataset used here, "
        "nearly one reading in five from the reference analysers is absent. Holes "
        "break forecasting, alerts and reports, and a single sensor's history often "
        "cannot fill them, because a long outage leaves nothing nearby in time."
    )
    st.write(
        "**Problem statement:** build a Hopfield-based associative memory that "
        "reconstructs missing or corrupted patterns in multivariate sensor data."
    )
with right, card(stretch=True):
    st.subheader("What is a Hopfield network?")
    st.write(
        "A Hopfield network is a memory that is searched by content, not by address. "
        "Hearing the first notes of a song brings back the rest; in the same way, the "
        "network takes a partial pattern and settles into the closest stored one."
    )
    st.write(
        "Each stored pattern is a valley in an energy landscape. A damaged pattern "
        "starts on a slope, and each update moves it downhill until it comes to rest. "
        "Known readings stay fixed, and only the unknown ones move."
    )

st.subheader("How the system works")
flow(
    [
        ("Clean", "Decode the -200 gap code, and drop channels that are mostly empty."),
        ("Store", "Keep each complete, real training pattern as one memory."),
        ("Damage", "Hide or corrupt test readings whose true values are known."),
        ("Retrieve", "Clamp known sensors and update the rest as energy falls."),
        ("Evaluate", "Score hidden readings only, against the baselines."),
    ]
)

st.write("")
with card():
    st.subheader("Architecture")
    st.mermaid_chart(
        """
        flowchart LR
          A[Real sensor data] --> B[Clean and scale] --> C[(Hopfield memory)]
          D[Damaged reading] --> E[Clamp known neurons] --> F[Energy-descent retrieval]
          C --> F --> G[Reconstructed reading]
          classDef soft fill:#EAF4FB,stroke:#B8D8F0,color:#344B64
          class A,B,C,D,E,F,G soft
        """
    )

with card():
    st.subheader("Why a continuous network")
    st.write(
        "A classical Hopfield network stores only +1 and -1, and only about 0.14 "
        "patterns per neuron. A modern continuous network stores real numbers and "
        "thousands of patterns. This project builds both and tests both."
    )
    chips(["Continuous states", "Energy descent", "Clamped retrieval", "Binary comparison"])

if summary is not None:
    gaps = summary[(summary["pattern"] == "Random gaps") & (summary["rate"] == 0.3)].set_index("method")["RMSE"]
    outages = summary[(summary["pattern"] == "Sensor outages") & (summary["rate"] == 0.3)].set_index("method")["RMSE"]
    with card():
        st.subheader("What the test period showed")
        st.write(
            f"At 30% missing, RMSE in standard deviations: on **random gaps**, Hopfield "
            f"scored {gaps[HOPFIELD_WINDOW]:.3f}, windowed KNN {gaps[KNN_WINDOW]:.3f} and "
            f"interpolation {gaps[INTERPOLATION]:.3f}. On **sensor outages**, Hopfield "
            f"scored {outages[HOPFIELD_WINDOW]:.3f}, windowed KNN {outages[KNN_WINDOW]:.3f} "
            f"and interpolation {outages[INTERPOLATION]:.3f}."
        )
        st.page_link("app_pages/4_Model_Performance.py", label="See the full comparison", icon=":material/arrow_forward:")
