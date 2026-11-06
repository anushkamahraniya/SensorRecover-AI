"""Dataset explorer: load, inspect and understand the data."""

import pandas as pd
import streamlit as st

from utils import visualization as viz
from utils.preprocessing import (
    SENSOR_INFO,
    UCI_DOI,
    UCI_DOWNLOAD_URL,
    UCI_SENTINEL,
    UCI_URL,
    load_uploaded_csv,
)
from utils.state import active_dataset, uci_published_table, use_uci, use_uploaded
from utils.ui import card, chart, page_header

page_header(
    "Dataset explorer",
    "Inspect the real sensor data, see where it is incomplete, or upload your own file.",
)

with card():
    st.subheader("Choose a dataset")
    source = st.segmented_control(
        "Data source",
        ["UCI Air Quality", "Upload CSV"],
        default="UCI Air Quality" if "uploaded_dataset" not in st.session_state else "Upload CSV",
        key="data_source",
    )
    if source == "Upload CSV":
        upload = st.file_uploader("CSV file", type=["csv"], key="csv_file")
        gap_code = st.text_input(
            "Code that means \"no reading\" (optional)",
            placeholder="for example -200 or -999",
            key="gap_code",
        )
        if st.button("Load this file", type="primary", icon=":material/upload:", disabled=upload is None):
            try:
                sentinel = float(gap_code) if gap_code.strip() else None
                file_bytes = upload.getvalue()
                table = load_uploaded_csv(file_bytes, sentinel)
                use_uploaded(upload.name, table, file_bytes)
                st.toast(f"Loaded {upload.name}", icon=":material/check_circle:")
                st.rerun()
            except ValueError as error:
                st.error(f"This file could not be used. {error}")
    elif "uploaded_dataset" in st.session_state:
        use_uci()
        st.rerun()

dataset = active_dataset()
raw, clean, split = dataset.raw, dataset.clean, dataset.split

if dataset.is_uci:
    with st.expander("About this dataset and its source", expanded=False, icon=":material/info:"):
        st.write(
            "An array of five metal-oxide chemical sensors ran at road level in a polluted "
            "area of an Italian city from March 2004 to April 2005. Co-located reference "
            "analysers measured the true gas concentrations. Readings are hourly averages."
        )
        st.markdown(
            f"- **Source:** [UCI Machine Learning Repository]({UCI_URL})\n"
            f"- **Download:** [{UCI_DOWNLOAD_URL}]({UCI_DOWNLOAD_URL})\n"
            f"- **DOI:** {UCI_DOI}\n"
            "- **Reference:** De Vito et al. (2008), *Sensors and Actuators B: Chemical*, 129(2)\n"
            "- **Licence:** CC BY 4.0"
        )

cols = st.columns(4)
cols[0].metric("Rows", f"{len(raw):,}")
cols[1].metric("Sensor columns", raw.shape[1])
cols[2].metric("Natural gaps", f"{int(raw.isna().sum().sum()):,}", help="Cells with no reading in the real data.")
cols[3].metric("Complete rows, used columns", f"{int((~clean.isna().any(axis=1)).sum()):,}")

preview, sensors, gaps, relations, distributions = st.tabs(
    ["Preview", "Sensors", "Missing values", "Correlation", "Distributions"]
)

with preview:
    if dataset.is_uci:
        st.caption("Cleaned table: real timestamps, and -200 replaced by an empty cell.")
    st.dataframe(raw.head(200), height=360)
    st.write("**Summary statistics**")
    st.dataframe(raw.describe().T.round(2))
    if dataset.is_uci:
        published = uci_published_table()
        with st.expander("File as published", icon=":material/description:"):
            st.caption(
                f"{published.shape[0]:,} rows and {published.shape[1]} columns, including "
                f"{int(published.isna().all(axis=1).sum())} blank rows and "
                f"{int(published.isna().all().sum())} blank columns."
            )
            st.dataframe(published.head(20))

with sensors:
    info = pd.DataFrame(
        [
            {
                "Sensor": column,
                "Measures": SENSOR_INFO.get(column, ("Uploaded column", "", ""))[0],
                "Unit": SENSOR_INFO.get(column, ("", "not stated", ""))[1],
                "Type": SENSOR_INFO.get(column, ("", "", "Uploaded"))[2],
                "Used in model": "Yes" if column in clean.columns else "No",
                "Minimum": raw[column].min(),
                "Mean": raw[column].mean(),
                "Maximum": raw[column].max(),
            }
            for column in raw.columns
        ]
    )
    st.dataframe(info.round(2), hide_index=True)
    for column, reason in dataset.dropped.items():
        st.info(f"**{column}** was left out of the model because {reason}.", icon=":material/info:")

with gaps:
    if dataset.is_uci:
        coded = int((uci_published_table().select_dtypes("number") == UCI_SENTINEL).sum().sum())
        st.write(
            f"The file stores {coded:,} failed readings as the code -200, not as blanks. "
            "Left in place, -200 would pass as a real measurement and distort every "
            "average, so it is converted to an empty cell first."
        )
    left, right = st.columns([2, 3], gap="medium")
    with left:
        table = raw.isna().sum().rename("Natural gaps").to_frame()
        table["Share"] = (raw.isna().mean() * 100).round(1).astype(str) + "%"
        st.dataframe(table)
    with right:
        chart(viz.missing_bars(raw, "Share of missing readings by sensor"), "explorer_missing")
    st.write(
        "These natural gaps have no known true value, so they are never used to score "
        "a method. Scores use artificial gaps only, placed on readings that do exist."
    )

with relations:
    st.write(
        "Reconstruction depends on these links: when sensors move together, the ones "
        "still working carry information about the ones that failed."
    )
    chart(viz.correlation_heatmap(clean, "Correlation between sensors"), "explorer_corr")

with distributions:
    column = st.selectbox("Sensor", list(clean.columns), key="distribution_sensor")
    chart(viz.distribution(clean, column, f"Distribution of {column}"), "explorer_distribution")

with card():
    st.subheader("How the data is prepared")
    st.markdown(
        f"1. **Clean:** convert gap codes to empty cells and remove blank rows and columns.\n"
        f"2. **Select features:** keep {clean.shape[1]} columns; a column more than half "
        "empty is dropped.\n"
        f"3. **Split by time:** training {len(split.train):,} rows, validation "
        f"{len(split.val):,}, test {len(split.test):,}. A random split would leak, "
        "since neighbouring hours are almost identical.\n"
        "4. **Normalise:** convert to z-scores, using training statistics only.\n"
        "5. **Encode:** continuous network: keep real values. Binary network: thermometer code.\n"
        "6. **Natural gaps stay empty:** only complete rows are stored as memories, so "
        "nothing invented enters the memory."
    )
