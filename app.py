"""SensorRecover AI: entry point. Run with:  streamlit run app.py"""

import streamlit as st

from utils.state import active_dataset
from utils.ui import apply_theme, sidebar_brand

st.set_page_config(page_title="SensorRecover AI", page_icon=":material/sensors:", layout="wide")
apply_theme()
sidebar_brand()

navigation = st.navigation(
    {
        "Overview": [
            st.Page("app_pages/0_Home.py", title="Home", icon=":material/home:", default=True),
            st.Page("app_pages/1_Dataset_Explorer.py", title="Dataset explorer", icon=":material/table_chart:"),
        ],
        "Reconstruct": [
            st.Page("app_pages/2_Missing_Data_Simulator.py", title="Missing data simulator", icon=":material/tune:"),
            st.Page("app_pages/3_Reconstruction.py", title="Hopfield reconstruction", icon=":material/memory:"),
        ],
        "Results": [
            st.Page("app_pages/4_Model_Performance.py", title="Model performance", icon=":material/monitoring:"),
            st.Page("app_pages/5_Visualization_Dashboard.py", title="Visualization dashboard", icon=":material/dashboard:"),
            st.Page("app_pages/6_Download_Results.py", title="Download results", icon=":material/download:"),
        ],
    }
)

try:
    dataset_name = active_dataset().name
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
st.sidebar.caption(f"Active dataset: {dataset_name}")

navigation.run()
