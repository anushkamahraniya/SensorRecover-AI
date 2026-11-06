"""Drive every page of the real app headlessly. Run: python -m pytest tests -q"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = ["app_pages/0_Home.py", "app_pages/1_Dataset_Explorer.py", "app_pages/2_Missing_Data_Simulator.py",
         "app_pages/3_Reconstruction.py", "app_pages/4_Model_Performance.py",
         "app_pages/5_Visualization_Dashboard.py", "app_pages/6_Download_Results.py"]


def open_page(path):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.switch_page(path)
    at.run()
    return at


@pytest.mark.parametrize("path", PAGES)
def test_page_opens_without_error(path):
    at = open_page(path)
    assert not at.exception, at.exception


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception, at.exception


def test_full_user_journey():
    at = open_page("app_pages/2_Missing_Data_Simulator.py")
    at.slider(key="sim_noise").set_value(10)
    at.run()
    click(at, "Apply damage")
    assert at.session_state["scenario"]["corrupted"].any()
    assert at.session_state["scenario"]["hidden"].any()

    at.switch_page("app_pages/3_Reconstruction.py")
    at.run()
    click(at, "Run reconstruction")
    result = at.session_state["reconstruction"]
    assert result["precision"] is not None
    assert len(result["metrics"]) == 7

    for path in ("app_pages/5_Visualization_Dashboard.py", "app_pages/6_Download_Results.py"):
        at.switch_page(path)
        at.run()
        assert not at.exception, at.exception


def test_uploaded_dataset_runs_through_the_whole_app():
    from utils.preprocessing import UCI_CSV_PATH, load_uploaded_csv
    from utils.state import build_dataset

    lines = UCI_CSV_PATH.read_bytes().splitlines(keepends=True)[:1500]
    data = b"".join(lines)
    table = load_uploaded_csv(data, sentinel=-200)
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["uploaded_dataset"] = build_dataset("sample.csv", table, "upload-test")
    at.run()
    assert not at.exception, at.exception

    at.switch_page("app_pages/3_Reconstruction.py")
    at.run()
    click(at, "Run reconstruction")
    assert len(at.session_state["reconstruction"]["metrics"]) == 7
    for path in PAGES:
        at.switch_page(path)
        at.run()
        assert not at.exception, at.exception
