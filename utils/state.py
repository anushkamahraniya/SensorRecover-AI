"""Everything the pages share: the active dataset, scenario and results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from utils.evaluation import (
    ALL_METHODS,
    HOPFIELD_WINDOW,
    PATTERNS,
    Toolkit,
    calibrate_threshold,
    detect_corruption,
    fit_toolkit,
    per_sensor_scores,
    score,
    tune,
)
from utils.preprocessing import (
    PROJECT_ROOT,
    SENSOR_INFO,
    SplitData,
    inject_noise,
    load_uci_raw,
    prepare_uci,
    select_features,
    split_and_scale,
)

RESULTS = PROJECT_ROOT / "results"
UCI_NAME = "UCI Air Quality"


@dataclass
class Dataset:
    name: str
    key: str  # changes whenever the content changes; used for caching
    raw: pd.DataFrame  # after turning gap codes into NaN, before feature selection
    clean: pd.DataFrame
    dropped: dict
    split: SplitData

    @property
    def is_uci(self) -> bool:
        return self.name == UCI_NAME


@st.cache_data(show_spinner=False, max_entries=1)
def _uci_tables():
    raw = load_uci_raw()
    return raw, prepare_uci(raw)


def build_dataset(name: str, table: pd.DataFrame, key: str) -> Dataset:
    clean, dropped = select_features(table)
    return Dataset(name, key, table, clean, dropped, split_and_scale(clean))


def uci_published_table() -> pd.DataFrame:
    return _uci_tables()[0]


def active_dataset() -> Dataset:
    """The uploaded dataset if there is one, otherwise the UCI dataset."""
    uploaded = st.session_state.get("uploaded_dataset")
    if uploaded is not None:
        return uploaded
    return build_dataset(UCI_NAME, _uci_tables()[1], "uci")


def use_uploaded(name: str, table: pd.DataFrame, file_bytes: bytes) -> None:
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    dataset = build_dataset(name, table, digest)
    if len(dataset.split.test) < 20 or len(dataset.split.val) < 10:
        raise ValueError("The file is too short to split into train, validation and test.")
    train = dataset.split.train.to_numpy()
    if (~np.isnan(train).any(axis=1)).sum() < 20:
        raise ValueError(
            "The training period needs at least 20 rows with every sensor present, "
            "because only complete rows are stored as memories."
        )
    st.session_state["uploaded_dataset"] = dataset
    clear_results()


def use_uci() -> None:
    st.session_state.pop("uploaded_dataset", None)
    clear_results()


def clear_results() -> None:
    for key in ("scenario", "reconstruction", "live_benchmark"):
        st.session_state.pop(key, None)


def sensor_unit(column: str) -> str:
    return SENSOR_INFO.get(column, ("", "", ""))[1]


# --------------------------------------------------------------------------- #
# Fitted models
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False, max_entries=3)
def _toolkit(key: str, _dataset: Dataset):
    """Store the memories. Cached per dataset, and shared by all pages."""
    split = _dataset.split
    z_train = split.to_z(split.train.to_numpy())
    config_path = RESULTS / "model_config.json"
    if _dataset.is_uci and config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = tune(split, quick=True)
        config.pop("search", None)
    kit = fit_toolkit(z_train, config)
    threshold = calibrate_threshold(kit.single, split.to_z(split.val.to_numpy()))
    return kit, threshold


def get_toolkit(dataset: Dataset) -> tuple[Toolkit, np.ndarray]:
    return _toolkit(dataset.key, dataset)


# --------------------------------------------------------------------------- #
# Scenarios: a test-period table with damage applied
# --------------------------------------------------------------------------- #
def build_scenario(
    dataset: Dataset, sensors, missing_rate, pattern, noise_rate, noise_kind, noise_strength, seed
) -> dict:
    """Damage the test period. Nothing here touches the training data."""
    split = dataset.split
    z_true = split.to_z(split.test.to_numpy())
    columns = [split.columns.index(name) for name in sensors]
    rng = np.random.default_rng(seed)

    hidden = PATTERNS[pattern](z_true, missing_rate, rng, columns=columns)
    damaged = z_true.copy()
    corrupted = np.zeros(z_true.shape, dtype=bool)
    if noise_rate > 0:
        # Readings that are hidden cannot also be corrupted.
        visible = np.where(hidden, np.nan, z_true)
        noisy, corrupted = inject_noise(
            visible, noise_rate, rng, columns=columns, kind=noise_kind, strength=noise_strength
        )
        damaged = np.where(corrupted, noisy, z_true)
    damaged = np.where(hidden, np.nan, damaged)

    return {
        "dataset_key": dataset.key, "sensors": list(sensors), "rate": missing_rate,
        "pattern": pattern, "noise_rate": noise_rate, "noise_kind": noise_kind,
        "noise_strength": noise_strength, "seed": seed, "z_true": z_true, "z_damaged": damaged,
        "hidden": hidden, "corrupted": corrupted, "natural": np.isnan(z_true),
    }


def get_scenario(dataset: Dataset) -> dict | None:
    scenario = st.session_state.get("scenario")
    return scenario if scenario and scenario["dataset_key"] == dataset.key else None


def cell_states(scenario: dict) -> np.ndarray:
    states = np.zeros(scenario["z_true"].shape, dtype=int)
    states[scenario["natural"]] = 1
    states[scenario["hidden"]] = 2
    states[scenario["corrupted"]] = 3
    return states


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
def run_reconstruction(dataset: Dataset, scenario: dict, on_progress=None) -> dict:
    """Run every method on one scenario and score them on the same cells."""
    kit, threshold = get_toolkit(dataset)
    say = on_progress or (lambda share, text: None)
    z_true, z_damaged = scenario["z_true"], scenario["z_damaged"]
    hidden, corrupted = scenario["hidden"], scenario["corrupted"]

    flagged = np.zeros(z_true.shape, dtype=bool)
    if corrupted.any():
        say(0.05, "Checking each reading against the memory")
        flagged = detect_corruption(kit.single, z_damaged, threshold)

    unknown = hidden | flagged
    scored = hidden | corrupted
    completed, traces = {}, {}
    for step, method in enumerate(ALL_METHODS):
        say(0.15 + 0.8 * step / len(ALL_METHODS), f"Reconstructing with {method}")
        if method == HOPFIELD_WINDOW:
            start = 0.15 + 0.8 * step / len(ALL_METHODS)
            span = 0.8 / len(ALL_METHODS)
            result, trace = kit.window.reconstruct(
                z_damaged, unknown,
                on_batch=lambda share: say(start + span * share, f"Retrieving memories: {share:.0%}"),
            )
        else:
            result, trace = kit.run(method, z_damaged, unknown)
        completed[method] = result
        if trace is not None:
            traces[method] = trace

    metrics = pd.DataFrame(
        [{"method": m, **score(z_true, completed[m], scored)} for m in ALL_METHODS]
    )
    caught = int((flagged & corrupted).sum())
    say(1.0, "Finished")
    return {
        "dataset_key": dataset.key, "scenario": scenario, "completed": completed,
        "traces": traces, "metrics": metrics, "flagged": flagged, "scored": scored,
        "precision": caught / max(int(flagged.sum()), 1) if corrupted.any() else None,
        "recall": caught / max(int(corrupted.sum()), 1) if corrupted.any() else None,
    }


def get_reconstruction(dataset: Dataset) -> dict | None:
    result = st.session_state.get("reconstruction")
    return result if result and result["dataset_key"] == dataset.key else None


def default_scenario(dataset: Dataset) -> dict:
    return build_scenario(dataset, dataset.split.columns, 0.3, "Random gaps", 0.0, "gaussian", 2.0, 42)


def sensor_metrics(dataset: Dataset, result: dict) -> pd.DataFrame:
    """Errors per sensor in real units, for every method."""
    split, truth = dataset.split, result["scenario"]["z_true"]
    tables = []
    for method, completed in result["completed"].items():
        table = per_sensor_scores(
            split.from_z(truth), split.from_z(completed), result["scored"], split.columns
        )
        table.insert(0, "method", method)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def reconstructed_table(dataset: Dataset, result: dict, method: str) -> pd.DataFrame:
    """The completed test period in real units, with a source label per reading.

    The labels keep estimates apart from real readings: a natural gap has no
    true value, so its filled value is an estimate that could not be checked.
    """
    split, scenario = dataset.split, result["scenario"]
    table = pd.DataFrame(
        split.from_z(result["completed"][method]), index=split.test.index, columns=split.columns
    )
    source = np.full(scenario["z_true"].shape, "observed", dtype=object)
    source[scenario["natural"]] = "estimated_natural_gap"
    source[scenario["corrupted"]] = "corrupted_not_flagged"
    source[result["flagged"]] = "estimated_flagged_reading"
    source[scenario["hidden"]] = "estimated_hidden_reading"
    labels = pd.DataFrame(source, index=table.index, columns=[f"{c}__source" for c in split.columns])
    return pd.concat([table, labels], axis=1)


# --------------------------------------------------------------------------- #
# Saved benchmark
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=16)
def _read_csv(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_result(name: str) -> pd.DataFrame | None:
    """Read a saved result. The file time is in the cache key, so new runs show up."""
    path = RESULTS / name
    return _read_csv(str(path), path.stat().st_mtime) if path.exists() else None
