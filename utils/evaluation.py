"""Baselines, metrics, tuning, and the benchmark.

Rules that keep the evaluation honest
-------------------------------------
* Scores use artificially damaged cells only, because only those have a known
  true value. Natural gaps are never scored.
* Every method is tuned on the validation period. The test period is used only
  for the final numbers.
* A method never sees the hidden values: each one receives the table with those
  cells set to NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from models.hopfield import BinaryHopfieldImputer, ContinuousHopfield, build_windows
from utils.preprocessing import (
    SplitData,
    inject_noise,
    outage_missing_mask,
    random_missing_mask,
)

HOPFIELD_WINDOW = "Hopfield (3-hour)"
HOPFIELD_SINGLE = "Hopfield (1-hour)"
HOPFIELD_BINARY = "Hopfield (binary)"
KNN_SINGLE = "KNN (1-hour)"
KNN_WINDOW = "KNN (3-hour)"
MEAN = "Mean imputation"
INTERPOLATION = "Linear interpolation"

ALL_METHODS = [
    HOPFIELD_WINDOW,
    HOPFIELD_SINGLE,
    HOPFIELD_BINARY,
    KNN_WINDOW,
    KNN_SINGLE,
    INTERPOLATION,
    MEAN,
]
HOPFIELD_METHODS = [HOPFIELD_WINDOW, HOPFIELD_SINGLE, HOPFIELD_BINARY]

MISSING_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]
PATTERNS = {"Random gaps": random_missing_mask, "Sensor outages": outage_missing_mask}

DEFAULT_CONFIG = {
    "hopfield_single": {"beta": 8.0, "free_weight": 0.05},
    "hopfield_window": {"beta": 2.0, "free_weight": 0.05},
    "binary": {"levels": 8, "n_prototypes": 32, "rule": "pseudoinverse"},
    "knn_single": {"n_neighbors": 10},
    "knn_window": {"n_neighbors": 10},
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def score(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> dict:
    """MAE, MSE, RMSE and R2 over the cells selected by `mask`."""
    mask = mask & ~np.isnan(truth)
    if not mask.any():
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan, "cells": 0}
    t, e = truth[mask], estimate[mask]
    error = t - e
    mse = float(np.mean(error**2))
    total = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / total if total > 0 else np.nan
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2,
        "cells": int(mask.sum()),
    }


def per_sensor_scores(truth, estimate, mask, columns) -> pd.DataFrame:
    """One row of metrics per sensor, in whatever units the inputs are in."""
    rows = [
        {"sensor": name, **score(truth[:, [j]], estimate[:, [j]], mask[:, [j]])}
        for j, name in enumerate(columns)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def mean_impute(z: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """Fill with the training mean, which is exactly 0 after z-scoring."""
    return np.where(unknown | np.isnan(z), 0.0, z)


def interpolate_impute(z: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """Draw a straight line in time between the nearest known readings."""
    hidden = pd.DataFrame(np.where(unknown, np.nan, z))
    filled = hidden.interpolate(method="linear", limit_direction="both")
    return filled.fillna(0.0).to_numpy()


class KnnWindowImputer:
    """sklearn's KNNImputer, optionally on the same windows Hopfield uses."""

    def __init__(self, n_neighbors=10, context=0):
        self.n_neighbors = n_neighbors
        self.context = context

    def fit(self, z_train):
        self.n_features = z_train.shape[1]
        windows = build_windows(z_train, self.context)
        complete = windows[~np.isnan(windows).any(axis=1)]
        k = min(self.n_neighbors, max(len(complete) - 1, 1))
        self.imputer = KNNImputer(n_neighbors=k).fit(complete)
        return self

    def reconstruct(self, z, unknown):
        hidden = np.where(unknown, np.nan, z)
        filled = self.imputer.transform(build_windows(hidden, self.context))
        d = self.n_features
        return filled[:, self.context * d : (self.context + 1) * d]


@dataclass
class Toolkit:
    """Every fitted method, ready to use."""

    single: ContinuousHopfield
    window: ContinuousHopfield
    binary: BinaryHopfieldImputer
    knn_single: KnnWindowImputer
    knn_window: KnnWindowImputer
    config: dict

    def run(self, method: str, z: np.ndarray, unknown: np.ndarray):
        """Return (completed table, RetrievalTrace or None)."""
        if method == HOPFIELD_WINDOW:
            return self.window.reconstruct(z, unknown)
        if method == HOPFIELD_SINGLE:
            return self.single.reconstruct(z, unknown)
        if method == HOPFIELD_BINARY:
            return self.binary.reconstruct(z, unknown)
        if method == KNN_SINGLE:
            return self.knn_single.reconstruct(z, unknown), None
        if method == KNN_WINDOW:
            return self.knn_window.reconstruct(z, unknown), None
        if method == MEAN:
            return mean_impute(z, unknown), None
        if method == INTERPOLATION:
            return interpolate_impute(z, unknown), None
        raise ValueError(f"Unknown method: {method}")


def fit_toolkit(z_train: np.ndarray, config: dict | None = None) -> Toolkit:
    config = config or DEFAULT_CONFIG
    return Toolkit(
        single=ContinuousHopfield(context=0, **config["hopfield_single"]).fit(z_train),
        window=ContinuousHopfield(context=1, **config["hopfield_window"]).fit(z_train),
        binary=BinaryHopfieldImputer(**config["binary"]).fit(z_train),
        knn_single=KnnWindowImputer(context=0, **config["knn_single"]).fit(z_train),
        knn_window=KnnWindowImputer(context=1, **config["knn_window"]).fit(z_train),
        config=config,
    )


# --------------------------------------------------------------------------- #
# Tuning on the validation period
# --------------------------------------------------------------------------- #
def _validation_masks(z_val, rates, seed=2024):
    rng = np.random.default_rng(seed)
    return [make_mask(z_val, rate, rng) for make_mask in PATTERNS.values() for rate in rates]


def _mean_rmse(fitted, z_val, masks) -> float:
    return float(np.mean([score(z_val, fitted.reconstruct(z_val, m)[0] if _returns_trace(fitted)
                                else fitted.reconstruct(z_val, m), m)["RMSE"] for m in masks]))


def _returns_trace(fitted) -> bool:
    return not isinstance(fitted, KnnWindowImputer)


def tune(split: SplitData, quick=False, on_step: Callable[[str], None] | None = None) -> dict:
    """Pick each method's settings by validation RMSE.

    Returns a config for fit_toolkit, plus the search log under "search".
    quick=True uses a smaller grid, for tuning on uploaded files in the website.
    """
    z_train = split.to_z(split.train.to_numpy())
    z_val = split.to_z(split.val.to_numpy())
    masks = _validation_masks(z_val, [0.3] if quick else [0.1, 0.3, 0.5])
    say = on_step or (lambda text: None)
    log = []

    def search(family, candidates, build):
        best, best_rmse = None, np.inf
        for settings in candidates:
            say(f"{family}: {settings}")
            rmse = _mean_rmse(build(settings), z_val, masks)
            log.append({"family": family, **settings, "validation_rmse": rmse})
            if rmse < best_rmse:
                best, best_rmse = settings, rmse
        return best

    betas = [1.0, 2.0, 4.0, 8.0]
    gammas = [0.05] if quick else [0.05, 0.25]
    hopfield_grid = [{"beta": b, "free_weight": g} for b in betas for g in gammas]
    knn_grid = [{"n_neighbors": k} for k in ([5, 10] if quick else [5, 10, 20])]
    binary_grid = [
        {"levels": levels, "n_prototypes": prototypes, "rule": rule}
        for levels in ([8] if quick else [8, 16])
        for prototypes, rule in [(24, "hebbian"), (32, "pseudoinverse"), (64, "pseudoinverse")]
    ]

    config = {
        "hopfield_single": search(
            "hopfield_single", hopfield_grid,
            lambda s: ContinuousHopfield(context=0, **s).fit(z_train)),
        "hopfield_window": search(
            "hopfield_window", hopfield_grid,
            lambda s: ContinuousHopfield(context=1, **s).fit(z_train)),
        "binary": search(
            "binary", binary_grid, lambda s: BinaryHopfieldImputer(**s).fit(z_train)),
        "knn_single": search(
            "knn_single", knn_grid, lambda s: KnnWindowImputer(context=0, **s).fit(z_train)),
        "knn_window": search(
            "knn_window", knn_grid, lambda s: KnnWindowImputer(context=1, **s).fit(z_train)),
    }
    config["search"] = log
    return config


# --------------------------------------------------------------------------- #
# Benchmark: hidden values
# --------------------------------------------------------------------------- #
def run_benchmark(
    kit: Toolkit,
    z_test: np.ndarray,
    rates=MISSING_RATES,
    seeds=(0, 1, 2, 3, 4),
    methods=ALL_METHODS,
    patterns=PATTERNS,
    on_step: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Score every method on every pattern, rate and seed. One row per run."""
    rows = []
    total = len(patterns) * len(rates) * len(seeds)
    done = 0
    for pattern, make_mask in patterns.items():
        for rate in rates:
            for seed in seeds:
                # The same damaged table goes to every method in this run.
                mask = make_mask(z_test, rate, np.random.default_rng(seed))
                for method in methods:
                    completed, _ = kit.run(method, z_test, mask)
                    rows.append(
                        {"pattern": pattern, "rate": rate, "seed": seed, "method": method,
                         **score(z_test, completed, mask)}
                    )
                done += 1
                if on_step:
                    on_step(done / total, f"{pattern}, {rate:.0%} missing, run {seed + 1}")
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds, keeping the spread between seeds."""
    grouped = results.groupby(["pattern", "rate", "method"], sort=False)
    table = grouped[["MAE", "MSE", "RMSE", "R2"]].mean()
    table["RMSE_std"] = grouped["RMSE"].std().fillna(0.0)
    table["runs"] = grouped["RMSE"].size()
    return table.reset_index()


def iteration_ablation(kit: Toolkit, z_test, rate=0.3, seeds=(0, 1, 2), steps=(0, 1, 2, 5, 10, 20)):
    """Does running more update steps help? Reports accuracy at each step count."""
    rows = []
    for pattern, make_mask in PATTERNS.items():
        for seed in seeds:
            mask = make_mask(z_test, rate, np.random.default_rng(seed))
            for count in steps:
                completed, _ = kit.window.reconstruct(z_test, mask, max_iter=count)
                rows.append({"pattern": pattern, "seed": seed, "steps": count,
                             "RMSE": score(z_test, completed, mask)["RMSE"]})
    return pd.DataFrame(rows).groupby(["pattern", "steps"], as_index=False)["RMSE"].mean()


# --------------------------------------------------------------------------- #
# Corrupted readings: nobody tells the system which cells are wrong
# --------------------------------------------------------------------------- #
def leave_one_out_residuals(model: ContinuousHopfield, z, flagged=None) -> np.ndarray:
    """How far each reading is from what the other sensors imply.

    Each sensor is hidden in turn and recalled from the rest. Cells already
    flagged are hidden too, so a bad reading cannot vouch for another one.
    """
    flagged = np.zeros(z.shape, dtype=bool) if flagged is None else flagged
    residual = np.zeros(z.shape)
    for j in range(z.shape[1]):
        hide = flagged.copy()
        hide[:, j] = True
        recalled, _ = model.reconstruct(z, hide, max_iter=0)
        residual[:, j] = np.abs(z[:, j] - recalled[:, j])
    return residual


def calibrate_threshold(model, z_val, quantile=0.99) -> np.ndarray:
    """Per-sensor limit: residuals seen on real validation data set the bar."""
    return np.nanquantile(leave_one_out_residuals(model, z_val), quantile, axis=0)


def detect_corruption(model, z, threshold, passes=2) -> np.ndarray:
    """Flag readings that disagree with the memory. Natural gaps are never flagged."""
    flagged = np.zeros(z.shape, dtype=bool)
    for _ in range(passes):
        residual = leave_one_out_residuals(model, z, flagged)
        flagged = np.nan_to_num(residual, nan=0.0) > threshold
    return flagged


def run_corruption_benchmark(
    kit: Toolkit,
    z_test,
    threshold,
    kinds=(("gaussian", 2.0), ("spike", 4.0)),
    rates=(0.05, 0.1, 0.2),
    seeds=(0, 1, 2),
    methods=(HOPFIELD_WINDOW, KNN_WINDOW, KNN_SINGLE, INTERPOLATION, MEAN),
) -> pd.DataFrame:
    """Detect with the Hopfield memory, then repair with each method.

    Every method repairs the same flagged cells. "No repair" is the corrupted
    table left as it is. Scores cover the truly corrupted cells.
    """
    rows = []
    for kind, strength in kinds:
        for rate in rates:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                noisy, corrupted = inject_noise(z_test, rate, rng, kind=kind, strength=strength)
                flagged = detect_corruption(kit.single, noisy, threshold)
                caught = int((flagged & corrupted).sum())
                base = {
                    "kind": kind, "rate": rate, "seed": seed,
                    "precision": caught / max(int(flagged.sum()), 1),
                    "recall": caught / max(int(corrupted.sum()), 1),
                }
                rows.append({**base, "method": "No repair",
                             **score(z_test, np.nan_to_num(noisy), corrupted)})
                for method in methods:
                    completed, _ = kit.run(method, noisy, flagged)
                    rows.append({**base, "method": method, **score(z_test, completed, corrupted)})
    return pd.DataFrame(rows)
