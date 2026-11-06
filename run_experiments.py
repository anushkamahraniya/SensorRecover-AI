"""Reproduce every number shown in the website and the report.

Usage:  python run_experiments.py

The script tunes all methods on the validation period, then scores them once on
the test period, and writes the results to the results/ folder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.evaluation import (
    ALL_METHODS,
    calibrate_threshold,
    fit_toolkit,
    iteration_ablation,
    per_sensor_scores,
    run_benchmark,
    run_corruption_benchmark,
    summarise,
    tune,
)
from utils.preprocessing import (
    load_uci_raw,
    outage_missing_mask,
    prepare_uci,
    random_missing_mask,
    select_features,
    split_and_scale,
)

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    started = time.time()
    RESULTS.mkdir(exist_ok=True)

    clean, dropped = select_features(prepare_uci(load_uci_raw()))
    split = split_and_scale(clean)
    print(f"Sensors kept: {clean.shape[1]}, dropped: {dropped}")
    print(f"Rows: train {len(split.train)}, validation {len(split.val)}, test {len(split.test)}")

    print("Tuning on the validation period ...")
    config = tune(split, on_step=lambda text: print("  ", text, flush=True))
    pd.DataFrame(config.pop("search")).to_csv(RESULTS / "tuning_log.csv", index=False)
    (RESULTS / "model_config.json").write_text(json.dumps(config, indent=2))
    print("Chosen settings:", config)

    z_train = split.to_z(split.train.to_numpy())
    z_val = split.to_z(split.val.to_numpy())
    z_test = split.to_z(split.test.to_numpy())
    kit = fit_toolkit(z_train, config)

    print("Scoring on the test period ...")
    results = run_benchmark(
        kit, z_test, on_step=lambda share, text: print(f"  {share:4.0%} {text}", flush=True)
    )
    results.to_csv(RESULTS / "benchmark_runs.csv", index=False)
    summary = summarise(results)
    summary.to_csv(RESULTS / "benchmark_summary.csv", index=False)

    # Per-sensor errors in real units, at 30% missing.
    truth = split.test.to_numpy()
    sensor_tables = []
    for pattern, make_mask in (("Random gaps", random_missing_mask),
                               ("Sensor outages", outage_missing_mask)):
        mask = make_mask(z_test, 0.3, np.random.default_rng(0))
        for method in ALL_METHODS:
            completed, _ = kit.run(method, z_test, mask)
            table = per_sensor_scores(truth, split.from_z(completed), mask, split.columns)
            table.insert(0, "method", method)
            table.insert(0, "pattern", pattern)
            sensor_tables.append(table)
    pd.concat(sensor_tables).to_csv(RESULTS / "per_sensor_30pct.csv", index=False)

    print("Iteration ablation ...")
    iteration_ablation(kit, z_test).to_csv(RESULTS / "iteration_ablation.csv", index=False)

    print("Corruption benchmark ...")
    threshold = calibrate_threshold(kit.single, z_val)
    corruption = run_corruption_benchmark(kit, z_test, threshold)
    corruption.to_csv(RESULTS / "corruption_runs.csv", index=False)

    print(f"Done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
