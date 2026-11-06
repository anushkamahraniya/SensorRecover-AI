"""Data loading, cleaning, splitting and corruption utilities.

Vocabulary used throughout the project
--------------------------------------
natural gap     : a value that is missing in the real dataset (we do NOT know its
                  true value, so it can never be used for scoring).
artificial gap  : a value we hid on purpose (we DO know its true value, so this
                  is the only kind of cell that is ever scored).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UCI_CSV_PATH = PROJECT_ROOT / "data" / "AirQualityUCI.csv"
UCI_URL = "https://archive.ics.uci.edu/dataset/360/air+quality"
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
UCI_DOI = "10.24432/C59K5F"

# The UCI file uses -200 as its "no reading" code.
UCI_SENTINEL = -200.0

# Columns with more missing values than this are dropped: too little evidence
# remains to learn from them and too little ground truth to test on them.
MAX_MISSING_FRACTION = 0.5

SENSOR_INFO = {
    "CO(GT)": ("Carbon monoxide, reference analyser", "mg/m³", "Reference analyser"),
    "PT08.S1(CO)": ("Tin-oxide sensor targeting CO", "sensor response", "Metal-oxide sensor"),
    "NMHC(GT)": ("Non-methane hydrocarbons, reference analyser", "µg/m³", "Reference analyser"),
    "C6H6(GT)": ("Benzene, reference analyser", "µg/m³", "Reference analyser"),
    "PT08.S2(NMHC)": ("Titania sensor targeting NMHC", "sensor response", "Metal-oxide sensor"),
    "NOx(GT)": ("Nitrogen oxides, reference analyser", "ppb", "Reference analyser"),
    "PT08.S3(NOx)": ("Tungsten-oxide sensor targeting NOx", "sensor response", "Metal-oxide sensor"),
    "NO2(GT)": ("Nitrogen dioxide, reference analyser", "µg/m³", "Reference analyser"),
    "PT08.S4(NO2)": ("Tungsten-oxide sensor targeting NO2", "sensor response", "Metal-oxide sensor"),
    "PT08.S5(O3)": ("Indium-oxide sensor targeting O3", "sensor response", "Metal-oxide sensor"),
    "T": ("Air temperature", "°C", "Weather"),
    "RH": ("Relative humidity", "%", "Weather"),
    "AH": ("Absolute humidity", "g/m³", "Weather"),
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_uci_raw(path: Path | str = UCI_CSV_PATH) -> pd.DataFrame:
    """Read the UCI file exactly as published (semicolons, decimal commas)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from {UCI_DOWNLOAD_URL} "
            "and place AirQualityUCI.csv in the data/ folder."
        )
    return pd.read_csv(path, sep=";", decimal=",")


def prepare_uci(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw UCI table into a time-indexed table with real NaNs."""
    df = raw.dropna(axis=1, how="all").dropna(how="all").copy()
    stamp = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    df = df.drop(columns=["Date", "Time"])
    df.index = pd.DatetimeIndex(stamp, name="timestamp")
    df = df.astype(float).replace(UCI_SENTINEL, np.nan)
    return df[~df.index.duplicated(keep="first")].sort_index()


def load_uploaded_csv(file_bytes: bytes, sentinel: float | None = None) -> pd.DataFrame:
    """Read a user CSV, guessing its layout, and keep the numeric columns.

    Raises ValueError with a readable message if the file cannot be used.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    first_line = file_bytes[:4096].decode("utf-8", errors="ignore").splitlines()
    header = first_line[0] if first_line else ""
    separator = max([",", ";", "\t"], key=header.count)
    # With semicolon or tab separators, a comma is usually the decimal mark.
    decimal = "," if separator != "," else "."
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=separator, decimal=decimal)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise ValueError(f"This file could not be read as a CSV: {error}") from error

    df = df.dropna(axis=1, how="all").dropna(how="all")
    df = _attach_time_index(df)

    numeric = df.apply(pd.to_numeric, errors="coerce")
    # Keep a column only if most of its filled cells really are numbers.
    filled = df.notna().sum().clip(lower=1)
    numeric = numeric.loc[:, numeric.notna().sum() / filled >= 0.8].astype(float)
    if sentinel is not None:
        numeric = numeric.replace(sentinel, np.nan)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric sensor columns are needed.")
    if len(numeric) < 100:
        raise ValueError("At least 100 rows are needed to train and test the model.")
    return numeric


def _attach_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Use the first date-like text column as the index when there is one."""
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        name = str(column).lower()
        if not any(word in name for word in ("date", "time", "stamp")):
            continue
        stamp = pd.to_datetime(df[column], errors="coerce")
        # A date-only column repeats within a day. It cannot identify a row, so
        # using it would make the duplicate check below delete real readings.
        if stamp.notna().mean() > 0.9 and stamp.nunique() > 0.95 * len(stamp):
            out = df.drop(columns=[column])
            out.index = pd.DatetimeIndex(stamp, name="timestamp")
            out = out[out.index.notna()]
            return out[~out.index.duplicated(keep="first")].sort_index()
    return df


def select_features(df: pd.DataFrame, max_missing: float = MAX_MISSING_FRACTION):
    """Drop unusable columns. Returns (kept table, dropped {column: reason})."""
    dropped: dict[str, str] = {}
    keep = []
    for column in df.columns:
        fraction = df[column].isna().mean()
        if fraction > max_missing:
            dropped[column] = f"{fraction:.1%} of values are missing"
        elif df[column].nunique(dropna=True) <= 1:
            dropped[column] = "the column never changes"
        else:
            keep.append(column)
    if len(keep) < 2:
        raise ValueError("Fewer than two usable sensor columns remain after cleaning.")
    return df[keep], dropped


# --------------------------------------------------------------------------- #
# Train / validation / test split and scaling
# --------------------------------------------------------------------------- #
@dataclass
class SplitData:
    """Chronological split plus the scaler, all fitted on the training part only.

    The split is by time, not random. Neighbouring hours are nearly identical,
    so a random split would put near-copies of test rows in the training set and
    make every method look better than it would on genuinely new data.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    mean: np.ndarray
    std: np.ndarray
    columns: list[str] = field(default_factory=list)

    def to_z(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def from_z(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def split_and_scale(df: pd.DataFrame, train: float = 0.70, val: float = 0.15) -> SplitData:
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    train_df, val_df, test_df = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    # nanmean/nanstd skip natural gaps, so nothing is filled in to compute them.
    mean = np.nanmean(train_df.to_numpy(), axis=0)
    std = np.nanstd(train_df.to_numpy(), axis=0)
    std = np.where(std > 0, std, 1.0)
    return SplitData(train_df, val_df, test_df, mean, std, list(df.columns))


# --------------------------------------------------------------------------- #
# Artificial damage. Every function below only touches naturally observed cells,
# so each damaged cell has a known true value to be scored against.
# --------------------------------------------------------------------------- #
def random_missing_mask(values, rate, rng, columns=None) -> np.ndarray:
    """Hide `rate` of the observed cells uniformly at random (MCAR)."""
    allowed = _allowed_cells(values, columns)
    mask = np.zeros(values.shape, dtype=bool)
    cells = np.flatnonzero(allowed)
    chosen = rng.choice(cells, size=int(round(rate * cells.size)), replace=False)
    mask.flat[chosen] = True
    return mask


def outage_missing_mask(values, rate, rng, columns=None, outage_hours=24) -> np.ndarray:
    """Hide contiguous runs of one sensor, as when a device really fails.

    Runs of `outage_hours` are placed until `rate` of the cells are hidden.
    """
    allowed = _allowed_cells(values, columns)
    mask = np.zeros(values.shape, dtype=bool)
    target = int(round(rate * allowed.sum()))
    usable = np.flatnonzero(allowed.any(axis=0))
    n = values.shape[0]
    length = min(outage_hours, n)
    for _ in range(100_000):
        if mask.sum() >= target:
            break
        column = rng.choice(usable)
        start = rng.integers(0, n - length + 1)
        mask[start : start + length, column] = True
        mask &= allowed
    return mask


def inject_noise(values, rate, rng, columns=None, kind="gaussian", strength=3.0):
    """Corrupt `rate` of the observed cells. Works on z-scored values.

    gaussian : add random noise with standard deviation `strength`
    spike    : jump by +/- `strength` standard deviations
    stuck    : freeze the reading at its value `strength` hours earlier
    Returns (corrupted values, boolean mask of the corrupted cells).
    """
    mask = random_missing_mask(values, rate, rng, columns)
    out = values.copy()
    count = int(mask.sum())
    if kind == "gaussian":
        out[mask] += rng.normal(0.0, strength, count)
    elif kind == "spike":
        out[mask] += rng.choice([-1.0, 1.0], count) * strength
    elif kind == "stuck":
        lag = max(int(strength), 1)
        rows, cols = np.nonzero(mask)
        earlier = values[np.maximum(rows - lag, 0), cols]
        usable = ~np.isnan(earlier)
        out[rows[usable], cols[usable]] = earlier[usable]
        mask[rows[~usable], cols[~usable]] = False
    else:
        raise ValueError(f"Unknown corruption type: {kind}")
    return out, mask


def _allowed_cells(values, columns) -> np.ndarray:
    allowed = ~np.isnan(values)
    if columns is not None:
        keep = np.zeros(values.shape[1], dtype=bool)
        keep[list(columns)] = True
        allowed &= keep
    return allowed
