"""Flatten raw NASA RW .mat files into feature-engineered, scaled parquet files."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

from battery_rul.config import (
    BATTERIES,
    PROCESSED_DIR,
    RAW_DIR,
    SOH_THRESHOLD,
    TRAIN_BATTERY,
)

logger = logging.getLogger(__name__)

STEP_TYPE_CODE = {"C": 1, "D": -1, "R": 0}
REFERENCE_CHARGE_COMMENT = "reference charge"
ROLLING_WINDOW = 10

FEATURE_COLUMNS = [
    "voltage",
    "current",
    "temperature",
    "absolute_time",
    "step_type",
    "voltage_rolling_mean_10",
    "voltage_rolling_std_10",
    "dv_dt",
    "cycle_count",
]


def load_raw_mat(path: Path) -> pd.DataFrame:
    """Flatten a NASA RW battery .mat file's nested step structure into a long DataFrame."""
    raw = loadmat(path, simplify_cells=True)
    steps = raw["data"]["step"]

    absolute_time: list[np.ndarray] = []
    relative_time: list[np.ndarray] = []
    voltage: list[np.ndarray] = []
    current: list[np.ndarray] = []
    temperature: list[np.ndarray] = []
    step_type: list[np.ndarray] = []
    comment: list[str] = []
    cycle_count: list[np.ndarray] = []

    cycle = 0
    for step in steps:
        v = np.atleast_1d(step["voltage"]).astype(np.float64)
        n = v.shape[0]
        if step["comment"] == REFERENCE_CHARGE_COMMENT:
            cycle += 1

        absolute_time.append(np.atleast_1d(step["time"]).astype(np.float64))
        relative_time.append(np.atleast_1d(step["relativeTime"]).astype(np.float64))
        voltage.append(v)
        current.append(np.atleast_1d(step["current"]).astype(np.float64))
        temperature.append(np.atleast_1d(step["temperature"]).astype(np.float64))
        step_type.append(np.full(n, STEP_TYPE_CODE[step["type"]], dtype=np.int8))
        cycle_count.append(np.full(n, cycle, dtype=np.int32))
        comment.extend([step["comment"]] * n)

    return pd.DataFrame(
        {
            "absolute_time": np.concatenate(absolute_time),
            "relative_time": np.concatenate(relative_time),
            "voltage": np.concatenate(voltage),
            "current": np.concatenate(current),
            "temperature": np.concatenate(temperature),
            "step_type": np.concatenate(step_type),
            "cycle_count": np.concatenate(cycle_count),
            "comment": pd.Categorical(comment),
        }
    )


def compute_soh(voltage: pd.Series, threshold: float = SOH_THRESHOLD) -> pd.Series:
    """Compute State of Health from voltage: SOH = (Vi - Vf) / (V0 - Vf), Vf = threshold * V0.

    V0 is the 99.9th percentile voltage (proxy for a fresh full charge) rather than
    the first sample or raw max, since a random-walk trace can start at an arbitrary
    point in the charge/discharge cycle and the raw max is sensitive to single-sample
    sensor noise spikes.
    """
    v0 = voltage.quantile(0.999)
    vf = threshold * v0
    return (voltage - vf) / (v0 - vf)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling voltage statistics and the voltage derivative."""
    df = df.copy()
    df["voltage_rolling_mean_10"] = df["voltage"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df["voltage_rolling_std_10"] = (
        df["voltage"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0.0)
    )
    dt = df["absolute_time"].diff()
    dv = df["voltage"].diff()
    df["dv_dt"] = (dv / dt.replace(0, np.nan)).fillna(0.0)
    return df


def build_features(raw_path: Path) -> pd.DataFrame:
    """Load a raw .mat file and produce the full feature + target DataFrame."""
    df = load_raw_mat(raw_path)
    df["soh"] = compute_soh(df["voltage"])
    df = add_engineered_features(df)
    # Kept unscaled for dashboard plotting and inference defaults; FEATURE_COLUMNS
    # scaling below doesn't touch these.
    df["absolute_time_raw"] = df["absolute_time"]
    df["voltage_raw"] = df["voltage"]
    df["cycle_count_raw"] = df["cycle_count"]
    return df


def run_preprocessing(
    raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> dict[str, Path]:
    """Process all batteries: fit the scaler on TRAIN_BATTERY, transform all, write parquet."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: build_features(raw_dir / f"{name}.mat") for name in BATTERIES}

    scaler = MinMaxScaler()
    scaler.fit(frames[TRAIN_BATTERY][FEATURE_COLUMNS])

    outputs: dict[str, Path] = {}
    for name in BATTERIES:
        df = frames[name]
        df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
        out_path = processed_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        outputs[name] = out_path
        logger.info("Wrote %s (%d rows)", out_path, len(df))

    scaler_path = processed_dir / "scaler.pkl"
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to %s", scaler_path)

    return outputs
