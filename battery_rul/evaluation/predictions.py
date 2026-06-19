"""Run a trained model over a battery's processed parquet to produce actual-vs-predicted SOH."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from torch import nn

from battery_rul.config import WINDOW_SIZE


def predict_battery(
    model: nn.Module,
    parquet_path: Path,
    feature_columns: list[str],
    device: torch.device,
    input_mode: str,
    window_size: int = WINDOW_SIZE,
    stride: int = 1,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Slide a window over a battery's series and predict SOH at each target index.

    Returns columns: absolute_time_raw, voltage_raw, soh_actual, soh_predicted.
    """
    df = pd.read_parquet(parquet_path)
    features = df[feature_columns].to_numpy(dtype=np.float32)
    n_windows = len(df) - window_size + 1
    starts = list(range(0, n_windows, stride))

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            windows = np.stack([features[s : s + window_size] for s in batch_starts])
            x = torch.from_numpy(windows).to(device)
            if input_mode == "flat":
                x = x[:, -1, :]
            out = model(x).squeeze(-1).cpu().numpy()
            preds.append(out)
    soh_predicted = np.concatenate(preds)

    target_idx = [s + window_size - 1 for s in starts]
    result = df.iloc[target_idx][["absolute_time_raw", "voltage_raw", "soh"]].reset_index(drop=True)
    result = result.rename(columns={"soh": "soh_actual"})
    result["soh_predicted"] = soh_predicted
    return result


def predict_battery_lightgbm(
    model: LGBMRegressor,
    parquet_path: Path,
    feature_columns: list[str],
    stride: int = 1,
) -> pd.DataFrame:
    """Predict per-row SOH for a battery with a tree model — no windowing needed.

    Returns columns: absolute_time_raw, voltage_raw, soh_actual, soh_predicted.
    """
    df = pd.read_parquet(parquet_path)
    sampled = df.iloc[::stride]
    soh_predicted = model.predict(sampled[feature_columns].to_numpy(dtype=np.float32))

    result = sampled[["absolute_time_raw", "voltage_raw", "soh"]].reset_index(drop=True)
    result = result.rename(columns={"soh": "soh_actual"})
    result["soh_predicted"] = soh_predicted
    return result
