"""Load a trained model + scaler and run single-reading SOH inference."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from battery_rul.config import (
    INFERENCE_DEFAULTS_PATH,
    RUL_HEALTHY_SOH,
    RUL_REPLACE_SOH,
    SAMPLE_INTERVAL_SECONDS,
)
from battery_rul.data.preprocess import FEATURE_COLUMNS, ROLLING_WINDOW
from battery_rul.training.trainer import get_default_device

STEP_TYPE_CODE = {"charge": 1, "discharge": -1, "rest": 0}


def rul_estimate(soh: float) -> str:
    """RUL bucket from SOH: paper's 80% threshold informs the "replace soon" cutoff."""
    if soh < RUL_REPLACE_SOH:
        return "Replace soon"
    if soh > RUL_HEALTHY_SOH:
        return "Healthy"
    return "Monitor"


class Predictor:
    """Loads a standalone model + scaler artifact and predicts SOH from live readings."""

    def __init__(
        self, model_path: Path, scaler_path: Path, device: torch.device | None = None
    ) -> None:
        self.device = device or get_default_device()
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()

        with scaler_path.open("rb") as f:
            self.scaler: MinMaxScaler = pickle.load(f)

        defaults_path = scaler_path.parent / INFERENCE_DEFAULTS_PATH.name
        with defaults_path.open() as f:
            self.defaults: dict[str, float] = json.load(f)

        voltage_idx = FEATURE_COLUMNS.index("voltage")
        self._voltage_min = float(self.scaler.data_min_[voltage_idx])
        self._voltage_max = float(self.scaler.data_max_[voltage_idx])

    def predict(
        self, voltage_history: list[float], current: float, temperature: float, step_type: str
    ) -> dict[str, float | str]:
        """Predict SOH from the most recent voltage readings plus instantaneous current/temp.

        voltage_history must have at least 2 readings (needed for the dv/dt feature);
        the rolling mean/std features use up to the last ROLLING_WINDOW readings.
        """
        voltages = np.asarray(voltage_history, dtype=np.float64)
        recent = voltages[-ROLLING_WINDOW:]
        voltage = float(voltages[-1])
        dv_dt = float((voltages[-1] - voltages[-2]) / SAMPLE_INTERVAL_SECONDS)

        raw_features = {
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "absolute_time": self.defaults["absolute_time"],
            "step_type": STEP_TYPE_CODE[step_type],
            "voltage_rolling_mean_10": float(recent.mean()),
            "voltage_rolling_std_10": float(recent.std(ddof=1)) if len(recent) > 1 else 0.0,
            "dv_dt": dv_dt,
            "cycle_count": self.defaults["cycle_count"],
        }
        x = np.array([[raw_features[col] for col in FEATURE_COLUMNS]], dtype=np.float64)
        x_scaled = self.scaler.transform(x)

        with torch.no_grad():
            tensor = torch.from_numpy(x_scaled).float().to(self.device)
            soh = float(self.model(tensor).squeeze().item())
        soh = min(max(soh, 0.0), 1.0)

        confidence = "high" if self._voltage_min <= voltage <= self._voltage_max else "low"
        return {"soh": soh, "rul_estimate": rul_estimate(soh), "confidence": confidence}
