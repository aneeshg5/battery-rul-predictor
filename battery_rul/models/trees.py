"""Gradient-boosted tree baseline. Not an nn.Module — doesn't fit the PyTorch training loop,
operates directly on per-row tabular features instead of windowed sequences."""

from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor

from battery_rul.config import (
    LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_N_ESTIMATORS,
    LIGHTGBM_NUM_LEAVES,
    SEED,
)


class BatteryLightGBM:
    """Wraps LGBMRegressor with a fit/predict/save/load contract."""

    def __init__(
        self,
        n_estimators: int = LIGHTGBM_N_ESTIMATORS,
        learning_rate: float = LIGHTGBM_LEARNING_RATE,
        num_leaves: int = LIGHTGBM_NUM_LEAVES,
    ) -> None:
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=SEED,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BatteryLightGBM":
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x))

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "BatteryLightGBM":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
