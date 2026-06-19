"""PyTorch Dataset and DataLoader construction for windowed battery feature sequences."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from battery_rul.config import (
    BATCH_SIZE,
    PROCESSED_DIR,
    TEST_BATTERIES,
    TRAIN_BATTERY,
    WINDOW_SIZE,
)
from battery_rul.data.preprocess import FEATURE_COLUMNS


class BatteryDataset(Dataset):
    """Windowed (feature history, SOH target) pairs from one battery's processed parquet."""

    def __init__(
        self,
        parquet_path: Path,
        window_size: int = WINDOW_SIZE,
        feature_columns: list[str] | None = None,
        stride: int = 1,
    ) -> None:
        columns = feature_columns if feature_columns is not None else FEATURE_COLUMNS
        df = pd.read_parquet(parquet_path, columns=[*columns, "soh"])
        self.features = df[columns].to_numpy(dtype=np.float32)
        self.targets = df["soh"].to_numpy(dtype=np.float32)
        self.window_size = window_size
        self.stride = stride

    def __len__(self) -> int:
        n_windows = len(self.targets) - self.window_size + 1
        return (n_windows + self.stride - 1) // self.stride

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        window = self.features[start : start + self.window_size]
        target = self.targets[start + self.window_size - 1]
        return torch.from_numpy(window), torch.tensor(target)


def get_dataloaders(
    train_battery: str = TRAIN_BATTERY,
    test_batteries: list[str] | None = None,
    batch_size: int = BATCH_SIZE,
    window_size: int = WINDOW_SIZE,
    feature_columns: list[str] | None = None,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Build the training DataLoader for train_battery and one per test battery."""
    test_batteries = test_batteries if test_batteries is not None else TEST_BATTERIES

    train_ds = BatteryDataset(
        processed_dir / f"{train_battery}.parquet", window_size, feature_columns
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_dls = {
        name: DataLoader(
            BatteryDataset(processed_dir / f"{name}.parquet", window_size, feature_columns),
            batch_size=batch_size,
            shuffle=False,
        )
        for name in test_batteries
    }
    return train_dl, test_dls
