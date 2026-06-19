"""Optuna hyperparameter search for UpgradedDNN over Approach 2 (train on RW9)."""

import logging

import optuna
import torch
from torch.utils.data import DataLoader, Subset

from battery_rul.data.dataset import BatteryDataset
from battery_rul.data.preprocess import FEATURE_COLUMNS
from battery_rul.models.dnn import UpgradedDNN
from battery_rul.training.trainer import Trainer, get_default_device

logger = logging.getLogger(__name__)

TUNING_EPOCHS = 10
TUNING_PATIENCE = 3


def objective(
    trial: optuna.Trial,
    train_subset: Subset,
    val_subset: Subset,
    device: torch.device,
) -> float:
    """One Optuna trial: sample hyperparameters, train UpgradedDNN, return best val RMSE."""
    n_layers = trial.suggest_int("n_layers", 2, 5)
    layer_sizes = [trial.suggest_int(f"n_units_l{i}", 8, 128) for i in range(n_layers)]
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_subset, batch_size=batch_size)

    model = UpgradedDNN(input_dim=len(FEATURE_COLUMNS), layer_sizes=layer_sizes, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(
        model,
        optimizer,
        device,
        mlflow_run_name=f"optuna_trial_{trial.number}",
        input_mode="flat",
    )
    result = trainer.fit(
        train_dl, val_dl, epochs=TUNING_EPOCHS, early_stopping_patience=TUNING_PATIENCE
    )
    return result["best_val_rmse"]


def run_study(
    train_dataset: BatteryDataset, n_trials: int, device: torch.device | None = None
) -> optuna.Study:
    """Split train_dataset 80/20 and run an Optuna study minimizing val RMSE."""
    device = device if device is not None else get_default_device()
    n_train = int(0.8 * len(train_dataset))
    train_subset = Subset(train_dataset, range(n_train))
    val_subset = Subset(train_dataset, range(n_train, len(train_dataset)))

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_subset, val_subset, device),
        n_trials=n_trials,
    )
    return study
