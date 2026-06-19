"""Generic training loop with MLflow logging and early stopping."""

import copy
import logging

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from battery_rul.evaluation.metrics import mae, r2_score, rmse

logger = logging.getLogger(__name__)


def get_default_device() -> torch.device:
    """Pick the fastest available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rmse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE loss, matching the paper's loss function."""
    return torch.sqrt(nn.functional.mse_loss(prediction, target))


class Trainer:
    """Trains a model with RMSE loss, MLflow logging, and early stopping.

    input_mode "flat" feeds the final timestep of each window to the model (for
    instantaneous-prediction models like the DNNs); "sequence" feeds the full window
    (for the LSTM).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mlflow_run_name: str,
        input_mode: str = "sequence",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.mlflow_run_name = mlflow_run_name
        self.input_mode = input_mode

    def _forward_batch(self, features: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "flat":
            features = features[:, -1, :]
        prediction = self.model(features)
        return prediction.squeeze(-1)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch and return the example-weighted RMSE."""
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        for features, targets in dataloader:
            features, targets = features.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            predictions = self._forward_batch(features)
            loss = rmse_loss(predictions, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(targets)
            total_examples += len(targets)
        return total_loss / total_examples

    def eval_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate on a dataloader and return {rmse, mae, r2}."""
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(self.device)
                predictions = self._forward_batch(features).cpu().numpy()
                all_preds.append(predictions)
                all_targets.append(targets.numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def fit(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epochs: int,
        early_stopping_patience: int = 5,
    ) -> dict[str, float]:
        """Train for up to `epochs`, restoring the best validation-RMSE weights on exit."""
        best_val_rmse = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        with mlflow.start_run(run_name=self.mlflow_run_name):
            mlflow.log_params({"epochs": epochs, "patience": early_stopping_patience})
            for epoch in range(epochs):
                train_rmse = self.train_epoch(train_dl)
                val_metrics = self.eval_epoch(val_dl)
                mlflow.log_metrics(
                    {"train_rmse": train_rmse, **{f"val_{k}": v for k, v in val_metrics.items()}},
                    step=epoch,
                )
                logger.info(
                    "epoch %d/%d train_rmse=%.4f val_rmse=%.4f",
                    epoch + 1,
                    epochs,
                    train_rmse,
                    val_metrics["rmse"],
                )

                if val_metrics["rmse"] < best_val_rmse:
                    best_val_rmse = val_metrics["rmse"]
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        logger.info("early stopping at epoch %d", epoch + 1)
                        break

            self.model.load_state_dict(best_state)
            mlflow.log_metric("best_val_rmse", best_val_rmse)
            mlflow.pytorch.log_model(self.model, name="model", serialization_format="pickle")

        return {"best_val_rmse": best_val_rmse}
