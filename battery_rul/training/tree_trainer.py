"""Training path for tree models, which don't fit the epoch-based PyTorch Trainer loop."""

import logging

import mlflow
import mlflow.lightgbm
import pandas as pd

from battery_rul.evaluation.metrics import mae, r2_score, rmse
from battery_rul.models.trees import BatteryLightGBM

logger = logging.getLogger(__name__)


def fit_lightgbm(
    train_df: pd.DataFrame,
    eval_dfs: dict[str, pd.DataFrame],
    feature_columns: list[str],
    mlflow_run_name: str,
) -> dict[str, float]:
    """Fit a BatteryLightGBM on train_df and log params + per-battery RMSE/MAE/R2 to MLflow.

    Returns a dict of battery name -> RMSE.
    """
    x_train = train_df[feature_columns].to_numpy()
    y_train = train_df["soh"].to_numpy()

    model = BatteryLightGBM()
    results: dict[str, float] = {}

    with mlflow.start_run(run_name=mlflow_run_name):
        mlflow.log_params(model.model.get_params())
        model.fit(x_train, y_train)

        for name, df in eval_dfs.items():
            x_eval = df[feature_columns].to_numpy()
            y_eval = df["soh"].to_numpy()
            y_pred = model.predict(x_eval)
            metrics = {
                "rmse": rmse(y_eval, y_pred),
                "mae": mae(y_eval, y_pred),
                "r2": r2_score(y_eval, y_pred),
            }
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
            results[name] = metrics["rmse"]
            logger.info("%s rmse=%.4f mae=%.4f r2=%.4f", name, *metrics.values())

        mlflow.lightgbm.log_model(model.model, name="model", serialization_format="pickle")

    return results
