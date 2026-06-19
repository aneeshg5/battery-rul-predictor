"""Evaluation metrics for SOH prediction, including safety-oriented error splits."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def overestimation_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE over points where the prediction exceeds the actual value."""
    mask = y_pred > y_true
    if not np.any(mask):
        return 0.0
    return rmse(y_true[mask], y_pred[mask])


def underestimation_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE over points where the prediction falls below the actual value."""
    mask = y_pred < y_true
    if not np.any(mask):
        return 0.0
    return rmse(y_true[mask], y_pred[mask])


def safety_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ratio of underestimates to overestimates; > 1.0 means a safer, more conservative model."""
    underestimates = int(np.sum(y_pred < y_true))
    overestimates = int(np.sum(y_pred > y_true))
    if overestimates == 0:
        return float("inf") if underestimates > 0 else 0.0
    return underestimates / overestimates
