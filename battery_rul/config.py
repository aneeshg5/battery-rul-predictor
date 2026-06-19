"""Centralized configuration: paths, hyperparameters, and constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"

BATTERIES = ["RW9", "RW10", "RW11", "RW12"]
TRAIN_BATTERY = "RW9"
TEST_BATTERIES = ["RW10", "RW11", "RW12"]

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
HIDDEN_LAYERS = [15, 10]
EPOCHS = 6
WINDOW_SIZE = 50

SOH_THRESHOLD = 0.80
SEED = 42

# Published SRA 2023 baselines, Avg RMSE (Approach 2) — for dashboard/README comparison only.
PAPER_BASELINE_RMSE = {
    "BLS-RVM": 0.0155,
    "RNN + LSTM": 0.0161,
    "Paper DNN (original)": 0.0149,
}

# Inference API: serves the upgraded_dnn / approach-2 model exported from MLflow.
MODEL_VERSION = "upgraded_dnn_v1"
MODEL_PATH = PROCESSED_DIR / "model_upgraded_dnn_v1.pt"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"
INFERENCE_DEFAULTS_PATH = PROCESSED_DIR / "inference_defaults.json"
UPGRADED_DNN_RMSE = 0.0126  # Approach 2 avg, from the Phase 4 training run

# Reference voltage sampling interval in the NASA RW dataset, used to compute dv/dt
# from a live voltage_history list (the API doesn't receive per-sample timestamps).
SAMPLE_INTERVAL_SECONDS = 10.0
RUL_REPLACE_SOH = 0.85
RUL_HEALTHY_SOH = 0.90

# Phase 7: lightweight self-attention over the voltage history window (Section 5 upgrade).
ATTENTION_D_MODEL = 32
ATTENTION_NHEAD = 4
ATTENTION_NUM_LAYERS = 2
ATTENTION_DIM_FEEDFORWARD = 64
ATTENTION_DROPOUT = 0.1

# Phase 7: gradient-boosted tree baseline on the existing engineered tabular features.
LIGHTGBM_N_ESTIMATORS = 200
LIGHTGBM_LEARNING_RATE = 0.05
LIGHTGBM_NUM_LEAVES = 31
