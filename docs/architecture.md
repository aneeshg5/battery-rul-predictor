# Architecture

## Data Pipeline

1. **`data/download.py`** pulls the four NASA RW9-RW12 `.mat` files (the dataset ships
   as MATLAB structs, not flat CSVs) into `data/raw/`, with checksum verification so
   re-runs skip re-downloading.
2. **`data/preprocess.py`** parses each battery's step records, classifies each step
   as charge, discharge, or rest, and computes SOH via `(V_i - V_f) / (V_0 - V_f)`
   (paper formula, `V_f = SOH_THRESHOLD * V_0`). It also adds engineered features like
   `voltage_rolling_mean_10`, `voltage_rolling_std_10`, `dv_dt`, and `cycle_count`. All
   features are min-max scaled to `[0, 1]` with the scaler fit on RW9 only and applied
   to RW10-RW12, so test batteries never leak into the fit. Unscaled `voltage_raw`,
   `absolute_time_raw`, and `cycle_count_raw` are preserved alongside the scaled
   columns for plotting (dashboard, notebooks) where physical units matter.
3. **`data/dataset.py`**'s `BatteryDataset` windows a processed parquet into
   `(WINDOW_SIZE, n_features)` sequences with an SOH target. `get_dataloaders` returns
   a train `DataLoader` for RW9 and a dict of test `DataLoader`s for RW10-RW12.

## Model Architecture Decisions

Four PyTorch models share `training/trainer.py`'s `Trainer` via two input modes.

- **`paper_dnn`** is an exact paper replica
  (`Linear(5,15)→ReLU→Linear(15,10)→Sigmoid→Linear(10,1)`), taking a single timestep's
  features (`input_mode="tabular"`).
- **`upgraded_dnn`** is a configurable `Linear→BatchNorm1d→activation→Dropout` stack
  with optional residual connections, also tabular.
- **`lstm`** and **`attention`** both consume the full windowed sequence
  (`input_mode="sequence"`). `BatteryLSTM` is a 2-layer LSTM with a linear head, and
  `BatteryAttention` is a small `nn.TransformerEncoder` (input projection, then a
  learned positional embedding, 1-2 encoder layers, and a head on the final timestep).

**`lightgbm`** (`models/trees.py`) is not an `nn.Module`. Gradient-boosted trees don't
fit a PyTorch training loop, so it has its own trainer (`training/tree_trainer.py`)
that fits directly on flat per-row `FEATURE_COLUMNS` (no windowing, since the
rolling/derivative features already encode short-term history per row) and logs to
MLflow via `mlflow.lightgbm.log_model`.

`lightgbm` and `attention` were added specifically to test why `upgraded_dnn`
underperformed the simpler `paper_dnn`. One model has no access to sequence order at
all, and the other was built specifically to learn temporal structure itself rather
than relying on hand-engineered rolling windows. Neither beat `paper_dnn`, which is the
central finding of this project. For this dataset, the engineered per-row features
already carry the predictive signal, so additional model capacity or architectural
sophistication doesn't help.

## Training Methodology

- `Trainer.fit` runs a standard epoch loop with RMSE loss (`rmse_loss`, which is
  `sqrt(MSELoss)`, matching the paper), early stopping on validation RMSE
  (`early_stopping_patience`), and per-epoch metric logging to MLflow.
- Two evaluation approaches run for every model. **Approach 1** is a train/test split
  within a single battery (RW9 80/20). **Approach 2** trains on all of RW9 and tests on
  RW10/RW11/RW12, the more practically significant cross-battery generalization
  result, and the one reported in the README/dashboard comparison table.
- `training/tuning.py` runs an Optuna study (`n_layers`, per-layer width, `lr`,
  `dropout`, `batch_size`) minimizing validation RMSE after 10 epochs on RW9, with
  best params saved to `data/processed/best_params.json`.
- `evaluation/metrics.py` adds a safety-oriented metric beyond plain RMSE.
  `overestimation_rmse` and `underestimation_rmse` split error by direction, and
  `safety_ratio` tracks what fraction of predictions underestimate SOH.
  Underestimating (prematurely flagging a healthy battery) is the safer failure mode
  for a BMS than overestimating (missing a battery that needs replacement).

## API Design Rationale

`inference/predictor.py`'s `Predictor` loads a trained model and scaler once at
process startup (via FastAPI's `lifespan` context, not per-request) and exposes a
single `predict(voltage_history, current, temperature, step_type)` method that mirrors
the shape of a real BMS read, a rolling voltage buffer plus the latest
current/temperature/step-type sample, not a full historical dataset. `dv_dt` is
recomputed from consecutive `voltage_history` samples using
`SAMPLE_INTERVAL_SECONDS`, since a live caller has no per-sample timestamps to derive
it from directly.

`POST /predict` returns both the raw `soh` and a `rul_estimate` string
(`"Replace soon"` below `RUL_REPLACE_SOH`, `"Healthy"` above `RUL_HEALTHY_SOH`) so a
downstream BMS dashboard can act on the response without re-deriving thresholds.
`GET /health` and `GET /model-info` exist for the same reason any production inference
service needs them. A load balancer or orchestrator needs a cheap liveness check, and
an on-call engineer needs to know which model version and training RMSE is actually
deployed without reading code.
