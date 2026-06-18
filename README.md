# Battery Remaining Useful Life Predictor

> Predicts Remaining Useful Life (RUL) and State of Health (SOH) for Li-ion batteries
> using deep learning and gradient-boosted trees.

![Python 3.11](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)
![Plotly Dash](https://img.shields.io/badge/Plotly%20Dash-2.15-3F4F75)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Why This Matters

Remaining Useful Life (RUL) is how much usable life a battery has left before it
needs replacing. Knowing it matters wherever cells can't be swapped out casually,
like battery management system, fleet telemetry, satellite and launch vehicle
power systems, and EV powertrains. All of which depend on accurate health estimates.

This project reimplements my SRA 2023 paper's approach to predicting battery
State of Health (SOH), then tests whether more sophisticated models actually improve
on it. See Results below.

## Results

Cross-battery generalization (Approach 2: train on RW9, predict RW10/RW11/RW12).
Average RMSE across the three held-out batteries:

| Model                 | Avg RMSE (Approach 2) |
|------------------------|------------------------|
| **Paper DNN (ours)**   | **0.66%**              |
| LightGBM (ours)        | 0.68%                  |
| Attention (ours)       | 0.80%                  |
| LSTM (ours)            | 0.81%                  |
| Upgraded DNN (ours)    | 1.26%                  |
| Paper DNN (original)   | 1.49%                  |
| BLS-RVM (original)     | 1.55%                  |
| RNN + LSTM (original)  | 1.61%                  |

All six models beat every published baseline. The interesting result is within
our own models where the exact paper-replica DNN, with 2 hidden layers and no batch norm,
dropout, or attention, beats every upgrade we tried, including a Transformer encoder.
LightGBM, using the same per-row engineered features (rolling mean/std, dv/dt), comes
close second. This suggests the engineered features already carry the temporal
signal that matters, and the bottleneck is the data/feature relationship, not model
capacity. See `CHECKPOINTS.md` Phase 7 for the full investigation.

## Quick Start

```bash
git clone https://github.com/aneeshg5/battery-rul-predictor.git
cd battery-rul-predictor
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
brew install libomp && bash scripts/fix_macos_libomp.sh  # macOS only
python scripts/download_data.py
python scripts/train.py --model paper_dnn --approach 2   # repeat for other --model values
python scripts/precompute_predictions.py
python scripts/serve.py                                  # http://localhost:8050
```

## Dashboard

![Dashboard screenshot](docs/images/dashboard_screenshot.png)

Select a battery, model, and approach to see live voltage traces, an SOH gauge,
SOH-over-time, training curves, and a full model comparison table.

## Architecture and Project Structure

```
src/battery_rul/
├── data/         NASA RW9-RW12 (.mat) ──► preprocess.py ──► data/processed/*.parquet
├── models/       paper_dnn, upgraded_dnn, lstm, attention, lightgbm
├── training/     Trainer (MLflow + Optuna) for the torch models, tree_trainer.py for LightGBM
├── evaluation/   metrics.py + precompute_predictions.py ──► data/processed/predictions/*.parquet
├── inference/    Predictor + FastAPI app (uvicorn battery_rul.inference.api:app)
└── dashboard/    Plotly Dash app (scripts/serve.py)

notebooks/        EDA and model-comparison notebooks
scripts/          CLI entry points (download, train, tune, serve)
tests/            pytest suite
CHECKPOINTS.md    full phase-by-phase engineering log
```

See [`docs/architecture.md`](docs/architecture.md) for the data pipeline, model and
training design decisions, and API rationale in detail.
