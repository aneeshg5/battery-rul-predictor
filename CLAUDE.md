# CLAUDE.md — Battery RUL Prediction: Portfolio Project Orchestration
> **Read this entire file before taking any action.**
> This is the single source of truth for the project. All phases, decisions, and
> constraints flow from here. Update `CHECKPOINTS.md` after every phase completes.
---
## 0. BOOTSTRAP — Run This First (One-Time Setup)
Claude Code is launched from the user's **Desktop folder**. The very first thing to do
is set up the repository so all subsequent work happens inside it.
```bash
# 0.1 — Move to Desktop and create the project folder
cd ~/Desktop
mkdir battery-rul-predictor
cd battery-rul-predictor
# 0.2 — Copy the paper transcription into the repo
cp ~/Desktop/predicting_rul_lithium_ion_batteries.md ./docs/paper_transcription.md
# If the file is elsewhere, adjust the path above. Create docs/ if needed: mkdir -p docs
# 0.3 — Copy this CLAUDE.md into the repo root
cp ~/Desktop/CLAUDE.md ./CLAUDE.md
# 0.4 — Initialize git
git init
git branch -M main
# 0.5 — Create initial commit scaffolding
touch CHECKPOINTS.md README.md .gitignore
git add .
git commit -m "chore: initial repo scaffold"
# 0.6 — (Optional) Link to GitHub remote if desired
# git remote add origin https://github.com/<your-username>/battery-rul-predictor.git
# git push -u origin main
```
After bootstrap, **all work happens inside `~/Desktop/battery-rul-predictor/`**.
---
## 1. PROJECT OVERVIEW
### What This Is
A modern, recruiter-ready reimplementation and significant upgrade of a published
research paper: *"Predicting the Remaining Useful Life of Lithium-Ion Batteries
Using Machine Learning Techniques"* (Chen, Ganti, Matsumura — SRA 2023).
The original paper built a Deep Neural Network (DNN) to predict battery State of
Health (SOH) using the NASA Randomized Battery Usage 1: Random Walk dataset. This
repo rebuilds that work from scratch with production-quality engineering, adds modern
ML upgrades, and wraps everything in a clean interactive dashboard.
### Why It Matters to Employers
Battery health prediction is mission-critical at:
- **Tesla** — Battery Management Systems (BMS), fleet health monitoring
- **SpaceX** — Satellite and Starship power systems, launch vehicle battery packs
- **Rivian, Lucid, BMW, Waymo** — EV powertrain reliability
- **NASA / aerospace** — Long-duration mission power management
This project demonstrates:
- Real ML engineering (not toy datasets)
- Production code practices (modular, tested, typed, documented)
- End-to-end pipeline ownership (data → model → serving → dashboard)
- Domain knowledge in battery electrochemistry + embedded systems awareness
- The ability to take academic research and turn it into deployable software
---
## 2. TECH STACK
| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Standard for ML |
| Data Processing | pandas, numpy | Fast tabular ops |
| ML Framework | PyTorch | Industry standard; shows deep learning fluency |
| Experiment Tracking | MLflow (local) | Reproducible runs; no account needed |
| Hyperparameter Tuning | Optuna | Modern Bayesian search; replaces paper's trial-and-error |
| Visualization | Plotly + Dash | Interactive dashboard; far beyond static matplotlib |
| API Layer | FastAPI | Production REST inference endpoint |
| Testing | pytest | Full unit + integration test suite |
| Code Quality | ruff, black, mypy | Typed, linted, formatted |
| Dependency Mgmt | uv (or pip + requirements.txt) | Fast, reproducible installs |
| CI | GitHub Actions | Lint + test on push |
| Packaging | pyproject.toml | Modern Python packaging |
---
## 3. REPOSITORY STRUCTURE
Create this exact layout during Phase 1:
```
battery-rul-predictor/
├── CLAUDE.md                    ← this file
├── CHECKPOINTS.md               ← progress log (update after every phase)
├── README.md                    ← polished project README (built in Phase 5)
├── pyproject.toml               ← project metadata + deps
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml               ← lint + test CI
├── docs/
│   ├── paper_transcription.md   ← the original paper text (moved here in bootstrap)
│   └── architecture.md          ← system design notes
├── data/
│   ├── raw/                     ← NASA dataset files land here (gitignored)
│   └── processed/               ← cleaned, feature-engineered parquet files
├── notebooks/
│   ├── 01_eda.ipynb             ← exploratory data analysis
│   └── 02_model_comparison.ipynb← BLS-RVM vs RNN vs DNN comparison
├── src/
│   └── battery_rul/
│       ├── __init__.py
│       ├── config.py            ← central config (paths, hyperparams, constants)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download.py      ← NASA dataset downloader
│       │   ├── preprocess.py    ← cleaning, SOH calculation, feature engineering
│       │   └── dataset.py       ← PyTorch Dataset + DataLoader
│       ├── models/
│       │   ├── __init__.py
│       │   ├── dnn.py           ← upgraded DNN (paper's core model)
│       │   ├── lstm.py          ← LSTM baseline (paper comparison model)
│       │   └── ensemble.py      ← optional: ensemble wrapper
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py       ← generic training loop with MLflow logging
│       │   └── tuning.py        ← Optuna hyperparameter search
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py       ← RMSE, MAE, R², overestimation/underestimation ratio
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── predictor.py     ← load model + run inference
│       │   └── api.py           ← FastAPI app
│       └── dashboard/
│           ├── __init__.py
│           └── app.py           ← Plotly Dash interactive dashboard
├── tests/
│   ├── test_preprocess.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_api.py
├── scripts/
│   ├── download_data.py         ← CLI: python scripts/download_data.py
│   ├── train.py                 ← CLI: python scripts/train.py --config ...
│   ├── tune.py                  ← CLI: python scripts/tune.py
│   └── serve.py                 ← CLI: python scripts/serve.py
└── mlruns/                      ← MLflow tracking (gitignored)
```
---
## 4. DOMAIN KNOWLEDGE (Read Before Writing Any ML Code)
### The Paper's Core Contribution
The paper uses **NASA's Randomized Battery Usage 1: Random Walk** dataset with 4
18650 Li-ion batteries (RW9, RW10, RW11, RW12). Batteries were subjected to random
charging/discharging currents from -4.5A to +4.5A in 5-minute loading periods. After
every 1500 periods (~5 days), reference charge/discharge cycles established SOH
benchmarks.
### SOH Formula (from paper)
The paper derived SOH from voltage using:
```
SOH = (V_i - V_f) / (V_0 - V_f)
```
Where:
- `V_i` = instantaneous voltage
- `V_f` = final voltage (when battery reaches 80% SOH → 0.8 * V_i)
- `V_0` = initial voltage
The 80% threshold comes from Tektronix Li-ion battery maintenance guidelines.
### Dataset Features Used as Model Inputs
From correlation matrix analysis, the paper identified these as most predictive:
1. Voltage history
2. Current
3. Temperature
4. Absolute time
5. Step type (charging / discharging / resting)
### Paper's DNN Architecture
- Input: 5 features
- Hidden Layer 1: 15 nodes, ReLU activation
- Hidden Layer 2: 10 nodes, Sigmoid activation
- Output: 1 node (predicted voltage → converted to SOH), linear activation
- Optimizer: Adam
- Loss: RMSE
- Train/test split: 80/20
- Trained for 6 epochs
### Paper's Results to Beat
| Model | Avg RMSE (Approach 2) |
|---|---|
| BLS-RVM | 1.55% |
| RNN + LSTM | 1.61% |
| **Paper DNN** | **1.49%** |
Our upgraded DNN should aim for **< 1.3% RMSE** through better architecture search
and training practices.
### Two Approaches from the Paper
- **Approach 1**: Train on part of battery X → predict rest of battery X (same battery)
- **Approach 2**: Train on all of battery RW9 → predict RW10, RW11, RW12 (cross-battery generalization)
Approach 2 is the more practically significant result — replicate and improve both.
---
## 5. UPGRADES OVER THE PAPER
These are the concrete improvements this repo makes over the original SRA 2023 paper:
| Upgrade | Details |
|---|---|
| **Framework** | PyTorch replaces unspecified framework; full GPU support |
| **Architecture search** | Optuna replaces manual trial-and-error for layer depth/width |
| **Residual connections** | Optional skip connections in DNN for deeper networks |
| **Attention mechanism** | Lightweight self-attention over voltage history window |
| **Batch normalization** | Added after hidden layers for training stability |
| **Early stopping** | Prevents overfitting beyond what 6 fixed epochs caught |
| **Richer features** | Rolling mean/std of voltage, delta-voltage (dV/dt), cycle count |
| **Experiment tracking** | Every run logged to MLflow with params, metrics, artifacts |
| **Systematic evaluation** | Overestimation vs underestimation ratio tracked (safety metric) |
| **REST API** | FastAPI endpoint for real-time SOH inference |
| **Interactive dashboard** | Plotly Dash: live voltage trace, SOH gauge, model comparison table |
| **CI pipeline** | GitHub Actions runs lint + tests on every push |
| **Full test suite** | pytest covering data pipeline, models, metrics, API |
---
## 6. DEVELOPMENT PHASES
Execute phases in order. Do not start a phase until the previous one is complete and
logged in CHECKPOINTS.md.
---
### PHASE 1 — Project Scaffold & Environment
**Goal:** Working repo structure with deps installed and CI passing.
Tasks:
1. Create all directories from Section 3 (`mkdir -p` for each)
2. Write `pyproject.toml` with all dependencies listed in Section 2
3. Write `.gitignore` (Python standard + data/ + mlruns/ + __pycache__ + .env)
4. Write `src/battery_rul/config.py` — centralized config with:
   - `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR` (use `pathlib.Path`)
   - `BATCH_SIZE = 512`
   - `LEARNING_RATE = 1e-3`
   - `HIDDEN_LAYERS = [15, 10]` (paper baseline)
   - `EPOCHS = 6` (paper baseline; Optuna will override)
   - `TRAIN_BATTERY = "RW9"`
   - `TEST_BATTERIES = ["RW10", "RW11", "RW12"]`
   - `SOH_THRESHOLD = 0.80`
   - `SEED = 42`
5. Write `.github/workflows/ci.yml`:
   - Trigger: push + PR to main
   - Steps: checkout → setup Python 3.11 → install deps → ruff check → black check → mypy → pytest
6. Write minimal `__init__.py` files in all `src/battery_rul/` subdirs
7. Run `git add . && git commit -m "feat: phase 1 — project scaffold"`
**Done when:** `pytest` collects 0 tests (not fails — just empty), `ruff` passes, `black` passes.
---
### PHASE 2 — Data Pipeline
**Goal:** Raw NASA data → clean, feature-engineered parquet files ready for training.
Tasks:
1. Write `src/battery_rul/data/download.py`:
   - Downloads the NASA RW dataset from the Socrata Open Data API
   - NASA dataset URL: `https://data.nasa.gov/Raw-Data/Randomized-Battery-Usage-1-Random-Walk/ugxu-9kjx`
   - Use the NASA Open Data Portal export endpoint or direct download
   - Save raw CSVs to `data/raw/RW{9,10,11,12}.csv`
   - Add checksum verification so re-runs skip re-download
   - CLI: `python scripts/download_data.py`
2. Write `src/battery_rul/data/preprocess.py`:
   - Load raw CSVs; parse columns: `Relative_time`, `Absolute_time`, `Voltage`, `Current`, `Temperature`, `Comment`, `Type`, `Date`
   - Parse `Type` / `Comment` field to extract step type: `{charge, discharge, rest}` → encode as `{1, -1, 0}`
   - **Implement SOH calculation** exactly as in the paper:
     ```python
     def compute_soh(voltage: pd.Series, threshold: float = 0.80) -> pd.Series:
         V0 = voltage.iloc[0]
         Vf = threshold * V0
         return (voltage - Vf) / (V0 - Vf)
     ```
   - **Add engineered features** (upgrades over paper):
     - `voltage_rolling_mean_10` — 10-sample rolling mean of voltage
     - `voltage_rolling_std_10` — 10-sample rolling std of voltage
     - `dv_dt` — first-order voltage derivative (delta-voltage)
     - `cycle_count` — cumulative reference cycle count
   - Normalize all features to [0, 1] using min-max scaling; fit scaler on RW9 only, apply to all
   - Save processed data as parquet: `data/processed/RW{9,10,11,12}.parquet`
   - Save scaler state as `data/processed/scaler.pkl`
3. Write `src/battery_rul/data/dataset.py`:
   - `BatteryDataset(torch.utils.data.Dataset)`:
     - Takes a parquet path and window size (default 50 timesteps)
     - `__getitem__` returns `(features_window, soh_target)` tensors
   - `get_dataloaders(train_battery, test_batteries, batch_size, window_size)`:
     - Returns train DataLoader from RW9 and dict of test DataLoaders
4. Write `tests/test_preprocess.py`:
   - Test SOH stays in [0, 1] range
   - Test feature columns all present
   - Test scaler inverse transform round-trips correctly
   - Test dataset `__len__` and `__getitem__` shapes
5. Write `scripts/download_data.py` (thin CLI wrapper around `download.py`)
6. Run `git add . && git commit -m "feat: phase 2 — data pipeline"`
**Done when:** `python scripts/download_data.py` produces 4 CSVs in `data/raw/`, preprocessing produces 4 parquets in `data/processed/`, and `pytest tests/test_preprocess.py` passes.
---
### PHASE 3 — Models
**Goal:** Three model implementations — paper's DNN (baseline), upgraded DNN, and LSTM comparison.
Tasks:
1. Write `src/battery_rul/models/dnn.py`:
   **PaperDNN** (exact paper replica):
   ```
   Linear(input_dim, 15) → ReLU
   Linear(15, 10) → Sigmoid
   Linear(10, 1)
   ```
   **UpgradedDNN** (configurable, production-ready):
   - Accepts `layer_sizes: list[int]` from Optuna / config
   - Each hidden layer: `Linear → BatchNorm1d → activation → Dropout(p)`
   - Final layer: `Linear → output` (no activation for regression)
   - Support for residual connections when consecutive layers have matching dims
   - Forward pass fully typed with `torch.Tensor` annotations
2. Write `src/battery_rul/models/lstm.py`:
   **BatteryLSTM** (for comparison, mirrors paper's RNN+LSTM baseline):
   - `LSTM(input_size, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)`
   - Fully connected output head: `Linear(64, 1)`
   - Forward accepts `(batch, seq_len, features)` tensors
3. Write `tests/test_models.py`:
   - Test PaperDNN forward pass with batch of shape `(32, 5)` → output `(32, 1)`
   - Test UpgradedDNN forward pass with various layer configs
   - Test LSTM forward pass with `(32, 50, 9)` → output `(32, 1)`
   - Test parameter counts are reasonable
4. Run `git add . && git commit -m "feat: phase 3 — model implementations"`
**Done when:** `pytest tests/test_models.py` passes with all shapes correct.
---
### PHASE 4 — Training, Tuning & Evaluation
**Goal:** Full training loop with MLflow tracking, Optuna tuning, and rigorous evaluation.
Tasks:
1. Write `src/battery_rul/training/trainer.py`:
   - `Trainer` class:
     - `__init__(model, optimizer, loss_fn, device, mlflow_run_name)`
     - `train_epoch(dataloader) -> float` — returns train RMSE
     - `eval_epoch(dataloader) -> dict` — returns `{rmse, mae, r2}`
     - `fit(train_dl, val_dl, epochs, early_stopping_patience=5)` — main loop
     - All metrics logged to MLflow per epoch
     - Best model checkpoint saved to `mlruns/` artifact store
   - Loss function: RMSE as in paper (torch custom or via `torch.nn.MSELoss` + sqrt)
   - Early stopping: restore best weights if val loss increases for N epochs
2. Write `src/battery_rul/evaluation/metrics.py`:
   - `rmse(y_true, y_pred) -> float`
   - `mae(y_true, y_pred) -> float`
   - `r2_score(y_true, y_pred) -> float`
   - `overestimation_rmse(y_true, y_pred) -> float` — RMSE on overestimated points only
   - `underestimation_rmse(y_true, y_pred) -> float` — RMSE on underestimated points only
   - `safety_ratio(y_true, y_pred) -> float` — fraction of predictions that underestimate (> 1.0 is safer)
   - All functions accept numpy arrays
3. Write `src/battery_rul/training/tuning.py`:
   - Optuna study: `create_study(direction="minimize")`
   - Search space:
     ```python
     n_layers = trial.suggest_int("n_layers", 2, 5)
     layer_sizes = [trial.suggest_int(f"n_units_l{i}", 8, 128) for i in range(n_layers)]
     lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
     dropout = trial.suggest_float("dropout", 0.0, 0.4)
     batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
     ```
   - Objective: val RMSE after 10 epochs on RW9
   - `n_trials=50` (reasonable for a single machine)
   - Log best params to MLflow and save to `data/processed/best_params.json`
4. Write `scripts/train.py`:
   ```
   python scripts/train.py --model paper_dnn --approach 1
   python scripts/train.py --model paper_dnn --approach 2
   python scripts/train.py --model upgraded_dnn --approach 2
   python scripts/train.py --model lstm --approach 2
   ```
   - `--approach 1`: train on first 80% of RW9, test on last 20% of RW9
   - `--approach 2`: train on all RW9, test on RW10, RW11, RW12
   - Outputs a table of per-battery RMSE + average RMSE
5. Write `scripts/tune.py`:
   ```
   python scripts/tune.py --n-trials 50
   ```
   - Runs Optuna, prints best params, saves to JSON
6. Write `tests/test_metrics.py`:
   - Test RMSE on known arrays
   - Test overestimation/underestimation split logic
   - Test safety_ratio > 1.0 when more underestimations exist
7. Run all four training configs; **record results in CHECKPOINTS.md**
8. Run `git add . && git commit -m "feat: phase 4 — training, tuning, evaluation"`
**Done when:** All 4 training scripts run without error; results table printed to console; MLflow UI (`mlflow ui`) shows all runs with logged metrics.
---
### PHASE 5 — Interactive Dashboard
**Goal:** A polished Plotly Dash app that a recruiter can run with one command.
Tasks:
1. Write `src/battery_rul/dashboard/app.py`:
   Layout:
   ```
   Header: "Battery RUL Predictor — Live SOH Dashboard"
   [Sidebar]                    [Main Panel]
   - Battery selector           - Voltage vs Time trace (actual + predicted overlay)
     (RW9, RW10, RW11, RW12)   - SOH gauge (0–100%, red zone below 80%)
   - Model selector             - SOH vs Time line chart
     (Paper DNN / Upgraded DNN  - RMSE per epoch (training curve)
      / LSTM)                   - Model comparison table
   - Approach toggle (1 / 2)      (BLS-RVM | LSTM | Paper DNN | Upgraded DNN)
   ```
   Callbacks:
   - Selecting a battery + model → update all charts from precomputed predictions
   - SOH gauge color: green (>90%), yellow (80–90%), red (<80%)
   - Model comparison table highlighted: best RMSE row bolded/green
2. Precompute predictions for all model × battery combinations during training
   and save to `data/processed/predictions/` as parquet files so the dashboard
   loads instantly (no model inference at runtime).
3. Write `scripts/serve.py`:
   ```
   python scripts/serve.py
   ```
   Opens at `http://localhost:8050`
4. Add dashboard screenshot instructions to README (for GitHub display).
5. Run `git add . && git commit -m "feat: phase 5 — interactive dashboard"`
**Done when:** `python scripts/serve.py` opens a working, interactive dashboard.
---
### PHASE 6 — FastAPI Inference Endpoint
**Goal:** REST API for real-time SOH prediction — mirrors production BMS integration.
Tasks:
1. Write `src/battery_rul/inference/predictor.py`:
   - `Predictor` class:
     - `__init__(model_path, scaler_path, device)` — loads weights + scaler
     - `predict(voltage_history: list[float], current: float, temperature: float, step_type: str) -> dict`
     - Returns `{"soh": float, "rul_estimate": str, "confidence": str}`
     - RUL estimate: "Replace soon" if SOH < 85%, "Healthy" if SOH > 90%
2. Write `src/battery_rul/inference/api.py`:
   ```python
   POST /predict
   Body: {
     "voltage_history": [4.1, 4.05, ...],  # last N voltage readings
     "current": -2.3,
     "temperature": 25.4,
     "step_type": "discharge"
   }
   Response: {
     "soh": 0.873,
     "soh_percent": 87.3,
     "rul_estimate": "Healthy",
     "model_version": "upgraded_dnn_v1"
   }
   GET /health       → {"status": "ok", "model_loaded": true}
   GET /model-info   → {architecture, rmse, training_battery, test_batteries}
   ```
3. Write `tests/test_api.py`:
   - Test `/health` returns 200
   - Test `/predict` with valid input returns correct schema
   - Test `/predict` with invalid input returns 422
4. Run `git add . && git commit -m "feat: phase 6 — FastAPI inference endpoint"`
**Done when:** `uvicorn battery_rul.inference.api:app --reload` starts; `/predict` returns valid JSON.
---
### PHASE 7 — EDA Notebooks
**Goal:** Two clean, well-documented notebooks that tell the data story.
Tasks:
1. Write `notebooks/01_eda.ipynb`:
   - Load all 4 battery raw CSVs
   - Plot: voltage over time for each battery (4 subplots)
   - Plot: temperature over time
   - Plot: current distribution (charge vs discharge cycles)
   - Plot: computed SOH over time → show degradation curve
   - Plot: correlation matrix heatmap (reproduce paper's Fig. 7 but styled cleanly)
   - Markdown cells explaining each finding
2. Write `notebooks/02_model_comparison.ipynb`:
   - Load precomputed predictions from `data/processed/predictions/`
   - Plot: actual vs predicted voltage for each battery (Approach 2)
   - Plot: RMSE training curves for all models
   - Table: final RMSE comparison (paper's Table reproduced + upgraded DNN row added)
   - Plot: overestimation vs underestimation distribution
   - Markdown: interpret results, explain why upgraded DNN outperforms
3. Run `git add . && git commit -m "feat: phase 7 — EDA and comparison notebooks"`
**Done when:** Both notebooks run top-to-bottom without errors.
---
### PHASE 8 — README & Final Polish
**Goal:** A README so good that a recruiter or engineer understands the project in 60 seconds.
Tasks:
1. Write `README.md` with this structure:
```markdown
# Battery RUL Predictor
> Predicting State of Health and Remaining Useful Life of Li-ion batteries
> using Deep Neural Networks — based on published SRA 2023 research.
[badges: Python 3.11 | PyTorch | FastAPI | Plotly Dash | MIT License]
## Why This Matters
[2-sentence context: Tesla BMS, SpaceX power systems, EV reliability]
## Results
| Model         | Avg RMSE (Approach 2) |
|---------------|-----------------------|
| BLS-RVM       | 1.55%                 |
| RNN + LSTM    | 1.61%                 |
| Paper DNN     | 1.49%                 |
| Upgraded DNN  | X.XX%  ← fill in     |
## Architecture
[diagram or ASCII art of DNN pipeline]
## Quick Start
[5-line install + run block]
## Dashboard
[screenshot of Dash app]
## Project Structure
[abbreviated tree]
## Paper Reference
Chen, Ganti, Matsumura. "Predicting the Remaining Useful Life of
Lithium-Ion Batteries Using Machine Learning Techniques." SRA 2023.
## Author
Aneesh Ganti — [LinkedIn] [GitHub]
```
2. Write `docs/architecture.md` — detailed system design document covering:
   - Data pipeline flow
   - Model architecture decisions
   - Training methodology
   - API design rationale
3. Final `git add . && git commit -m "docs: phase 8 — README and polish"`
4. Tag the release: `git tag -a v1.0.0 -m "Portfolio release v1.0.0"`
**Done when:** README renders cleanly on GitHub; project is shareable.
---
## 7. CHECKPOINTS.md PROTOCOL
After each phase completes, append a block to `CHECKPOINTS.md` in this format:
```markdown
## Phase N — [Name] — COMPLETE
**Date:** YYYY-MM-DD
**Commit:** <git hash>
### What Was Built
- bullet list of files created/modified
### Results / Metrics
- any numbers worth recording (RMSE, test counts, etc.)
### Issues Encountered
- anything non-trivial that was debugged
### Next Phase
Phase N+1 — [Name]
```
Never delete old checkpoint entries. They form the project's engineering log.
---
## 8. CODING STANDARDS
All code must follow these rules — Claude Code enforces them before committing:
- **Type hints everywhere** — all function signatures, all class attributes
- **Docstrings on all public functions/classes** — Google style
- **No magic numbers** — all constants live in `config.py`
- **No hardcoded paths** — use `config.DATA_DIR / "filename"` patterns
- **Logging, not print** — use Python's `logging` module; `print()` only in CLI scripts
- **Error handling** — never bare `except:`; always specific exception types
- **Line length** ≤ 100 chars (ruff enforced)
- **Test coverage** — every new module gets at least one test file
- **Commit message format**: `type: description` where type ∈ {feat, fix, docs, test, chore, refactor}
---
## 9. DATA NOTES
- The NASA dataset is **public domain** and freely downloadable
- Raw data files are **gitignored** (too large for GitHub)
- `scripts/download_data.py` makes the project fully reproducible from scratch
- If the Socrata endpoint is unavailable, fallback: manual download from
  `https://data.nasa.gov/Raw-Data/Randomized-Battery-Usage-1-Random-Walk/ugxu-9kjx`
  → Export → CSV; place files in `data/raw/` named `RW9.csv`, `RW10.csv`, etc.
- Each CSV has columns: `Relative_time`, `Absolute_time`, `Voltage`, `Current`,
  `Temperature`, `Comment`, `Type`, `Date_Time`
- Batteries: RW9, RW10, RW11, RW12 — 18650 Li-ion cells
---
## 10. ENVIRONMENT SETUP
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows
# Install all dependencies
uv pip install -e ".[dev]"
# Verify install
python -c "import torch; import dash; import fastapi; print('All good')"
```
---
## 11. COMMON COMMANDS
```bash
# Download dataset
python scripts/download_data.py
# Train all models (run in order)
python scripts/train.py --model paper_dnn --approach 1
python scripts/train.py --model paper_dnn --approach 2
python scripts/train.py --model upgraded_dnn --approach 2
python scripts/train.py --model lstm --approach 2
# Hyperparameter tuning
python scripts/tune.py --n-trials 50
# Launch dashboard
python scripts/serve.py
# Launch API
uvicorn battery_rul.inference.api:app --reload --port 8000
# Open MLflow UI
mlflow ui
# Run tests
pytest tests/ -v
# Lint + format
ruff check src/ tests/
black src/ tests/
# Type check
mypy src/
```
---
## 12. WHAT CLAUDE CODE SHOULD DO NEXT
If you are reading this as Claude Code starting fresh:
1. **Run Section 0 bootstrap commands** to create the folder, copy files, and init git
2. **Start Phase 1** — scaffold the directory structure and pyproject.toml
3. **Update CHECKPOINTS.md** when Phase 1 is done
4. **Proceed phase by phase** — never skip ahead
5. **If you encounter an ambiguity**, make the simplest reasonable choice, log it in CHECKPOINTS.md under "Issues Encountered", and continue — do not stop and ask unless the ambiguity is blocking
6. **The paper transcription** is at `docs/paper_transcription.md` — reference it for exact formulas, dataset details, and results to replicate/beat
The final product should be something Aneesh can link directly on his resume under Projects
and that any ML engineer at Tesla or SpaceX could clone, run in 10 minutes, and immediately
understand what it does, why it matters, and that it was built with production-quality discipline.
