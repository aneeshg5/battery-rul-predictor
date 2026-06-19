## Phase 0 — Bootstrap — COMPLETE

**Date:** 2026-06-17

### What Was Built
- Initialized git repo, set default branch to `main`
- Moved paper transcription to `docs/paper_transcription.md`
- Added `.gitignore`, placeholder `README.md`

### Issues Encountered
- None

### Next Phase
Phase 1 — Project Scaffold & Environment

## Phase 1 — Project Scaffold & Environment — COMPLETE

**Date:** 2026-06-17

### What Was Built
- Full directory layout (`src/battery_rul/*`, `tests/`, `scripts/`, `notebooks/`, `data/`)
- `pyproject.toml` with runtime + dev dependencies, ruff/black/mypy/pytest config
- `src/battery_rul/config.py` with paths and baseline hyperparameters
- `.github/workflows/ci.yml` running ruff, black, mypy, pytest on push/PR to main
- `__init__.py` stubs for all `battery_rul` subpackages
- `.venv` created via `uv`, dependencies installed and verified importable

### Results / Metrics
- `pytest tests/` collects 0 tests (expected)
- `ruff check`, `black --check`, `mypy src/` all pass clean

### Issues Encountered
- None

### Next Phase
Phase 2 — Data Pipeline

## Phase 2 — Data Pipeline — COMPLETE

**Date:** 2026-06-17

### What Was Built
- `src/battery_rul/data/download.py` — downloads and extracts RW9-RW12 from the actual
  NASA PCoE archive, with SHA-256 checksums written to skip re-download on rerun
- `src/battery_rul/data/preprocess.py` — flattens each battery's nested step structure
  into a long DataFrame, computes SOH, adds rolling voltage stats / dv_dt / cycle_count,
  fits a MinMaxScaler on RW9 and applies it to all four batteries, writes parquet
- `src/battery_rul/data/dataset.py` — `BatteryDataset` (windowed feature/SOH pairs) and
  `get_dataloaders()`
- `scripts/download_data.py` CLI wrapper
- `tests/test_preprocess.py` — SOH formula, feature presence, scaler round-trip, dataset shapes

### Results / Metrics
- Full pipeline run on real data: RW9 8,532,073 / RW10 8,596,025 / RW11 8,664,510 /
  RW12 ~8.7M rows; train DataLoader yields 16,665 batches of shape (512, 50, 9)
- `pytest`, `ruff`, `black`, `mypy` all pass clean

### Issues Encountered
- The original plan assumed the dataset was served as flat CSVs via a Socrata API at
  `data.nasa.gov/Raw-Data/.../ugxu-9kjx`. That URL 404s — data.nasa.gov runs CKAN, not
  Socrata, and the actual files are MATLAB `.mat` structs (one per battery, each a list
  of ~113k "step" records with nested per-step arrays) inside a single zip at
  `data.nasa.gov/docs/legacy/ames/1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip`.
  `download.py` and `preprocess.py` were written against the real format.
- Applying the paper's literal SOH formula (`V0` = first voltage sample of the whole
  series) to the full random-walk history produces SOH outside [0, 1] (observed range on
  RW9: -0.36 to 2.07), since later reference/pulsed-charge steps spike above the first
  sample's instantaneous voltage. This is a property of the formula on raw data, not a
  bug — the unit test instead verifies the formula's correctness on a monotonic decay
  curve (V0 -> 0.8*V0), which is what the formula is designed for. Worth keeping in mind
  for dashboard SOH gauge clamping and training stability in later phases.

### Next Phase
Phase 3 — Models

## Phase 3 — Models — COMPLETE

**Date:** 2026-06-17

### What Was Built
- `src/battery_rul/models/dnn.py` — `PaperDNN` (exact replica: Linear->ReLU->Linear->
  Sigmoid->Linear, sized from `config.HIDDEN_LAYERS`) and `UpgradedDNN` (configurable
  depth/width, Linear->BatchNorm1d->ReLU->Dropout per layer, residual add when a block's
  input/output dims match)
- `src/battery_rul/models/lstm.py` — `BatteryLSTM` (2-layer LSTM, hidden=64, dropout=0.2,
  Linear(64,1) head over the final timestep)
- `tests/test_models.py` — forward-pass shapes for all three models, residual-path shape
  check, parameter-count sanity bounds

### Results / Metrics
- `pytest` (9 tests total), `ruff`, `black`, `mypy` all pass clean

### Issues Encountered
- Initial `UpgradedDNN` used `nn.ModuleDict` blocks inside an `nn.ModuleList`; mypy
  couldn't infer dict-style indexing on a generic `nn.Module`. Refactored to four
  parallel `nn.ModuleList`s (linears/norms/dropouts) iterated with `zip(..., strict=True)`
  — cleaner and fully typed.

### Next Phase
Phase 4 — Training, Tuning & Evaluation

## Phase 4 — Training, Tuning & Evaluation — COMPLETE

**Date:** 2026-06-17

### What Was Built
- `src/battery_rul/training/trainer.py` — `Trainer` (train/eval epoch loops, RMSE loss,
  early stopping with best-state restore, MLflow logging per run)
- `src/battery_rul/evaluation/metrics.py` — `rmse`, `mae`, `r2_score`,
  `overestimation_rmse`, `underestimation_rmse`, `safety_ratio`
- `src/battery_rul/training/tuning.py` — Optuna `objective`/`run_study` over `UpgradedDNN`
  (n_layers, per-layer width, lr, dropout, batch_size)
- `scripts/train.py` — CLI for all 4 required configs (`paper_dnn`/`upgraded_dnn`/`lstm`
  x approach 1/2), with a `stride` parameter on `BatteryDataset` to subsample windows
- `scripts/tune.py` — CLI wrapper around `run_study`, writes `data/processed/best_params.json`
- `tests/test_metrics.py` — 6 tests covering all metric functions

### Results / Metrics
Final results (paper_dnn/upgraded_dnn on full data, lstm and Optuna on stride-subsampled
data per the compute-scope decision below):

| Model | Approach | RW10 | RW11 | RW12 | Average RMSE |
|---|---|---|---|---|---|
| paper_dnn | 1 (intra-battery, RW9 80/20) | — | — | — | 0.18% (RW9 holdout) |
| paper_dnn | 2 (cross-battery) | 0.12% | 0.14% | 1.71% | **0.66%** |
| upgraded_dnn | 2 | 0.81% | 0.75% | 2.22% | **1.26%** |
| lstm | 2 (stride=5) | 0.31% | 0.41% | 1.76% | **0.83%** |

All three Approach-2 models beat the paper's reported 1.49% (Paper DNN), 1.55% (BLS-RVM),
and 1.61% (RNN+LSTM) baselines.

Optuna search (15 trials, stride=10 subsampled RW9, 10 epochs/trial, patience=3):
best val RMSE 0.88% with `{n_layers: 2, n_units_l0: 77, n_units_l1: 80, lr: 0.00135,
dropout: 0.015, batch_size: 256}`, saved to `data/processed/best_params.json`.

### Issues Encountered
- **Compute-scope decision (user-approved):** a full-spec run of all 4 training configs
  plus a 50-trial Optuna search was estimated at ~13-14 hours on this machine. Chose
  the reduced-scope option instead: `paper_dnn` and `upgraded_dnn` train on the full dataset;
  `lstm` (approach 2) uses `stride=5` window subsampling; Optuna uses `stride=10` and
  `n_trials=15` instead of 50. This is a compute-driven deviation from the original
  Phase 4 plan, not a methodology change — `BatteryDataset` gained a `stride` parameter
  that only thins which windows are sampled, the windows themselves are unchanged.
- **Real bug found via first training pass, not assumed correct:** the initial Approach-2
  run (all 3 models) produced ~31-32% average RMSE across every model — same magnitude
  regardless of architecture, which pointed at the data pipeline rather than the models.
  Root cause: `compute_soh`'s `V0` was the first voltage sample of each battery's trace.
  Since a random-walk recording can start at an arbitrary point in the charge/discharge
  cycle, first-sample voltage varied widely across batteries (RW9=3.838V, RW10=4.196V,
  RW11=4.187V, RW12=3.957V), so the SOH target was a different affine function of voltage
  per battery — a model trained on RW9's mapping couldn't transfer to a different mapping.
  First fix attempt (`V0 = voltage.max()`) used each battery's true full-charge voltage
  instead (consistent ~4.65-4.81V across batteries) and dropped average RMSE to 0.66-4.94%
  range — but RW12 stayed an outlier (~13%) in every model. Investigated further: RW12's
  raw max (4.814V) was a single-sample sensor noise spike (next-highest value was 4.728V;
  99.9th percentile was 4.215V). Switched `V0` to `voltage.quantile(0.999)`, which recovers
  the same ~4.2V nominal full-charge voltage for all four batteries (the actual rated
  voltage for this 18650 cell) and is robust to single-sample noise. Re-ran preprocessing
  and all 4 training configs after each fix; final numbers above reflect the quantile-based
  `V0`. `tests/test_preprocess.py::test_soh_monotonic_decay_stays_in_unit_range` tolerances
  were loosened slightly (`abs=0.01`) since a quantile-based V0 can let the top ~0.1% of
  samples exceed SOH=1.0 by design.
- `mlflow.pytorch.log_model`'s newer default `serialization_format='pt2'` requires an
  `input_example` (traced-graph export); passed `serialization_format="pickle"` explicitly
  instead (acceptable security tradeoff for this local, non-served use case).

### Next Phase
Phase 5 — Interactive Dashboard

## Phase 5 — Interactive Dashboard — COMPLETE

**Date:** 2026-06-18

### What Was Built
- `src/battery_rul/data/preprocess.py` — `build_features()` now also keeps unscaled
  `absolute_time_raw`/`voltage_raw` columns (added after engineered features, before the
  MinMax scaler is fit/applied) so the dashboard can plot real seconds and volts instead
  of [0, 1]-scaled values; preprocessing re-run to regenerate all 4 parquets
- `src/battery_rul/evaluation/predictions.py` — `predict_battery()`, a windowed-inference
  helper that runs a trained model over a battery's processed parquet at a given stride
  and returns actual-vs-predicted SOH alongside the raw time/voltage columns
- `scripts/precompute_predictions.py` — loads each of the 4 Phase-4 models directly from
  MLflow (most recent run per run-name, via `MlflowClient.search_runs`) instead of
  retraining, runs `predict_battery` per battery at a stride chosen to cap each trace at
  ~5,000 points (browser-friendly), and pulls per-epoch `train_rmse`/`val_rmse` history
  via `get_metric_history` for the training-curve chart. Writes everything to
  `data/processed/predictions/*.parquet` (gitignored, same as other processed data)
- `src/battery_rul/dashboard/app.py` — Dash app: approach toggle (1/2) drives which
  models/batteries are selectable (Approach 1 only has `paper_dnn`/RW9; Approach 2 has
  all 3 models x RW10-12); voltage-vs-time trace, SOH gauge (red/yellow/green at the
  paper's 80/90% thresholds, clamped to [0, 100] for display), actual-vs-predicted SOH
  overlay, RMSE-per-epoch training curve, and a model comparison table (published
  baselines from `config.PAPER_BASELINE_RMSE` + our 3 models, best RMSE row bolded)
- `scripts/serve.py` — launches the dashboard at `http://localhost:8050`
- `src/battery_rul/dashboard/assets/style.css` — minimal sidebar/main-panel flex layout
- `docs/images/dashboard_screenshot.png` + a Dashboard section in `README.md`

### Results / Metrics
- RMSE computed from the precomputed (subsampled) prediction files matches the
  full-resolution Phase 4 numbers to within rounding (e.g. paper_dnn/RW12: 1.71% both
  ways), confirming the ~5,000-point subsample is representative for the table and charts
- `pytest` (15 tests), `ruff`, `black`, `mypy` all pass clean
- Manually exercised in a browser via Playwright: approach toggle correctly re-scopes the
  model/battery dropdowns, all 4 charts update on selection change, gauge color and
  comparison-table highlighting both verified visually

### Issues Encountered
- **Voltage-vs-predicted overlay isn't possible as literally specified**: the original
  Phase 5 plan described a "Voltage vs Time trace (actual + predicted overlay)," but
  every model in this project predicts SOH, not voltage — there is no predicted-voltage
  series to overlay. Adapted to the models' actual output: the voltage panel shows the
  real (unscaled) actual trace only, and the actual-vs-predicted overlay was moved to the
  SOH panel, which is the quantity the models actually predict.
- **SOH gauge clamping was needed, as flagged back in Phase 2**: the quantile-based SOH
  formula can produce values outside [0, 1] on noisy samples (confirmed live — RW12's
  raw predicted SOH hit -6.6% at one selected point). The gauge display is clamped to
  [0, 100]; the SOH-vs-time line chart is left unclamped so the chart still shows the true
  predicted value.
- Considered smoothing the gauge's "latest SOH" reading over the last 10 stride-subsampled
  points to reduce noise, but those points are spread across a wide time range at this
  stride (not actually "recent"), so averaging them would misrepresent the signal rather
  than clean it up. Kept the gauge as the single latest predicted point (matches what the
  model actually outputs at one instant), clamped for display only.
- Reused the already-trained Phase 4 MLflow models instead of retraining for prediction
  precomputation — `MlflowClient.search_runs` filtered by run name and sorted by start
  time picks up the most recent (post-SOH-fix) run automatically, so this works correctly
  even though each run name was trained 3 times across the Phase 4 bug-fix iterations.

### Next Phase
Phase 6 — FastAPI Inference Endpoint

## Phase 6 — FastAPI Inference Endpoint — COMPLETE

**Date:** 2026-06-18

### What Was Built
- `scripts/export_model.py` — exports the `upgraded_dnn_approach2` model from MLflow
  (most recent run by name) to a standalone `torch.save` artifact
  (`data/processed/model_upgraded_dnn_v1.pt`), decoupling the inference API from the
  MLflow tracking store at request time. Also computes mean `absolute_time`/`cycle_count`
  from RW9 and writes them to `data/processed/inference_defaults.json` for use as
  inference-time defaults for the two features a single live reading can't supply
- `src/battery_rul/data/preprocess.py` — added a `cycle_count_raw` preserved column
  (mirrors `absolute_time_raw`/`voltage_raw` from Phase 5) so `export_model.py` can read
  genuinely-unscaled cycle counts; preprocessing re-run to regenerate all 4 parquets
- `src/battery_rul/inference/predictor.py` — `Predictor` class: loads the exported model +
  `scaler.pkl` + inference defaults, reconstructs all 9 model features from a live
  `voltage_history`/`current`/`temperature`/`step_type` reading (rolling mean/std and dv/dt
  computed from the history list, `absolute_time`/`cycle_count` filled from defaults),
  scales, runs inference, and returns `{soh, rul_estimate, confidence}`. Confidence is
  "high" when the requested voltage falls inside the training data's observed min/max
  range, else "low"
- `src/battery_rul/inference/api.py` — FastAPI app with `GET /health`, `GET /model-info`,
  `POST /predict`; a `lifespan` handler loads the `Predictor` singleton at startup and
  degrades gracefully (logs a warning, leaves it unset) if artifacts are missing rather
  than crashing; `get_predictor` dependency raises 503 when unset. Pydantic request model
  uses `Literal["charge","discharge","rest"]` and `Field(min_length=2)` for automatic 422s
- `tests/test_api.py` — 5 tests using `app.dependency_overrides` to swap in an in-memory
  fake predictor, so the suite never touches the real model/scaler artifacts (`data/` is
  gitignored and CI does a fresh checkout)
- Added `httpx` to dev dependencies (required by FastAPI's `TestClient`)

### Results / Metrics
- `pytest` (20 tests total), `ruff`, `black`, `mypy` all pass clean
- Manual smoke test against the real exported model via
  `uvicorn battery_rul.inference.api:app --port 8000`:
  `/health` → `{"status":"ok","model_loaded":true}`,
  `/model-info` → architecture/rmse/training_battery/test_batteries all correct,
  `/predict` on a sample discharge reading → `soh=0.756`, `rul_estimate="Replace soon"`,
  `confidence="high"`

### Issues Encountered
- The original spec only defines RUL buckets for SOH < 85% ("Replace soon") and > 90%
  ("Healthy"), leaving 85-90% undefined. Filled the gap with a "Monitor" bucket — simplest
  reasonable choice for the unspecified middle range.
- The model only ever sees the last single timestep's 9 features at inference (`Trainer`'s
  flat input mode slices the final row of each window), not a full 50-step window — this
  simplified `Predictor.predict` to a single feature-row reconstruction instead of
  simulating an entire window.

### Next Phase
Phase 7 — Advanced Model Comparison (LightGBM + Attention)

## Phase 7 — Advanced Model Comparison (LightGBM + Attention) — COMPLETE

**Date:** 2026-06-18

### What Was Built
- `src/battery_rul/models/trees.py` — `BatteryLightGBM`, a thin `fit`/`predict`/`save`/`load`
  wrapper around `LGBMRegressor` operating directly on per-row tabular features (no
  windowing — the rolling/derivative features already encode short-term history per row)
- `src/battery_rul/models/attention.py` — `BatteryAttention`: input projection, learned
  positional embedding, 2 `nn.TransformerEncoderLayer` blocks, head on the final timestep.
  Same `(batch, seq_len, features) -> (batch, 1)` contract as `BatteryLSTM`, so it plugs
  into `Trainer`'s existing `input_mode="sequence"` path unchanged — this fulfills a
  lightweight self-attention upgrade over the voltage history window that had been
  planned but never implemented in Phases 1-6
- `src/battery_rul/training/tree_trainer.py` — `fit_lightgbm`, a non-epoch training path
  for tree models that logs params + per-battery RMSE/MAE/R2 to MLflow via
  `mlflow.lightgbm.log_model`
- `scripts/train.py` — added `attention` (reuses the `Trainer`/sequence path) and
  `lightgbm` (bypasses `Trainer` entirely, calls `fit_lightgbm` on flat parquet rows)
- `src/battery_rul/evaluation/predictions.py` — added `predict_battery_lightgbm` (flat
  per-row prediction, no windowing) alongside the existing windowed `predict_battery`
- `scripts/precompute_predictions.py` — added RUN_CONFIGS entries for both new models;
  LightGBM loads via `mlflow.lightgbm.load_model` and skips per-epoch history (boosting
  rounds aren't epochs)
- `src/battery_rul/dashboard/app.py` — added Attention and LightGBM to the Approach-2
  model dropdown, comparison table, and label map; LightGBM's training-curve panel shows
  a placeholder explaining why no per-round history exists
- `tests/test_models.py` — attention forward-shape tests (full window + shorter window),
  extended parameter-count sanity check
- `tests/test_trees.py` — fit/predict shape, fit reduces error on synthetic data,
  save/load round-trip

### Results / Metrics
| Model | Approach | RW10 | RW11 | RW12 | Average RMSE |
|---|---|---|---|---|---|
| paper_dnn | 2 | 0.12% | 0.14% | 1.71% | **0.66%** |
| lstm | 2 | 0.31% | 0.41% | 1.76% | **0.83%** |
| upgraded_dnn | 2 | 0.81% | 0.75% | 2.22% | **1.26%** |
| lightgbm | 2 | 0.20% | 0.21% | 1.64% | **0.68%** |
| attention | 2 | 0.26% | 0.27% | 1.87% | **0.80%** |

Neither new model beat `paper_dnn`'s 0.66%. LightGBM came closest (0.68%) — strong
evidence that the per-row engineered features (rolling mean/std, dv/dt) already capture
the temporal signal that matters, so a model that ignores sequence order entirely loses
almost nothing. Attention (0.80%) landed between `lstm` and `paper_dnn`, better than
`upgraded_dnn` but not better than the simplest model. Combined with the Phase 4 finding
that `upgraded_dnn` underperformed `paper_dnn`, the pattern across all 5 models is
consistent: on this dataset, model capacity and architectural sophistication don't buy
accuracy past what `paper_dnn`'s small 2-layer network already extracts from the 5 raw
features — the bottleneck is the data/feature relationship itself, not the network.

### Issues Encountered
- **macOS dual-OpenMP-runtime segfault**: PyTorch's wheel bundles its own `libomp.dylib`;
  LightGBM's compiled extension links against a separate Homebrew-installed copy at
  `/opt/homebrew/opt/libomp/lib/libomp.dylib`. Loading both into one process crashes with
  `Fatal Python error: Segmentation fault` on LightGBM's first native call (`.fit()`).
  `brew install libomp` was a separate, necessary prerequisite (LightGBM's wheel doesn't
  bundle the runtime it links against) but didn't fix the dual-runtime conflict on its own.
  First attempted a `DYLD_LIBRARY_PATH` + process re-exec workaround (`os.execve` after
  setting the env var, since `DYLD_LIBRARY_PATH` only affects dylib resolution for a fresh
  process, not one already running) — this worked standalone but broke under pytest:
  pytest's default fd-level output capturing replaces stdout/stderr with internal temp
  files *before* `conftest.py` runs, so the re-exec'd process inherited those redirected,
  soon-to-be-orphaned fds and all of its output vanished (confirmed exit 0, zero bytes,
  even with explicit shell redirection; running with `-s` to disable capturing fixed it,
  confirming the capture manager as the cause). Replaced the env-var approach entirely
  with a one-time, venv-scoped binary patch: `install_name_tool -change @rpath/libomp.dylib
  <path-to-torch's-libomp.dylib> <path-to-lib_lightgbm.dylib>`, rewriting LightGBM's
  compiled extension to load PyTorch's copy directly. No environment variables, re-exec,
  or code in the repo are needed — confirmed working under plain `python -c`, the full
  `pytest tests/ -v` run (25 passed), and `scripts/train.py --model lightgbm`. This is a
  local machine fix scoped to this project's `.venv` only (not global Homebrew state) and
  needs to be reapplied if the venv is recreated from scratch.

### Next Phase
Phase 8 — EDA Notebooks

## Phase 8 — EDA Notebooks — COMPLETE

**Date:** 2026-06-18

### What Was Built
- `notebooks/01_eda.ipynb` — loads all 4 processed parquets (stride-subsampled),
  voltage-vs-time and temperature-vs-time per battery, current distribution by
  charge/discharge/rest, SOH degradation curves, and a feature correlation heatmap on
  RW9 (reproducing the paper's Fig. 7), each followed by a markdown finding
- `notebooks/02_model_comparison.ipynb` — loads precomputed predictions for all 5
  Approach-2 models, actual-vs-predicted SOH overlay, per-epoch training curves for the
  4 torch models, a final RMSE comparison table (published baselines + all 5 of our
  models), and an overestimation-vs-underestimation bar chart using `evaluation/metrics.py`
- Added `jupyter`/`nbconvert` to dev dependencies (needed to author/execute notebooks;
  charts use `plotly`, already a runtime dependency, instead of adding matplotlib/seaborn)

### Results / Metrics
- Both notebooks executed top-to-bottom via `jupyter nbconvert --execute` with zero
  error cells
- `02_model_comparison.ipynb`'s computed comparison table matches Phase 7's training-run
  numbers to within rounding: paper_dnn 0.66%, lightgbm 0.68%, attention 0.80%,
  lstm 0.81% (vs. 0.83% reported during training — same small variance already noted in
  Phase 5 between full-resolution and stride-subsampled prediction files), upgraded_dnn
  1.26%
- `pytest` (25 tests), `ruff`, `black`, `mypy` all pass clean

### Issues Encountered
- The original Phase 8 plan said "load all 4 battery raw CSVs," but the raw data is
  `.mat` (see Phase 2) and the processed parquets already carry preserved unscaled
  `voltage_raw`/`absolute_time_raw` columns alongside the engineered features — loaded
  from `data/processed/*.parquet` instead, consistent with the dashboard's data source
- `current` and `temperature` have no preserved unscaled column (only voltage and
  absolute_time do, added in Phase 5 for the dashboard); the current-distribution and
  temperature plots use the min-max scaled [0, 1] values with a markdown note explaining
  the scale, rather than adding new raw-preserving columns to `preprocess.py` for a plot
- Plotly's `application/vnd.plotly.v1+json` output mimetype renders natively in GitHub's
  notebook viewer, so the interactive charts display correctly without requiring
  matplotlib/static images or a kaleido dependency

### Next Phase
Phase 9 — README & Final Polish

## Phase 9 — README & Final Polish — COMPLETE

**Date:** 2026-06-18

### What Was Built
- `README.md` rewritten in full: why-it-matters framing, results table, architecture
  diagram, quick start, dashboard screenshot, project structure, paper reference
- `docs/architecture.md` — system design notes covering the data pipeline, model
  architecture decisions (including why Phase 7 added LightGBM/Attention), training
  methodology, and API design rationale

### Results / Metrics
- README results table deviates from the original template (which only listed
  3 published baselines + 3 blank "ours" rows) — used the full 8-row sorted comparison
  from `02_model_comparison.ipynb` instead, since it surfaces the actual headline result
  (paper_dnn replica at 0.66% beats every model and baseline, including its own
  "upgraded" successor) rather than hiding it
- `ruff`, `black --check`, `mypy`, `pytest` (25 tests) all pass clean

### Issues Encountered
- None blocking; the only deviation is the results-table format noted above, logged per
  this project's deviation-logging convention (simplest reasonable choice, logged, continue)

### Next Phase
None — v1.0.0 portfolio release.
