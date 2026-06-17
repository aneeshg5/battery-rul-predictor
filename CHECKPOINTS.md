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
- Full directory layout per Section 3 (`src/battery_rul/*`, `tests/`, `scripts/`, `notebooks/`, `data/`)
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
- CLAUDE.md assumed the dataset was served as flat CSVs via a Socrata API at
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
