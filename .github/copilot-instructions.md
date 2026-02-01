# AI Agent Instructions for Hypotension Prediction Pipeline

## Project Overview

**Purpose**: Predict intraoperative hypotension (MAP < 65 mmHg) **5 minutes in advance** using VitalDB surgical vital signs data.

**Architecture**: Two-stage pipeline with built-in cost control:
1. **Data Stage** (`build_dataset.py`): Extract features from raw VitalDB vital files → CSV dataset
2. **Training Stage** (`train_model.py`): PyTorch CUDA model on extracted features → saved checkpoint

**Critical Feature**: All stages auto-save and halt when `MAX_RUNTIME_MINUTES` or `MAX_TRAIN_STEPS` is reached (prevents cloud billing overruns).

## Data Flow & Key Patterns

### Data Sources
- **VitalDB Vital Files**: Binary `.vital` files at `C:\Users\sck32\Documents\Python_Scripts\Open VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients\vital_files\{caseid:04d}.vital`
- **Clinical Metadata**: `clinical_data.csv` with caseid index
- **Loaded via**: `vitaldb` library (`load_vital_case()` in `data_loader.py`)

### Feature Engineering (Lookback Window)
- **Window**: 5 minutes (300 seconds) of prior vital signals
- **Granularity**: 1-second sample interval
- **Extracted Features**: Per vital signal (MAP, HR):
  - `{SIGNAL}_mean`, `{SIGNAL}_std`, `{SIGNAL}_min`
  - NaN handling: `fillna(0)` in train_model.py
- **Pattern**: Non-overlapping 1-min steps through each case (see `extract_features()` and `build_labels_for_case()`)

### Label Definition (5-Min Horizon)
- **Hypotension Event**: MAP < 65 mmHg for ≥60 consecutive seconds
- **Prediction Window**: 5 minutes (300 sec) after lookback ends
- **Logic**: Convolve future MAP with 60-second kernel to detect sustained low events (`build_labels_for_case()`)
- **Class Balance**: Moderate imbalance typical; uses stratified split in train_model.py

## Critical Configuration & Cost Control

**File**: `config.py`

| Setting | Default | Purpose |
|---------|---------|---------|
| `MAX_RUNTIME_MINUTES` | 30 | Stop dataset building after N minutes (prevents billing) |
| `MAX_TRAIN_STEPS` | 500 | Stop training after N batches (prevents CUDA hours) |
| `MAX_CASES` | 100 | Limit cases processed in build_dataset (for testing) |
| `DEVICE` | "cuda" | Force GPU (auto-falls back to CPU if unavailable) |

**Pattern**: Set to `None` to remove limits (only for local dev, never in cloud).

## Workflow Commands

### Full Pipeline
```bash
python run_all.py  # Builds dataset IF needed, then trains
```

### Individual Stages
```bash
python build_dataset.py   # Extract features only
python train_model.py     # Train model (requires dataset)
```

### VSCode Tasks (Ctrl+Shift+B)
- **"2. 전체 파이프라인 실행 (run_all)"** ← Default; auto-installs deps
- **"3. 주피터 노트북 자동 실행"** → Runs notebook with nbconvert

### Quick Start Scripts
- `run.bat` or `run.ps1` → Installs deps + runs `run_all.py`

## Model Architecture & Training

**Model Class**: `HypoNet` (in `train_model.py`)
```python
Input → Linear(in_dim, 64) → ReLU → Dropout(0.2) → Linear(64, 32) → ReLU → Dropout(0.2) → Linear(32, 1) → sigmoid
```

**Training Setup**:
- **Loss**: `BCEWithLogitsLoss()` (binary classification)
- **Optimizer**: Adam(lr=1e-3)
- **Batch Size**: 256
- **Data Split**: 80/20 stratified train/test (via `sklearn.train_test_split`)

**Checkpoint Pattern**: On max steps/OOM, saves then exits with `SystemExit(0)` (intentional abort, not error).

## File Organization

```
hypo_vitaldb/
├── config.py                  # Central config (edit thresholds, limits here)
├── data_loader.py             # VitalDB I/O + label logic
├── build_dataset.py           # Feature extraction (with auto-save on timeout)
├── train_model.py             # PyTorch training (with auto-save on timeout/OOM)
├── run_all.py                 # Orchestrator (calls build_dataset, then train_model)
├── hypotension_pipeline.ipynb # Interactive Jupyter version
├── requirements.txt           # vitaldb, torch+CUDA, pandas, sklearn, tqdm
└── checkpoints/               # Model outputs (hypo_model.pt, train_state.pt)
```

## Common Debugging Patterns

### Dataset is empty or too small
- **Check**: `MAX_CASES` in `config.py` (default 100 for testing)
- **Check**: VitalDB vital_files path exists and contains `.vital` files
- **Check**: `build_labels_for_case()` filters out cases with NaN MAP values

### CUDA OOM during training
- **Pattern**: Intentionally caught and triggers checkpoint save + `SystemExit(0)`
- **To resume**: Load checkpoint state from `TRAIN_STATE_PATH` (not yet auto-resumed)
- **Workaround**: Reduce batch size in train_model.py or set `DEVICE = "cpu"` in config.py

### Train/test split logic
- **Stratified by**: `y` (label distribution preserved across splits)
- **Fixed seed**: `RANDOM_STATE = 42` ensures reproducibility

## Conventions & Pitfalls

- **Feature naming**: `{signal}_{stat}` (e.g., `MBP_mean`, `HR_std`) — avoid spaces
- **Progress bars**: All loops use `tqdm` with descriptive `desc=` (user visibility)
- **Time-based exit**: Exit with `raise SystemExit(0)` when hitting time/step limits (not exception)
- **Config paths**: Use `Path(__file__).resolve().parent` for robustness (handles VitalDB's comma-laden folder name)
- **NaN handling**: `fillna(0)` in model input; features may contain NaN if signal length < 10 samples

## Testing & Validation

**Minimal smoke test**: Set `MAX_CASES=1` and `MAX_TRAIN_STEPS=5` in config.py, run `python run_all.py` (should complete in <1 min).

**Validation outputs** (from train_model.py):
- Classification report (precision, recall, F1)
- AUC-ROC score
- Confusion matrix
