# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an antibody non-specificity prediction pipeline using ESM-1v protein language models. It implements the methodology from Sakhnini et al. (2025) - training on the Boughter dataset and testing on Jain, Harvey, and Shehata datasets. The pipeline combines ESM-1v embeddings with logistic regression for binary classification of antibody polyreactivity.

## Development Commands

### Environment Setup
```bash
uv sync --all-extras     # Install all dependencies including dev tools
```

### Testing
```bash
uv run pytest                                    # Run all tests
uv run pytest -m unit                           # Run only unit tests (fast)
uv run pytest -m integration                    # Run integration tests
uv run pytest -m e2e                           # Run end-to-end tests (expensive)
uv run pytest tests/unit/core/test_trainer.py  # Run specific test file
uv run pytest -k test_function_name            # Run specific test by name
uv run pytest --cov=. --cov-report=html --cov-report=term-missing --cov-fail-under=70  # Coverage report
```

**Test markers:** All tests must be tagged with `unit`, `integration`, `e2e`, or `slow` markers. Register new markers in `pyproject.toml` before using.

### Code Quality
```bash
make format      # Auto-format with ruff
make lint        # Lint with ruff
make typecheck   # Type check with mypy (strict mode)
make hooks       # Run pre-commit hooks
make all         # Run format → lint → typecheck → test
```

**Critical:** This repo maintains 100% type safety. All functions must have complete type annotations. Mypy runs with `disallow_untyped_defs=true`.

### Training & Testing
```bash
make train                                                    # Train with default config
uv run antibody-train --config configs/config.yaml           # Train with specific config
uv run antibody-test --model models/model.pkl --dataset jain # Test trained model
```

### Preprocessing
```bash
# Boughter (training set)
python3 preprocessing/boughter/stage1_dna_translation.py
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py

# Jain (test set - Novo parity benchmark)
python3 preprocessing/jain/step1_convert_excel_to_csv.py
python3 preprocessing/jain/step2_preprocess_p5e_s2.py

# Harvey (nanobody test set)
python3 preprocessing/harvey/step1_convert_raw_csvs.py
python3 preprocessing/harvey/step2_extract_fragments.py

# Shehata (PSR assay test set)
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
python3 preprocessing/shehata/step2_extract_fragments.py
```

## Architecture

### Core Pipeline Flow
1. **Data Loading** (`src/antibody_training_esm/data/loaders.py`) → Load CSV datasets
2. **Embedding Extraction** (`src/antibody_training_esm/core/embeddings.py`) → ESM-1v embeddings with batching and caching
3. **Classification** (`src/antibody_training_esm/core/classifier.py`) → LogisticRegression on embeddings
4. **Training** (`src/antibody_training_esm/core/trainer.py`) → 10-fold CV, model persistence, evaluation
5. **CLI** (`src/antibody_training_esm/cli/`) → User-facing commands

### Key Modules

**`core/embeddings.py`**: ESMEmbeddingExtractor handles:
- Loading ESM-1v from HuggingFace with pinned revisions
- Batch processing with GPU memory management
- Mean-pooling of last hidden states
- Device support: CPU, CUDA, MPS

**`core/classifier.py`**: BinaryClassifier provides:
- Dual initialization API (dict-based legacy + sklearn kwargs)
- Assay-specific thresholds (ELISA: 0.5, PSR: 0.5495)
- Logistic regression hyperparameters from config
- Embedding extraction + classification pipeline

**`core/trainer.py`**: train_model orchestrates:
- Config loading from YAML
- Embedding caching (SHA-256 hashed paths)
- 10-fold stratified cross-validation on training set
- Train on full training set, test on hold-out test set
- Model persistence to `.pkl` files
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)

**`datasets/base.py`**: AntibodyDataset abstract base class defines:
- Standard fragment types (VH, VL, CDRs, FWRs, Full)
- ANARCI annotation interface (IMGT numbering)
- Common preprocessing methods
- Fragment extraction for all datasets

**`datasets/{boughter,jain,harvey,shehata}.py`**: Dataset-specific loaders that:
- Implement AntibodyDataset interface
- Handle dataset-specific quirks
- Provide default paths to canonical CSV files
- Support fragment-level loading

### Directory Structure
```
src/antibody_training_esm/    # Main package
├── core/                     # Core ML pipeline
│   ├── embeddings.py        # ESM-1v embedding extraction
│   ├── classifier.py        # Binary classifier (LogReg + ESM)
│   ├── trainer.py           # Training orchestration
│   └── config.py            # Config constants
├── data/                     # Data loading utilities
│   └── loaders.py           # CSV loading
├── datasets/                 # Dataset-specific loaders
│   ├── base.py              # Abstract base class
│   ├── boughter.py          # Training set
│   ├── jain.py              # Test set (Novo parity)
│   ├── harvey.py            # Test set (nanobodies)
│   └── shehata.py           # Test set (PSR assay)
├── cli/                      # Command-line interfaces
│   ├── train.py             # Training CLI
│   ├── test.py              # Testing CLI
│   └── preprocess.py        # Preprocessing CLI
└── evaluation/               # Evaluation utilities

preprocessing/                # Dataset preprocessing pipelines
├── boughter/                # 3-stage: DNA translation → annotation → QC
├── jain/                    # 2-step: Excel → CSV → P5e-S2
├── harvey/                  # 2-step: Combine CSVs → fragments
└── shehata/                 # 2-step: Excel → CSV → fragments

configs/                      # YAML configuration files
├── config.yaml              # Default production config (Boughter train, Jain test)

models/                       # Trained model checkpoints (.pkl)
embeddings_cache/            # Cached ESM embeddings
train_datasets/              # Training data CSVs
test_datasets/               # Test data CSVs
tests/                       # Test suite
├── unit/                    # Fast unit tests (< 1s each)
├── integration/             # Integration tests
└── e2e/                     # End-to-end tests (expensive)
```

## Important Patterns & Conventions

### Configuration System
- All training controlled via YAML configs in `configs/`
- Production config: `configs/config.yaml` (Boughter → Jain)
- Config structure: `model`, `data`, `classifier`, `training`, `experiment`, `hardware`
- HuggingFace model revisions pinned for reproducibility

### Dataset Organization
- **Training data**: `train_datasets/{dataset}/canonical/*.csv`
- **Test data**: `test_datasets/{dataset}/canonical/*.csv` or `fragments/*.csv`
- **Raw data**: Never committed to Git - stored in `test_datasets/` and preprocessed locally
- Each dataset has dedicated preprocessing pipeline in `preprocessing/{dataset}/`

### Embedding Caching
- ESM embeddings cached in `embeddings_cache/` as `.npy` files
- Cache key: SHA-256 hash of `model_name + dataset_path + revision`
- Prevents expensive re-computation during hyperparameter sweeps
- Cache invalidates automatically when model/data changes

### Model Persistence
- Trained models saved as `.pkl` files in `models/`
- Pickle usage limited to trusted local artifacts only
- **Threat model**: No internet-exposed API, no untrusted pickle loading
- Production deployment should migrate to JSON + NPZ (see `SECURITY_REMEDIATION_PLAN.md`)

### Type Safety
- 100% type coverage enforced via mypy with `disallow_untyped_defs=true`
- All public functions require complete type annotations
- Type failures block CI pipeline
- Track type remediation progress in `.mypy_failures.txt`

### Testing Strategy
- **Unit tests** (`tests/unit/`): Fast, isolated, mocked dependencies
- **Integration tests** (`tests/integration/`): Multi-component interactions
- **E2E tests** (`tests/e2e/`): Full pipeline (expensive, scheduled runs)
- HuggingFace downloads mocked via `tests/fixtures/mock_models.py` (even in e2e)
- Coverage requirement: ≥70% enforced in CI
- Test fixtures in `tests/fixtures/` with deterministic data

### Preprocessing Philosophy
- **Dataset-centric organization**: All preprocessing for a dataset lives in `preprocessing/{dataset}/`
- **Reproducibility**: All preprocessing scripts are versioned and documented
- **Validation**: Each preprocessing step has validation script (e.g., `validate_stages2_3.py`)
- **Intermediate outputs**: Staged outputs (raw → processed → canonical/fragments)

### Fragment Types
Standard fragments across all datasets:
- **VH/VL**: Variable heavy/light chains
- **CDRs**: H-CDR1/2/3, L-CDR1/2/3, H-CDRs, L-CDRs, All-CDRs
- **FWRs**: H-FWRs, L-FWRs, All-FWRs
- **Combined**: VH+VL, Full (VH+VL including linkers)
- **Nanobody-specific**: VHH_only, VHH-CDR1/2/3, VHH-CDRs, VHH-FWRs

### Assay-Specific Thresholds
- **ELISA** (Boughter, Jain): threshold = 0.5 (standard)
- **PSR** (Harvey, Shehata): threshold = 0.5495 (Novo Nordisk exact parity)
- Thresholds configured in `BinaryClassifier.ASSAY_THRESHOLDS`

## Security & Best Practices

### Pickle Usage
- **Approved use cases**: ML models, embedding caches, preprocessed datasets
- **All files generated locally** by trusted code
- **Never load untrusted pickle files**
- Run security scans: `uv run bandit -r src/` (must remain clean)

### Pre-commit Hooks
- Installed via `uv run pre-commit install`
- Auto-run on commit: ruff format, ruff lint, mypy
- Manual run: `make hooks`
- Failures block commits (intended behavior)

### CI Pipeline
- **Quality gate**: ruff, mypy, bandit (all must pass)
- **Unit tests**: Fast tests with ≥70% coverage
- **Integration tests**: Multi-component tests
- **E2E tests**: Scheduled runs only (expensive)
- **Security**: Bandit scan must show 0 findings

## Common Tasks

### Adding a New Dataset
1. Create `preprocessing/{dataset}/` directory
2. Implement preprocessing pipeline (Excel/CSV → canonical format)
3. Create `src/antibody_training_esm/datasets/{dataset}.py` extending `AntibodyDataset`
4. Add dataset documentation in `docs/{dataset}/`
5. Update `preprocessing/README.md`

### Training a New Model
1. Create config YAML in `configs/{experiment}.yaml`
2. Set `data.train_file`, `data.test_file`, `classifier` params
3. Run: `uv run antibody-train --config configs/{experiment}.yaml`
4. Model saved to `models/{model_name}.pkl`
5. Logs in `logs/{experiment}.log`

### Running Hyperparameter Sweeps
1. See `preprocessing/boughter/train_hyperparameter_sweep.py` for reference
2. Create sweep config with parameter grid
3. Embeddings auto-cached for fast re-runs
4. Results logged to sweep-specific directory

### Debugging Test Failures
1. Run specific test: `uv run pytest tests/unit/core/test_trainer.py -v`
2. Show print statements: `uv run pytest -s`
3. Drop into debugger: `uv run pytest --pdb`
4. Check fixtures: `tests/fixtures/mock_datasets/` for test data

## Git Workflow

### Main Branches
- `leroy-jenkins/full-send`: Main development branch (production-ready code)
- `dev`: Development branch (active work)

### Commit Conventions
- Conventional commits: `fix:`, `feat:`, `docs:`, `test:`, `refactor:`
- Imperative mood, ≤72 chars
- Example: `fix: Correct PSR threshold to 0.5495 for Novo parity`

### Pull Requests
- Must pass all CI checks (quality + tests)
- Include: scope summary, issue links, commands run (`make all`, `make coverage`)
- Call out new artifacts or data paths
- Keep refactors separate from feature/data work

## References

- **Paper**: Sakhnini et al. (2025) - Prediction of Antibody Non-Specificity using PLMs
- **Datasets**: See `CITATIONS.md` for full attributions
- **Security**: See `SECURITY_REMEDIATION_PLAN.md` for pickle mitigation
- **Architecture**: See `AGENTS.md` for build/test/commit guidelines
- **Dataset docs**: `docs/{boughter,jain,harvey,shehata}/` for dataset-specific details
