# Architecture

**Target Audience:** Developers contributing to the codebase

**Purpose:** Understand the system architecture, core components, and design patterns used throughout the pipeline

---

## When to Use This Guide

Use this guide if you're:
- ✅ **New to the codebase** (onboarding)
- ✅ **Understanding how components interact** (pipeline flow, module dependencies)
- ✅ **Adding new features** (need to know where code lives)
- ✅ **Debugging issues across modules** (tracing data flow)

---

## Related Documentation

- **User Guide:** [System Overview](../overview.md) - High-level system introduction
- **Developer Guides:**
  - [Development Workflow](development-workflow.md) - Git, make commands, pre-commit hooks
  - [Preprocessing Internals](preprocessing-internals.md) - Dataset preprocessing patterns
  - [Testing Strategy](testing-strategy.md) - Test architecture and patterns
  - [Type Checking](type-checking.md) - Type safety requirements
- **Implementation Details:** See source code in `src/antibody_training_esm/`

---

## Core Pipeline Flow

1. **Data Loading** (`src/antibody_training_esm/data/loaders.py`) → Load CSV datasets
2. **Embedding Extraction** (`src/antibody_training_esm/core/embeddings.py`) → ESM-1v embeddings with batching and caching
3. **Classification** (`src/antibody_training_esm/core/classifier.py`) → LogisticRegression on embeddings
4. **Training** (`src/antibody_training_esm/core/trainer.py`) → 10-fold CV, model persistence, evaluation
5. **CLI** (`src/antibody_training_esm/cli/`) → User-facing commands

---

## Key Modules

### `core/embeddings.py`

**ESMEmbeddingExtractor** handles:
- Loading ESM-1v from HuggingFace with pinned revisions
- Batch processing with GPU memory management
- Mean-pooling of last hidden states
- Device support: CPU, CUDA, MPS

### `core/classifier.py`

**BinaryClassifier** provides:
- Dual initialization API (dict-based legacy + sklearn kwargs)
- Assay-specific thresholds (ELISA: 0.5, PSR: 0.5495)
- Logistic regression hyperparameters from config
- Embedding extraction + classification pipeline

### `core/trainer.py`

**train_model** orchestrates:
- Config loading from YAML
- Embedding caching (SHA-256 hashed paths)
- 10-fold stratified cross-validation on training set
- Train on full training set, test on hold-out test set
- Model persistence to `.pkl` files
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)

### `datasets/base.py`

**AntibodyDataset** abstract base class defines:
- Standard fragment types (VH, VL, CDRs, FWRs, Full)
- ANARCI annotation interface (IMGT numbering)
- Common preprocessing methods
- Fragment extraction for all datasets

### `datasets/{boughter,jain,harvey,shehata}.py`

**Dataset-specific loaders** that:
- Implement AntibodyDataset interface
- Handle dataset-specific quirks
- Provide default paths to canonical CSV files
- Support fragment-level loading

---

## Directory Structure

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

conf/                         # Hydra configuration directory (inside package)
├── config.yaml              # Default Hydra config (Boughter train, Jain test)

models/                       # Trained model checkpoints (.pkl)
embeddings_cache/            # Cached ESM embeddings
train_datasets/              # Training data CSVs
test_datasets/               # Test data CSVs
tests/                       # Test suite
├── unit/                    # Fast unit tests (< 1s each)
├── integration/             # Integration tests
└── e2e/                     # End-to-end tests (expensive)
```

---

## Important Patterns & Conventions

### Configuration System

- All training controlled via Hydra configs in `conf/` (inside package)
- Default config: `conf/config.yaml` (Boughter → Jain)
- Override any parameter from CLI without editing files: `antibody-train hardware.device=cuda`
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

---

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

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
