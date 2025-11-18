# Antibody Training Pipeline (ESM) - Gemini Context

## Project Overview

This project is a machine learning pipeline designed to predict antibody non-specificity (polyreactivity). It implements the methodology from **Sakhnini et al. (2025)**, utilizing the **ESM-1v Protein Language Model (PLM)** for sequence embeddings and a **Logistic Regression** classifier for prediction.

### Key Features
*   **Two-Stage Architecture:**
    1.  **Embedding:** Generates high-dimensional vectors from antibody Variable Heavy (VH) domain sequences using ESM-1v.
    2.  **Classification:** Predicts specificity/non-specificity using a Logistic Regression model trained on these embeddings.
*   **Tech Stack:** Python 3.12, PyTorch, Transformers (Hugging Face), Scikit-learn, Hydra (configuration), uv (package management).
*   **Quality Assurance:** Strict typing (`mypy`), linting/formatting (`ruff`), and comprehensive testing (`pytest`).

## Directory Structure

*   `src/antibody_training_esm/`: Main source code.
    *   `conf/`: Hydra configuration files (`config.yaml` is the entry point).
    *   `core/`: Core application logic (e.g., `trainer.py`).
    *   `cli/`: Command-line interface entry points.
    *   `data/` & `datasets/`: Data loading and processing modules.
*   `configs/`: Additional configuration files (e.g., testing configs).
*   `docs/`: Extensive documentation (Developer guides, Research notes, User guides).
*   `experiments/`: Output directory for training runs, logs, and checkpoints.
*   `preprocessing/`: Scripts for raw data processing (Boughter, Harvey, Jain, Shehata datasets).
*   `tests/`: Unit, integration, and E2E tests.

## Setup & Usage

The project uses `uv` for dependency management and `make` for common tasks.

### 1. Installation

```bash
make install
# Or manually:
# uv sync --all-extras
```

### 2. Common Commands

| Command | Description |
| :--- | :--- |
| `make all` | Run the full quality pipeline (Format -> Lint -> Typecheck -> Test). **Run this before committing.** |
| `make train` | Launch the training pipeline using default Hydra config. |
| `make test` | Run the full test suite. |
| `make lint` | Run `ruff` linting. |
| `make format` | Auto-format code with `ruff`. |
| `make typecheck` | Run static type analysis with `mypy`. |

### 3. Training Configuration (Hydra)

The pipeline is configured via [Hydra](https://hydra.cc).
*   **Default Config:** `src/antibody_training_esm/conf/config.yaml`
*   **CLI Overrides:** You can override any config parameter from the command line.

**Examples:**
```bash
# Default training
uv run antibody-train

# Override specific parameters
uv run antibody-train hardware.device=cuda training.batch_size=32

# Hyperparameter sweep (multirun)
uv run antibody-train --multirun classifier.C=0.1,1.0,10.0
```

## Datasets

1.  **Boughter (Training):** 914 VH sequences, ELISA polyreactivity.
2.  **Jain (Test):** 86 clinical antibodies, used for "Novo Parity" benchmarking.
3.  **Harvey (Test):** ~141k nanobodies, PSR assay.
4.  **Shehata (Test):** 398 human antibodies, PSR cross-validation.

## Development Conventions

*   **Strict Typing:** All new code must be fully typed. `mypy` is configured with `disallow_untyped_defs = true`.
*   **Formatting:** We use `ruff` for formatting. Run `make format` to ensure compliance.
*   **Testing:** Add unit tests for new features. Use `pytest` markers (`@pytest.mark.unit`, `@pytest.mark.integration`) appropriately.
*   **Pre-commit:** Use `make hooks` to run pre-commit checks locally.

## Key Files

*   `README.md`: High-level entry point.
*   `pyproject.toml`: Project metadata, dependencies, and tool configurations.
*   `Makefile`: Task automation.
*   `src/antibody_training_esm/core/trainer.py`: Main training loop entry point.
*   `docs/developer-guide/`: Detailed architectural and workflow documentation.
