# Antibody Training Pipeline (ESM) - Gemini Context

## Project Overview

This project is a machine learning pipeline designed to predict antibody non-specificity (polyreactivity). It implements the methodology from **Sakhnini et al. (2025)**, utilizing the **ESM-1v Protein Language Model (PLM)** for sequence embeddings and a **Logistic Regression** or **XGBoost** classifier for prediction.

### Key Features
*   **Two-Stage Architecture:**
    1.  **Embedding:** Generates high-dimensional vectors from antibody Variable Heavy (VH) domain sequences using ESM-1v (or other PLMs).
    2.  **Classification:** Predicts specificity/non-specificity using a classifier trained on these embeddings.
*   **Production-Ready Persistence:** Supports dual-format model saving:
    *   **Development:** Standard Pickle (`.pkl`) for full state preservation.
    *   **Production:** Secure, zero-code-execution NumPy Archives (`.npz`) + JSON Metadata (`.json`) for safe deployment.
*   **Interactive Web UI:** Built-in **Gradio** application for real-time inference, featuring request queueing, input validation, and observability.
*   **Tech Stack:** Python 3.12, PyTorch, Transformers (Hugging Face), Scikit-learn, XGBoost, Gradio, Hydra (configuration), uv (package management).
*   **Quality Assurance:** Strict typing (`mypy`), linting/formatting (`ruff`), and comprehensive testing (`pytest`).

## Directory Structure

*   `src/antibody_training_esm/`: Main source code.
    *   `conf/`: Hydra configuration files (`config.yaml` is the entry point).
    *   `core/`: Core application logic (Trainer, Predictor, Embeddings).
    *   `cli/`: Command-line interface entry points (`train`, `test`, `predict`, `app`).
    *   `data/` & `datasets/`: Data loading and processing modules.
*   `configs/`: Additional configuration files (e.g., testing configs).
*   `docs/`: Extensive documentation.
    *   `user-guide/`: **Web App Guide**, Inference, Training, etc.
    *   `developer-guide/`: Architecture, Testing Strategy, etc.
*   `experiments/`: Output directory for training runs, logs, and checkpoints.
*   `preprocessing/`: Scripts for raw data processing (Boughter, Harvey, Jain, Shehata).
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
| `make test` | Run fast test suite (~95s, skips e2e/slow/gpu tests). |
| `make test-e2e` | Run end-to-end tests (honors opt-in env vars like RUN_NOVO_E2E=1). |
| `make test-all` | Run full test suite (env-gated tests may still skip without flags). |
| `make train` | Launch the training pipeline using default Hydra config. |
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

### 4. Web Application (Gradio)

Launch the interactive web interface for predictions.

```bash
# Launch with production NPZ model
uv run antibody-app \
    classifier.path=experiments/checkpoints/esm1v/logreg/model.npz \
    classifier.config_path=experiments/checkpoints/esm1v/logreg/model_config.json
```

## Datasets

1.  **Boughter (Training):** 914 VH sequences, ELISA polyreactivity.
2.  **Jain (Test):** 86 clinical antibodies, used for "Novo Parity" benchmarking.
3.  **Harvey (Test):** ~141k nanobodies, PSR assay.
4.  **Shehata (Test):** 398 human antibodies, PSR cross-validation.

## Development Conventions

*   **Strict Typing:** All new code must be fully typed (`mypy` strict mode).
*   **Formatting:** We use `ruff`. Run `make format` before committing.
*   **Testing:** We use `pytest` with specific markers:
    *   `@pytest.mark.unit`: Fast, isolated unit tests.
    *   `@pytest.mark.integration`: Tests involving file I/O or component interaction.
    *   `@pytest.mark.e2e`: End-to-end flows (expensive).
    *   `@pytest.mark.slow`: Tests taking >30s.
    *   `@pytest.mark.legacy`: Tests for backward compatibility.

## Key Files

*   `README.md`: High-level entry point.
*   `pyproject.toml`: Project metadata, dependencies, and tool configurations.
*   `Makefile`: Task automation.
*   `src/antibody_training_esm/core/trainer.py`: Main training loop & model saving logic.
*   `src/antibody_training_esm/cli/app.py`: Gradio web application entry point.
*   `src/antibody_training_esm/core/prediction.py`: Inference logic (supporting .pkl and .npz).
*   `docs/developer-guide/`: Detailed architectural and workflow documentation.