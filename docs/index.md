# Antibody Non-Specificity Prediction Pipeline using ESM

<div align="center">
  <picture>
    <source srcset="assets/leeroy_jenkins.webp" type="image/webp">
    <img src="assets/leeroy_jenkins.png" alt="Leeroy Jenkins" width="300" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  </picture>
  <br>
  <em>"⏰ Times up, let's do this." - Leeroy Jenkins</em>
</div>

---

[![CI Pipeline](https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM/actions/workflows/ci.yml/badge.svg)](https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM/actions/workflows/ci.yml)
[![Docker CI](https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM/actions/workflows/docker-ci.yml/badge.svg)](https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM/actions/workflows/docker-ci.yml)
[![codecov](https://codecov.io/gh/the-obstacle-is-the-way/antibody_training_pipeline_ESM/branch/main/graph/badge.svg)](https://codecov.io/gh/the-obstacle-is-the-way/antibody_training_pipeline_ESM)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

This repository provides a machine learning pipeline to predict the non-specificity of antibodies using embeddings from the ESM-1v Protein Language Model (PLM). The project is an implementation of the methods described in the paper *"Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"* by Sakhnini et al.

---

## What is Antibody Non-Specificity?

Non-specific binding of therapeutic antibodies can lead to faster clearance from the body and other unwanted side effects, compromising their effectiveness and safety. Predicting this property, also known as **polyreactivity**, from an antibody's amino acid sequence is a critical step in drug development.

This project offers a computational pipeline to tackle this challenge. It leverages the power of **ESM-1v**, a state-of-the-art protein language model, to convert antibody amino acid sequences into meaningful numerical representations (embeddings). These embeddings capture complex biophysical and evolutionary information, which is then used to train a machine learning classifier to predict non-specificity.

---

## Model Architecture

The model's architecture is a two-stage process designed for both power and interpretability:

1. **Sequence Embedding with ESM-1v**: The amino acid sequence of an antibody's Variable Heavy (VH) domain is fed into the pre-trained ESM-1v model. ESM-1v, trained on millions of diverse protein sequences, generates a high-dimensional vector (embedding) for the antibody. This vector represents the learned structural and functional properties of the sequence.

2. **Classification**: The generated embedding vector is then used as input for a classical machine learning model. The original paper found that a **Logistic Regression** classifier performed best, achieving up to 71% accuracy in 10-fold cross-validation. This second stage learns to map the sequence features captured by ESM-1v to a binary outcome: **specific** or **non-specific**.

This hybrid approach combines the deep contextual understanding of a protein language model with the efficiency and interpretability of a linear classifier.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM.git
cd antibody_training_pipeline_ESM

# Set up environment with uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv (Linux/macOS)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --all-extras
```

### Training a Model

```bash
# Train with default config
make train

# Override parameters with Hydra
uv run antibody-train hardware.device=cuda training.batch_size=32

# Run hyperparameter sweeps
uv run antibody-train --multirun classifier.C=0.1,1.0,10.0
```

### Making Predictions

```bash
# CLI prediction
uv run antibody-predict \
    input_file=data/test.csv \
    output_file=predictions.csv \
    classifier.path=experiments/checkpoints/esm1v/logreg/model.pkl

# Web UI (Gradio)
uv run antibody-app classifier.path=experiments/checkpoints/esm1v/logreg/model.pkl
```

---

## Features

### Implemented ✅

- **Data Processing**: Load, clean, and process antibody datasets including the Boughter et al. (2020) training dataset
- **Sequence Annotation**: ANARCI-based annotation of CDRs and extraction of VH domains from full antibody sequences
- **ESM-1v Embedding**: Generate embeddings for antibody sequences using the ESM-1v protein language model
- **Model Training**: Complete training pipeline with 10-fold cross-validation using Logistic Regression or XGBoost
- **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC) across multiple test sets
- **Prediction CLI**: Get predictions for new antibody sequences from trained models
- **Web Application**: Gradio-based frontend for interactive prediction with device optimization
- **Hydra Configuration**: Flexible config management with CLI overrides and hyperparameter sweeps

### Roadmap 🚧

- **Biophysical Descriptor Module**: Calculate and incorporate key biophysical parameters (e.g., isoelectric point)
- **Support for Other PLMs**: Integration of antibody-specific language models (AbLang, AntiBERTy)

---

## Datasets

This pipeline uses four antibody datasets for training and evaluation:

| Dataset | Size | Assay | Usage |
|---------|------|-------|-------|
| **Boughter** | 914 VH sequences | ELISA | Primary training dataset |
| **Jain** | 86 clinical antibodies | Per-antigen ELISA | Clinical antibody benchmark (68.60% - EXACT NOVO PARITY) |
| **Harvey** | 141k nanobody sequences | PSR assay | Large-scale nanobody test set |
| **Shehata** | 398 human antibodies | PSR assay | Cross-assay validation |

See the [Datasets](datasets/boughter/README.md) section for preprocessing details and data sources.

---

## Documentation Structure

- **[User Guide](user-guide/getting-started.md)**: Installation, training, testing, inference, troubleshooting
- **[Research](research/novo-parity.md)**: Novo parity analysis, methodology, benchmarks
- **[Developer Guide](developer-guide/architecture.md)**: Architecture, workflows, testing, CI/CD
- **[API Reference](api/core/trainer.md)**: Auto-generated API documentation from source code
- **[Datasets](datasets/boughter/README.md)**: Dataset-specific preprocessing and validation docs

---

## Development Workflow

This project uses modern Python tooling for a streamlined development experience:

```bash
# Install dependencies
make install

# Run full quality pipeline (format, lint, typecheck, test)
make all

# Run tests with coverage
make coverage

# Serve documentation locally
make docs-serve
```

See the [Development Workflow](developer-guide/development-workflow.md) guide for details.

---

## Citation

This work implements the methodology from:

**Sakhnini et al. (2025) - Novo Nordisk & University of Cambridge**

> Sakhnini, L.I., Beltrame, L., Fulle, S., Sormanni, P., Henriksen, A., Lorenzen, N., Vendruscolo, M., & Granata, D. (2025). Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. *bioRxiv*. https://doi.org/10.1101/2025.04.28.650927

```bibtex
@article{sakhnini2025prediction,
  title={Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters},
  author={Sakhnini, Laila I. and Beltrame, Ludovica and Fulle, Simone and Sormanni, Pietro and Henriksen, Anette and Lorenzen, Nikolai and Vendruscolo, Michele and Granata, Daniele},
  journal={bioRxiv},
  year={2025},
  month={May},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.04.28.650927},
  url={https://www.biorxiv.org/content/10.1101/2025.04.28.650927v1}
}
```

For complete dataset citations, see [CITATIONS.md](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/blob/main/CITATIONS.md).

---

## License

Apache License 2.0 - See [LICENSE](https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/blob/main/LICENSE) for details.
