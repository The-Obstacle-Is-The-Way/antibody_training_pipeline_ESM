# System Overview

**Target Audience:** Everyone (first-time readers, researchers, potential users)

**Purpose:** High-level introduction to the antibody non-specificity prediction pipeline

---

## What is This System?

This is a **production-grade machine learning pipeline** for predicting antibody non-specificity (polyreactivity) using protein language models. The system combines state-of-the-art deep learning (ESM-1v) with interpretable classical ML (logistic regression) to classify therapeutic antibodies as **specific** or **non-specific**.

### The Problem

Non-specific binding of therapeutic antibodies can lead to:
- Faster clearance from the body
- Reduced drug efficacy
- Unwanted side effects
- Failed clinical trials

Predicting polyreactivity from amino acid sequence is **critical for drug development** but traditionally requires expensive wet-lab experiments.

### Our Solution

A **two-stage computational pipeline** that:
1. **ESM-1v Protein Language Model** → Converts antibody sequences to 1280-dimensional embeddings
2. **Logistic Regression Classifier** → Maps embeddings to binary predictions (specific/non-specific)

**Key Achievement:** Reproduces Novo Nordisk's published results with **exact parity** (66.28% accuracy on Jain benchmark).

---

## System Architecture

### High-Level Pipeline

```
Antibody Sequence (FASTA/CSV)
         ↓
┌────────────────────────┐
│  Data Loading          │  ← Load datasets (Boughter, Jain, Harvey, Shehata)
│  (loaders.py)          │
└────────────────────────┘
         ↓
┌────────────────────────┐
│  Embedding Extraction  │  ← ESM-1v: Sequence → 1280-dim vector
│  (embeddings.py)       │     • Batch processing
└────────────────────────┘     • GPU/CPU support
         ↓                      • Embedding caching (SHA-256)
┌────────────────────────┐
│  Classification        │  ← Logistic Regression on embeddings
│  (classifier.py)       │     • Assay-specific thresholds
└────────────────────────┘     • sklearn-compatible API
         ↓
┌────────────────────────┐
│  Training/Evaluation   │  ← 10-fold cross-validation
│  (trainer.py)          │     • Model persistence (.pkl)
└────────────────────────┘     • Comprehensive metrics
         ↓
   Prediction: 0 (specific) or 1 (non-specific)
```

### Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Embedding Extractor** | Convert sequences to vectors | ESM-1v (HuggingFace transformers) |
| **Binary Classifier** | Predict specificity | sklearn LogisticRegression |
| **Dataset Loaders** | Load & preprocess data | pandas, ANARCI (IMGT numbering) |
| **Training Pipeline** | Train & evaluate models | 10-fold stratified CV |
| **CLI Tools** | User-facing commands | `antibody-train`, `antibody-test` |
| **Caching System** | Speed up re-runs | SHA-256 hashed embeddings |

---

## Key Capabilities

### 1. **Multi-Dataset Training & Testing**

**Training Set:**
- **Boughter (2020):** 914 VH sequences, ELISA polyreactivity assay

**Test Sets:**
- **Jain (2017):** 86 clinical antibodies, per-antigen ELISA (Novo parity benchmark)
- **Harvey (2022):** 141,021 nanobody sequences, PSR assay
- **Shehata (2019):** 398 human antibodies, PSR cross-validation

### 2. **Fragment-Level Predictions**

The pipeline supports predictions on antibody fragments:
- **Full sequences:** VH, VL, VH+VL
- **CDRs (Complementarity-Determining Regions):** H-CDR1/2/3, L-CDR1/2/3, All-CDRs
- **FWRs (Framework Regions):** H-FWRs, L-FWRs, All-FWRs
- **Nanobodies:** VHH domain (Harvey dataset)

This enables **ablation studies** to determine which antibody regions drive non-specificity.

### 3. **Assay-Specific Calibration**

Different experimental assays require different decision thresholds:
- **ELISA assays:** Threshold = 0.5 (Boughter, Jain datasets)
- **PSR assays:** Threshold = 0.5495 (Harvey, Shehata datasets - Novo Nordisk exact parity)

The classifier automatically applies the correct threshold based on assay type.

### 4. **Reproducibility & Validation**

- **Novo Parity:** Reproduces Novo Nordisk's exact confusion matrix [[40, 19], [10, 17]] on Jain dataset
- **10-fold CV:** Stratified cross-validation on training set (67-71% accuracy)
- **Embedding Caching:** SHA-256-keyed cache prevents expensive re-computation
- **Config-Driven:** YAML configs for reproducible experiments

### 5. **Production-Ready Infrastructure**

- **CI/CD:** 5 GitHub Actions workflows (quality gates, tests, Docker, security)
- **Test Coverage:** 90.80% (403 tests: unit, integration, E2E)
- **Type Safety:** 100% type coverage (mypy strict mode)
- **Docker Support:** Dev + prod containers for reproducible environments
- **Code Quality:** ruff (linting + formatting), bandit (security scanning)

---

## Technology Stack

### Machine Learning

| Component | Library | Version |
|-----------|---------|---------|
| Protein Language Model | ESM-1v (HuggingFace) | `facebook/esm1v_t33_650M_UR90S_1` |
| Classifier | scikit-learn | LogisticRegression |
| Embeddings | PyTorch | CPU, CUDA, MPS support |
| CDR Annotation | ANARCI | IMGT numbering scheme |

### Development Tools

| Tool | Purpose |
|------|---------|
| **uv** | Fast Python package manager |
| **pytest** | Test framework (unit/integration/e2e) |
| **mypy** | Static type checking (strict mode) |
| **ruff** | Linting + formatting (replaces black, isort, flake8) |
| **pre-commit** | Git hooks for code quality |
| **Docker** | Containerized environments |
| **GitHub Actions** | CI/CD (5 workflows) |
| **Codecov** | Coverage tracking |

### Data Processing

- **pandas:** DataFrame manipulation
- **numpy:** Numerical operations
- **openpyxl/xlrd:** Excel file parsing (dataset preprocessing)
- **PyYAML:** Config file management

---

## Quick Navigation

### 👤 For Users (Running the Pipeline)

- **Installation:** See [Installation Guide](user-guide/installation.md) *(pending Phase 3)*
- **Quick Start:** See [Getting Started](user-guide/getting-started.md) *(pending Phase 3)*
- **Training Models:** See [Training Guide](user-guide/training.md) *(pending Phase 3)*
- **Testing Models:** See [Testing Guide](user-guide/testing.md) *(pending Phase 3)*

**Temporary (until Phase 3):** See root `README.md` for installation and basic usage.

### 👨‍💻 For Developers (Contributing)

- **Architecture Deep Dive:** See [Developer Guide](developer-guide/architecture.md) *(pending Phase 4)*
- **Development Workflow:** See [Workflow Guide](developer-guide/development-workflow.md) *(pending Phase 4)*
- **Testing Strategy:** See [Testing Guide](developer-guide/testing-strategy.md) *(pending Phase 4)*
- **CI/CD Setup:** See [CI/CD Guide](developer-guide/ci-cd.md) *(pending Phase 4)*

**Temporary (until Phase 4):** See `CLAUDE.md` for development commands and architecture.

### 🔬 For Researchers (Validating Methodology)

- **Novo Parity Analysis:** [research/novo-parity.md](research/novo-parity.md)
- **Methodology & Divergences:** [research/methodology.md](research/methodology.md)
- **Assay Thresholds:** [research/assay-thresholds.md](research/assay-thresholds.md)
- **Benchmark Results:** [research/benchmark-results.md](research/benchmark-results.md)

### 📊 For Dataset Users

- **Boughter (Training):** [datasets/boughter/](datasets/boughter/)
- **Jain (Novo Parity):** [datasets/jain/](datasets/jain/)
- **Harvey (Nanobodies):** [datasets/harvey/](datasets/harvey/)
- **Shehata (PSR):** [datasets/shehata/](datasets/shehata/)

---

## Key Results

### Novo Nordisk Parity (Jain Dataset)

**Exact reproduction of published results:**

```
Confusion Matrix: [[40, 19], [10, 17]]
Accuracy: 66.28%
Precision: 0.472 (non-specific)
Recall: 0.630 (non-specific)
F1-Score: 0.540
```

**Methodology:** P5e-S2 subset (86 antibodies), PSR threshold 0.5495, ELISA 1-3 flags removed.

### Cross-Dataset Validation

| Dataset | Sequences | Assay | Accuracy | Notes |
|---------|-----------|-------|----------|-------|
| Boughter (train) | 914 | ELISA | 67-71% | 10-fold CV |
| Jain (test) | 86 | ELISA | 66.28% | Novo parity ✅ |
| Shehata (test) | 398 | PSR | 58.8% | Threshold 0.5495 |
| Harvey (test) | 141,021 | PSR | 61.5-61.7% | Nanobodies |

---

## Scientific Context

### Publication

**Paper:** *Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters*
**Authors:** Sakhnini et al. (2025)
**Method:** ESM-1v embeddings + Logistic Regression
**Key Finding:** Protein language models capture non-specificity signals from sequence alone

### Why This Matters

1. **Speed:** Computational prediction is 100x faster than wet-lab assays
2. **Cost:** Eliminates expensive experimental screening for early-stage candidates
3. **Scale:** Screen millions of sequences in silico before synthesis
4. **Interpretability:** Linear model allows feature importance analysis

### Limitations

- **Accuracy ceiling:** 66-71% accuracy (better than random, but not perfect)
- **Training data:** Limited to 914 labeled sequences (Boughter)
- **Assay dependency:** Models trained on ELISA may not generalize to PSR
- **No mechanistic insight:** Black-box embeddings, opaque feature extraction

See [research/methodology.md](research/methodology.md) for detailed analysis.

---

## License & Citation

### License

This project is licensed under the **Apache License 2.0** - see [LICENSE](../LICENSE) for details.

### Citation

If you use this pipeline in your research, please cite:

**Original Paper:**
```bibtex
@article{sakhnini2025prediction,
  title={Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters},
  author={Sakhnini, et al.},
  journal={TBD},
  year={2025}
}
```

**Datasets:**
- **Boughter:** Boughter et al. (2020) - Training set
- **Jain:** Jain et al. (2017) - Novo parity benchmark
- **Harvey:** Harvey et al. (2022) / Mason et al. (2021) - Nanobodies
- **Shehata:** Shehata et al. (2019) - PSR validation

See [CITATIONS.md](../CITATIONS.md) for full references.

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
**Version:** 2.0.0
