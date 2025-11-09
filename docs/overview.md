# System Overview

**Antibody Non-Specificity Prediction Pipeline using ESM-1v Protein Language Models**

This document provides a high-level overview of the system architecture, capabilities, and technology stack. Start here if you're new to the pipeline.

---

## Problem Statement

### What is Antibody Non-Specificity?

**Non-specific binding** (also called **polyreactivity**) occurs when therapeutic antibodies bind to unintended targets beyond their designed antigen. This can lead to:

- **Faster clearance** from the body (reduced efficacy)
- **Unwanted side effects** (safety concerns)
- **Failed clinical trials** (high development costs)

Predicting non-specificity from sequence alone is critical for early-stage drug development, enabling researchers to identify problematic candidates before expensive in vivo testing.

### Why Computational Prediction?

Traditional experimental assays (ELISA, PSR) are:
- ❌ **Expensive** - Require protein expression and lab resources
- ❌ **Time-consuming** - Weeks to months per antibody
- ❌ **Low throughput** - Cannot screen millions of candidates

**This pipeline enables:**
- ✅ **Fast prediction** - Seconds per antibody sequence
- ✅ **High throughput** - Millions of sequences (e.g., Harvey: 141k nanobodies)
- ✅ **Early screening** - Filter candidates before synthesis

---

## Solution Architecture

### High-Level Pipeline

```
Antibody Sequence (VH/VL)
        ↓
    ESM-1v Model (Protein Language Model)
        ↓
    Embeddings (1280-dimensional vector)
        ↓
    Logistic Regression Classifier
        ↓
    Prediction (Specific vs Non-Specific)
```

### Core Components

1. **ESM-1v (Evolutionary Scale Modeling)**
   - Pre-trained transformer model from Meta AI / FAIR
   - Trained on 250M protein sequences (UniRef dataset)
   - Captures evolutionary and structural information
   - Generates 1280-dimensional embeddings per sequence

2. **Embedding Extraction** (`src/antibody_training_esm/core/embeddings.py`)
   - Batch processing with GPU acceleration (CUDA/MPS)
   - Mean-pooling of last hidden states (attention-weighted)
   - Caching for fast hyperparameter sweeps (SHA-256 hashed)

3. **Binary Classifier** (`src/antibody_training_esm/core/classifier.py`)
   - Logistic Regression on embeddings (sklearn)
   - Assay-specific thresholds (ELISA: 0.5, PSR: 0.5495)
   - Dual API (dict-based legacy + sklearn kwargs)

4. **Training Pipeline** (`src/antibody_training_esm/core/trainer.py`)
   - 10-fold stratified cross-validation
   - Train on Boughter (914 VH sequences)
   - Test on Jain/Harvey/Shehata (hold-out sets)
   - Metrics: accuracy, precision, recall, F1, ROC-AUC

5. **CLI Tools** (`src/antibody_training_esm/cli/`)
   - `antibody-train` - Train models from YAML configs
   - `antibody-test` - Test models on new datasets
   - `antibody-preprocess` - Dataset preprocessing utilities

---

## Key Capabilities

### Training & Evaluation

- ✅ **Train on Boughter dataset** - 914 VH sequences, ELISA assay
- ✅ **10-fold cross-validation** - Robust performance estimation
- ✅ **Multiple test sets** - Jain (86), Harvey (141k), Shehata (398)
- ✅ **Fragment-level predictions** - VH, VL, CDRs, FWRs, Full sequences

### Reproducibility & Validation

- ✅ **Novo Nordisk exact parity** - [[40, 19], [10, 17]] confusion matrix match
- ✅ **Assay-specific thresholds** - ELISA (0.5) vs PSR (0.5495)
- ✅ **Embedding caching** - SHA-256 hashed, invalidates on model/data changes
- ✅ **Pinned model revisions** - HuggingFace revision parameter for reproducibility

### Infrastructure

- ✅ **Docker deployment** - Dev + prod containers
- ✅ **CI/CD pipeline** - 5 workflows (quality, tests, security, benchmarks)
- ✅ **90.79% test coverage** - 410 tests across unit/integration/e2e
- ✅ **100% type safety** - mypy --strict on core pipeline

---

## Technology Stack

### Core ML Framework

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Protein Language Model | ESM-1v (transformers) | Sequence → embeddings |
| Classifier | scikit-learn (LogisticRegression) | Embeddings → predictions |
| Sequence Annotation | ANARCI (IMGT numbering) | CDR/FWR extraction |
| Evaluation | sklearn.metrics | Performance metrics |

### Development Tools

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Package Manager | uv | Fast dependency resolution |
| Linter/Formatter | ruff | 10-100x faster than black/flake8 |
| Type Checker | mypy --strict | 100% type safety |
| Test Framework | pytest | Unit/integration/e2e tests |
| Security Scanner | bandit + CodeQL | SAST + dependency audits |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Python | 3.12 | Core runtime |
| GPU Support | CUDA / MPS | Accelerated inference |
| Containerization | Docker | Reproducible environments |
| CI/CD | GitHub Actions | 5 workflows (quality, tests, security, benchmarks) |
| Coverage Tracking | Codecov | 90.79% enforced |

---

## Datasets

### Training Dataset

**Boughter et al. (2020)** - 914 VH sequences, ELISA polyreactivity assay
- Used for 10-fold cross-validation
- Primary training data

### Test Datasets

**Jain et al. (2017)** - 86 clinical antibodies, ELISA (Novo parity benchmark)
- Primary test set
- Exact Novo Nordisk confusion matrix match: [[40, 19], [10, 17]]

**Harvey et al. (2022)** - 141,021 nanobody sequences, PSR assay
- Large-scale nanobody test set
- Tests scalability and assay generalization

**Shehata et al. (2019)** - 398 human antibodies, PSR assay
- Cross-assay validation (PSR vs ELISA)
- Tests assay-specific threshold handling

---

## Performance

### Novo Nordisk Parity (Jain Dataset)

**Result:** ✅ **EXACT MATCH** to Novo Nordisk's published confusion matrix

```
Confusion Matrix:
[[40, 19],    ← True Negatives: 40, False Positives: 19
 [10, 17]]    ← False Negatives: 10, True Positives: 17

Accuracy: 66.28%
```

### Cross-Dataset Validation

| Dataset | Size | Assay | Performance |
|---------|------|-------|-------------|
| Boughter (10-fold CV) | 914 | ELISA | ~71% accuracy |
| Jain | 86 | ELISA | 66.28% (Novo parity) |
| Harvey | 141,021 | PSR | Benchmark in progress |
| Shehata | 398 | PSR | Cross-validation complete |

---

## Quick Navigation

### For Users (Running the Pipeline)

**New to the pipeline?** Start here:

1. [Installation Guide](user-guide/installation.md) (pending Phase 3)
2. [Quick Start Tutorial](user-guide/getting-started.md) (pending Phase 3)
3. [Training Models](user-guide/training.md) (pending Phase 3)
4. [Testing on New Datasets](user-guide/testing.md) (pending Phase 3)

### For Developers (Contributing Code)

**Contributing to the codebase?** Start here:

1. [Architecture Deep Dive](developer-guide/architecture.md) (pending Phase 4)
2. [Development Workflow](developer-guide/development-workflow.md) (pending Phase 4)
3. [Testing Strategy](developer-guide/testing-strategy.md) (pending Phase 4)
4. [CI/CD Pipeline](developer-guide/ci-cd.md) (pending Phase 4)

### For Researchers (Validating Methodology)

**Validating scientific reproducibility?** Start here:

1. [Methodology & Divergences](research/METHODOLOGY_AND_DIVERGENCES.md)
2. [Novo Parity Analysis](research/NOVO_PARITY_ANALYSIS.md)
3. [Assay-Specific Thresholds](research/ASSAY_SPECIFIC_THRESHOLDS.md)
4. [Benchmark Results](research/BENCHMARK_TEST_RESULTS.md)

### For Data Scientists (Working with Datasets)

**Processing new datasets?** Start here:

1. [Boughter Dataset](datasets/boughter/) - Training data
2. [Jain Dataset](datasets/jain/) - Novo parity test set
3. [Harvey Dataset](datasets/harvey/) - Nanobody test set
4. [Shehata Dataset](datasets/shehata/) - PSR cross-validation

---

## Project Status

**Current Version:** v0.1.0 (Initial public release)
**Production Branch:** `leroy-jenkins/full-send` (stable, production-ready)
**Active Development:** `docs/canonical-structure` (documentation reorganization)
**CI Status:** ✅ All quality gates passing
**Test Coverage:** 90.79% (enforced, 70% minimum)
**Type Safety:** 100% (mypy --strict on core pipeline)

---

## References

### Paper Implementation

This pipeline implements the methodology from:

**Sakhnini et al. (2025)** - Novo Nordisk & University of Cambridge
> Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters

### Model Source

**ESM-1v** - Meta AI / FAIR (Facebook AI Research)
> Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences

### Dataset Citations

See [`CITATIONS.md`](../CITATIONS.md) for complete bibliographic information on all datasets.

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
