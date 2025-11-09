# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-09

### 🎉 Initial Release

First public release of the antibody non-specificity prediction pipeline implementing the methodology from Sakhnini et al. (2025).

### ✨ Features

**Core ML Pipeline**
- ESM-1v protein language model integration for sequence embeddings
- Logistic regression classifier with sklearn backend
- 10-fold stratified cross-validation training
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Model persistence with pickle serialization
- Embedding caching for performance (SHA-256 hashed paths)

**Dataset Support**
- Boughter et al. 2020 dataset (914 VH sequences, ELISA assay) - Training
- Jain et al. 2017 dataset (86 clinical antibodies) - Test/Novo parity validation
- Harvey et al. 2022 dataset (141k nanobodies, PSR assay) - Test
- Shehata et al. 2019 dataset (398 antibodies, PSR assay) - Test
- Fragment-level predictions (VH, VL, CDRs, FWRs, Full sequences)
- ANARCI annotation with IMGT numbering scheme

**Data Processing**
- Complete preprocessing pipelines for all datasets
- Excel/CSV conversion utilities
- Sequence validation and quality control
- Fragment extraction (16 fragment types per dataset)
- Staged preprocessing with validation scripts

**Command-Line Interface**
- `antibody-train` - Train models with YAML configuration
- `antibody-test` - Evaluate models on test sets
- Flexible configuration system
- Experiment tracking and logging

**Developer Tools**
- 100% type safety with mypy strict mode
- Automated code formatting (ruff)
- Comprehensive linting (ruff)
- Pre-commit hooks for quality enforcement
- `make` commands for all common tasks

**Infrastructure**
- Production Docker images (dev + prod)
- GitHub Actions CI/CD pipeline
- Automated testing (unit + integration + E2E)
- CodeQL security scanning
- Dependency vulnerability audits (pip-audit + Safety)
- Weekly automated dependency updates
- GitHub Container Registry publishing

**Documentation**
- Complete README with quickstart
- Dataset-specific documentation (`docs/`)
- Security remediation plan
- Citation and attribution guide
- Developer workflow documentation

### 🔒 Security

- Bandit static security analysis
- CodeQL code scanning (security-extended queries)
- Automated dependency vulnerability scanning
- Pre-commit security checks
- Pickle usage limited to trusted local artifacts

### 🐛 Bug Fixes

- Fixed uninitialized variable in batch permutation tests
- Removed unused variables in trainer and inference scripts
- Fixed GHCR lowercase repository name requirement
- Aligned mypy exclusions across pyproject.toml and pre-commit config

### 📦 Deliverables

- Docker images: `ghcr.io/the-obstacle-is-the-way/antibody-training:0.1.0`
- Source code: Available on GitHub
- Pre-trained models: Reproducible via training pipeline
- Test coverage: >70% with automated CI enforcement

### 🎯 Reproducibility

All preprocessing scripts, model training, and evaluation procedures are fully reproducible and validated against the Sakhnini et al. (2025) paper benchmarks:

- Jain confusion matrix: [[40, 19], [10, 17]] (66.28% accuracy)
- Shehata PSR threshold: 0.5495 (Novo Nordisk exact parity)
- Harvey nanobody accuracy: 61.5-61.7% on 141k sequences
- Boughter 10-fold CV: 67-71% accuracy

### 📚 References

Implements methodology from:
- Sakhnini et al. (2025) - Prediction of Antibody Non-Specificity using PLMs
- DOI: https://doi.org/10.1101/2025.04.28.650927

### 🔄 Migration Notes

This is the first versioned release. Previous development was unversioned. Starting from v0.1.0, all changes will be tracked in this changelog.

---

## How to Update This Changelog

We use [conventional commits](https://www.conventionalcommits.org/) for all changes:

```bash
# Feature additions
git commit -m "feat: Add support for AbLang embeddings"

# Bug fixes
git commit -m "fix: Correct PSR threshold calculation"

# Documentation
git commit -m "docs: Update installation instructions"

# Performance improvements
git commit -m "perf: Optimize embedding batch processing"

# Breaking changes (for future 1.0+)
git commit -m "feat!: Change classifier API interface"
```

### Automated Changelog Generation (Future)

For future releases, we can automate changelog generation using tools like:
- [git-cliff](https://git-cliff.org/) - Generates changelog from git history
- [standard-version](https://github.com/conventional-changelog/standard-version) - Automates versioning and changelog

To generate changelog automatically:
```bash
# Install git-cliff
cargo install git-cliff

# Generate changelog from commits
git cliff --tag v0.2.0 --output CHANGELOG.md
```

For now, we maintain this changelog manually to ensure high-quality release notes.

---

[0.1.0]: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/releases/tag/v0.1.0
