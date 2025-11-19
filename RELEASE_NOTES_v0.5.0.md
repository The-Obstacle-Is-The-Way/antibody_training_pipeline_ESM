# Release Notes: v0.5.0 - Production Inference Pipeline

**Release Date:** 2025-11-19
**Status:** 🚀 Production Ready
**Coverage:** 90.42% (up from 89.01%)
**Tests:** 476 passing (up from 468)
**Commits Since v0.4.0:** 256

---

## 🎯 Executive Summary

**v0.5.0** delivers the **production-ready inference pipeline** with comprehensive CLI tooling, testing improvements, and documentation overhauls. This release completes the "predict → test → train" workflow trilogy, making the antibody non-specificity prediction system fully operational for end users.

### 🔥 Headliners

1. **antibody-predict CLI** - Complete inference interface with validation (100% coverage)
2. **Modular Testing Pipeline** - CLI test.py refactored from 872-line monolith → 6 focused modules (83.8% size reduction)
3. **Column Name Flexibility** - Support for custom sequence/label columns in all CLIs
4. **Comprehensive Documentation** - 99 documentation updates, 3 completion reports
5. **Type Safety** - 100% mypy strict compliance maintained across all new code

---

## ✨ Major Features

### 1. Production Inference CLI (`antibody-predict`)

**Status:** ✅ **PRODUCTION READY**

Complete CLI for predicting antibody non-specificity from sequences.

**Features:**
- ✅ **Input Validation:** Checks for missing files, invalid columns, classifier paths
- ✅ **Flexible Columns:** `--sequence-column` and `--label-column` overrides
- ✅ **Assay-Specific Thresholds:** PSR (0.5495) and ELISA (0.5) support via `--assay-type`
- ✅ **Custom Thresholds:** Manual `--threshold` override (0.0-1.0)
- ✅ **CSV I/O:** Preserves all original columns + adds `prediction` and `probability`
- ✅ **Error Handling:** Clear, actionable error messages with usage examples
- ✅ **Resource Optimization:** Reuses embedder from classifier (avoids 2x 650MB model loading)

**Usage:**
```bash
uv run antibody-predict \
    input_file=sequences.csv \
    output_file=predictions.csv \
    classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

**Documentation:**
- `INFERENCE_GUIDE.md` - Comprehensive 112-line user guide
- `README.md` - Updated with prediction CLI section
- `docs/needs_integration/INFERENCE_COMPLETION_REPORT.md` - Full validation report

**Test Coverage:**
- ✅ **10 new tests** (6 unit + 4 core logic)
- ✅ **100% coverage** for `predict.py` (was 0%)
- ✅ **87.5% coverage** for `prediction.py` core logic
- ✅ **E2E test** (gated behind `RUN_PREDICT_CLI_E2E=1` to avoid CI model downloads)

**New Files:**
- `src/antibody_training_esm/cli/predict.py` (57 lines)
- `src/antibody_training_esm/core/prediction.py` (203 lines)
- `src/antibody_training_esm/conf/predict.yaml` (16 lines)
- `tests/unit/cli/test_predict.py` (122 lines, 6 tests)
- `tests/unit/core/test_prediction.py` (136 lines, 4 tests)
- `tests/e2e/test_predict_cli.py` (124 lines, gated E2E test)
- `INFERENCE_GUIDE.md` (111 lines)

**Commits:**
- `ed8b868` - feat: enhance prediction functionality with customizable sequence column and assay-specific thresholds
- `0dded1f` - refactor: implement Predictor class and enhance tests for clean architecture
- `c27a8b8` - test: add unit tests for predict CLI validation with type hints
- `43b5281` - docs: update README with classifier setup instructions

---

### 2. Modular Testing Pipeline Refactor

**Status:** ✅ **PRODUCTION READY**

Transformed CLI `test.py` from 872-line monolith into clean 6-module architecture.

**Impact:**
- ✅ **83.8% size reduction** in main file (872 → 141 lines)
- ✅ **Single Responsibility Principle** - Each module has one clear purpose
- ✅ **Zero circular dependencies** - Clean acyclic dependency graph (DAG)
- ✅ **100% backward compatible** - All 34 CLI tests pass
- ✅ **Type safety maintained** - mypy strict mode clean

**New Architecture:**
```
src/antibody_training_esm/cli/
├── test.py                       # 141 lines - CLI argument parser (thin wrapper)
└── testing/                      # Modular testing package
    ├── __init__.py              #   1 line  - Package init
    ├── config.py                #  62 lines - TestConfig dataclass + YAML loading
    ├── data.py                  #  73 lines - Dataset loading + validation
    ├── evaluation.py            # 141 lines - Metrics calculation + assay detection
    ├── tester.py                # 383 lines - Model orchestration (core logic)
    └── visualization.py         # 127 lines - Plotting + result serialization
```

**Benefits:**
- Easier testing (each module independently testable)
- Better maintainability (find bugs faster)
- Extensibility (add metrics/visualizations without touching core logic)
- Cleaner imports (no 872-line file in diffs)

**Documentation:**
- `docs/needs_integration/CLI_TEST_REFACTOR_VALIDATION.md` - 328-line validation report

**Commits:**
- `5414b3a` - refactor: modularize testing pipeline by splitting test.py into multiple components
- `883ab28` - enhance: add metrics_list parameter to evaluate_pretrained function

---

### 3. Column Name Flexibility

**Status:** ✅ **PRODUCTION READY**

All CLIs now support custom sequence and label column names.

**Problem Solved:**
- Canonical files use `vh_sequence` (original dataset column names)
- Fragment files use `sequence` (standardized column names)
- Users couldn't easily use canonical files with CLI direct flags

**Solution:**
```bash
# Testing CLI - custom columns
uv run antibody-test \
  --model model.pkl \
  --data data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv \
  --sequence-column vh_sequence \
  --label-column label

# Prediction CLI - custom columns
uv run antibody-predict \
  input_file=data.csv \
  sequence_column=vh_sequence \
  classifier.path=model.pkl
```

**Documentation:**
- `docs/needs_integration/DATASET_COLUMN_NAMING_INVESTIGATION.md` - 402-line design analysis

**Commits:**
- `5515795` - feat: add sequence and label column arguments to CLI test
- `e52806c` - docs: update DATASET_COLUMN_NAMING_INVESTIGATION.md with completed tasks

---

### 4. Logging Migration

**Status:** ✅ **COMPLETE**

Migrated all `print()` statements to proper `logging` module usage.

**Changes:**
- ✅ Preprocessing scripts now use `logging.info()` instead of `print()`
- ✅ Mypy/ruff compliance maintained
- ✅ Log levels configurable (INFO, DEBUG, WARNING, ERROR)
- ✅ Better production observability

**Commits:**
- `a5640f5` - refactor: complete migration from print() to logging in preprocessing scripts
- `cfdaded` - refactor: finalize logging migration (fix mypy/ruff errors)

---

## 🐛 Bug Fixes

### Critical Fixes

1. **Inference CLI Validation** - Missing classifier path now raises clear error with usage example
2. **Double ESM Loading** - Predictor reuses classifier's embedder (saves 650MB memory)
3. **E2E Test Downloads** - Gated behind env var to prevent CI hangs on fresh systems
4. **Gap Character Handling** - Documentation corrected (gaps NOT supported, was falsely claimed)
5. **Hydra CWD Switching** - Fixed via `hydra.job.chdir: False` in predict.yaml

### Other Fixes

- Fixed test markers (@pytest.mark.e2e, @pytest.mark.slow) for proper test selection
- Fixed .gitignore to exclude coverage.json (build artifact)
- Fixed sys.path hacks removed from tests
- Fixed integration test markers added to embedding compatibility tests

**Commits:**
- `43b5281` - docs: update README with classifier setup instructions and improve error handling in prediction CLI
- `182ce96` - chore: remove coverage.json from git tracking and add to .gitignore
- `004c08c` - fix: remove sys.path hack from Harvey PSR threshold test
- `580af5e` - test: add missing @pytest.mark.integration to embedding compatibility tests

---

## 🔧 Improvements

### Developer Experience

1. **Test Hygiene** - Proper markers (@slow, @e2e, @integration) for selective test runs
2. **Type Safety** - 100% mypy --strict compliance maintained
3. **Code Quality** - Zero ruff warnings across all new code
4. **Modular Design** - CLI components split into focused modules

### Documentation (99 Updates!)

**New Documentation:**
- `INFERENCE_GUIDE.md` - 111-line user guide for prediction CLI
- `docs/needs_integration/INFERENCE_COMPLETION_REPORT.md` - 44-line validation report
- `docs/needs_integration/CLI_TEST_REFACTOR_VALIDATION.md` - 328-line refactor validation
- `docs/needs_integration/DATASET_COLUMN_NAMING_INVESTIGATION.md` - 402-line design analysis
- `docs/needs_integration/SPEC_SHEET.md` - 79-line specification

**Updated Documentation:**
- `README.md` - Prediction CLI section added, updated installation/usage
- `CLAUDE.md` - Updated with inference CLI patterns
- `USAGE.md` - Updated with prediction examples
- `GEMINI.md` - Agent guidance for inference work
- `AGENTS.md` - Multi-agent workflow documentation

### Testing Infrastructure

**New Tests:**
- `tests/unit/cli/test_predict.py` - 6 tests for CLI validation
- `tests/unit/core/test_prediction.py` - 4 tests for Predictor class
- `tests/e2e/test_predict_cli.py` - End-to-end CLI test (gated)

**Test Coverage:**
- Before: 89.01% (468 tests)
- After: 90.42% (476 tests)
- **Improvement:** +1.41% coverage, +8 tests

---

## 📦 Files Changed

**Summary:**
- **473 files changed**
- **820,059 insertions**
- **230,640 deletions**
- **Net: +589,419 lines** (mostly documentation, validation reports, test files)

**Key Additions:**
- Inference pipeline (3 source files, 382 lines)
- Testing refactor (6 modules, 787 lines)
- Unit tests (2 files, 258 lines)
- Documentation (5 reports, 966 lines)

---

## 🚀 Migration Guide

### From v0.4.0 to v0.5.0

**100% BACKWARD COMPATIBLE** - No breaking changes!

### New CLI Available

```bash
# Prediction (NEW in v0.5.0)
uv run antibody-predict \
    input_file=sequences.csv \
    classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl

# Testing (Enhanced in v0.5.0)
uv run antibody-test \
    --model model.pkl \
    --data data.csv \
    --sequence-column vh_sequence \  # NEW
    --label-column label              # NEW

# Training (Unchanged)
uv run antibody-train  # Uses Hydra config
```

### Configuration Changes

**No action required** - All defaults preserved.

**Optional enhancements:**
- Use `sequence_column` argument in prediction CLI for custom columns
- Use `assay_type` argument for PSR/ELISA threshold calibration

---

## ✅ Validation & Quality Metrics

### Test Results

```bash
$ uv run pytest
===== 476 passed, 4 skipped, 1 warning in 83.57s =====
```

**Coverage:**
- Overall: 90.42%
- predict.py: 100% (was 0%)
- prediction.py: 87.5%

**Type Safety:**
```bash
$ make typecheck
✅ Success: no issues found in 97 source files
```

**Code Quality:**
```bash
$ make lint
✅ All checks passed!
```

**Security:**
```bash
$ uv run bandit -r src/
✅ No issues identified
```

---

## 🎯 Production Readiness Checklist

- [x] All tests pass (476/476)
- [x] Type checking clean (mypy --strict)
- [x] Linting clean (ruff)
- [x] Security scan clean (bandit)
- [x] Documentation complete (INFERENCE_GUIDE.md + README.md)
- [x] Test coverage > 90% (90.42%)
- [x] Backward compatible (no breaking changes)
- [x] CI/CD passing (GitHub Actions)
- [x] Release notes complete (this document)

**Status:** ✅ **READY FOR PRODUCTION DEPLOYMENT**

---

## 🔜 Next Steps

### Immediate (Post-Release)

1. **Deploy to HuggingFace Spaces** - Gradio wrapper for web UI
2. **Model Registry** - Auto-download checkpoints from GitHub Releases
3. **Docker Image** - Pre-built container with ESM models cached

### Short-Term (v0.6.0)

1. **Batch Prediction API** - Process thousands of sequences efficiently
2. **Confidence Intervals** - Bootstrap uncertainty quantification
3. **Feature Importance** - SHAP values for sequence interpretability

### Long-Term (v1.0.0)

1. **Biophysical Features** - Add pI, net charge, hydrophobicity
2. **Multi-Model Ensembles** - Combine ESM-1v + ESM-2 + AntiBERTy
3. **Web Dashboard** - Interactive visualization of prediction results

---

## 🙏 Contributors

**Primary Development:**
- **@the-obstacle-is-the-way** - Architecture, training pipeline, validation
- **Gemini Agent** - CLI refactoring, inference implementation, test generation
- **Claude Code (Sonnet 4.5)** - Code review, validation, documentation

**Special Thanks:**
- **Novo Nordisk Team** - Original methodology (Sakhnini et al. 2025)
- **Meta AI** - ESM-1v protein language model
- **Hydra Team** - Configuration framework

---

## 📚 References

**Paper:**
> Sakhnini, L.I., et al. (2025). Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. *bioRxiv*. https://doi.org/10.1101/2025.04.28.650927

**Datasets:**
- Boughter et al. (2020) - Training set (914 VH sequences)
- Jain et al. (2017) - Test set (86 clinical antibodies, Novo parity)
- Harvey et al. (2022) - Test set (141k nanobodies)
- Shehata et al. (2019) - Test set (398 antibodies, PSR validation)

---

**Full Changelog:** See [CHANGELOG.md](CHANGELOG.md)
**Installation:** See [README.md](README.md)
**User Guide:** See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
**Developer Docs:** See `docs/developer-guide/`
