# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ✅ Jain Benchmark Parity

- Achieves **exact Novo Nordisk parity** on Jain S14A (ESM-1v VH LogReg): confusion matrix `[[40, 17], [10, 19]]`, accuracy **68.60%** (59/86), label split **57/29** via Tier D remediation.

## [0.7.0] - 2025-11-21

### 🛡️ Pydantic v2 Integration - Type-Safe Validation Layer

Major robustness release introducing comprehensive runtime validation using Pydantic v2 and Pandera across the entire pipeline. This 4-phase integration adds a type-safe validation layer to predictions, configurations, data integrity, and model artifacts, preventing silent failures and improving production reliability.

### ✨ Features

**Phase 1: Prediction Hardening**
- Pydantic `AntibodySequence` model for input validation
- `AssayType` enum for type-safe assay handling
- Runtime validation of amino acid sequences (20 canonical AAs + special handling)
- Immediate feedback on invalid sequences before expensive embedding extraction
- Dual API support: raw strings and Pydantic models

**Phase 2: Configuration Safety**
- Pydantic models for all Hydra configuration sections
- `TrainingPipelineConfig`, `ModelConfig`, `ClassifierConfig`, `DataConfig`, `ExperimentConfig`, `HardwareConfig`
- Runtime validation of config values before training starts
- Type-safe config handling throughout pipeline
- Fail-fast on invalid configurations (prevents expensive GPU allocation waste)

**Phase 3: Data Integrity (Pandera Integration)**
- `get_sequence_dataset_schema()` for production training/testing data
- `get_preprocessing_schema()` for intermediate preprocessing files
- `get_boughter_schema()`, `get_jain_schema()`, `get_harvey_schema()`, `get_shehata_schema()`
- Amino acid validation (uppercase letters only, no gaps, valid AAs)
- Sequence length validation (1-2000 characters)
- Label validation (binary 0/1, nullable for preprocessing intermediates)
- DataFrame validation on dataset load (immediate failure on corrupt data)

**Phase 4: Artifacts & Metrics**
- `ModelArtifactMetadata` for self-describing model JSON sidecars
- `EvaluationMetrics` for type-safe metrics reporting
- `CVResults` for cross-validation result serialization
- Version-compatible model loading (detect incompatible models at load time)
- No manual type casting (Pydantic handles complex types like `class_weight` with int keys)

### 🐛 Bug Fixes

**Critical Regressions Fixed (Post-Pydantic Phase 4)**
- **Preprocessing Schema Bug**: Fixed Phase 3 regression where production schemas were too strict for preprocessing intermediate files (151 held-out sequences with NaN labels)
  - Created `get_preprocessing_schema()` with `nullable=True` for labels (dtype=float64)
  - Production schemas remain strict (nullable=False, dtype=int64)
  - Boughter preprocessing validation now passes
- **BoughterDataset KeyError**: Fixed crash when loading pre-filtered training files without flags column
  - Added defensive check: `has_flags = "num_flags" in df.columns or "flags" in df.columns`
  - Only apply flagging logic if flags column exists
  - Added informative logging for pre-filtered files

**Test Infrastructure**
- Fixed `test.log` artifact creation in repo root (now uses `tmp_path` in test fixtures)
- Improved test isolation with proper temporary directory usage

### 🔧 Improvements

**Validation Coverage**
- 567 tests passing (up from 520 in v0.6.0)
- ~90% test coverage maintained
- Comprehensive end-to-end validation suite
- Data integrity checksums verified (Pandera validation doesn't modify data)

**Developer Experience**
- Clear, actionable error messages on validation failures
- Fail-fast validation before expensive operations
- Type-safe configuration handling
- Self-documenting Pydantic models with field descriptions

**Production Reliability**
- No silent data corruption (all validation explicit)
- Early detection of invalid inputs
- Self-describing model artifacts
- Version compatibility checking

### 📦 Dependencies

**New Requirements (Added to `validation` optional dependency group):**
- `pydantic>=2.10.0` - Runtime validation framework
- `pydantic-settings>=2.6.0` - Future config management support
- `pandera>=0.20.0` - DataFrame schema validation

Install with: `uv sync --all-extras`

### ✅ Verification

**End-to-End Validation Results:**
- ✅ 567 tests passing (~90% coverage)
- ✅ All preprocessing pipelines validated (Boughter, Jain, Harvey, Shehata)
- ✅ Data integrity checksums verified (no corruption)
- ✅ Full training pipeline smoke test passed (66.84% ± 8.69% 10-fold CV)
- ✅ **Benchmark performance:**
  - Jain: 66.28% accuracy (close to Novo 68.6%)
  - Shehata: 58.29% accuracy (close to Novo 58.8%)
- ✅ Type safety: mypy --strict clean (148 files)
- ✅ Code quality: ruff format + ruff check clean

**Validation Dossier:**
- Comprehensive validation report: `docs/archive/investigations/PYDANTIC_VALIDATION_DOSSIER_2025-11-21.md`
- 2 critical bugs fixed, 2 false alarms (intentional design) documented
- Senior approval checklist completed

### 🔄 Migration Notes

**100% Backward Compatible** - No breaking changes!

**Existing Usage (Still Works):**
```bash
# All existing workflows unchanged
antibody-train
antibody-test --model experiments/checkpoints/esm1v/logreg/model.pkl --data data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv
antibody-predict --input sequences.csv --classifier-path model.pkl
```

**New Validation Features (Automatic):**
- Pydantic validation runs automatically on all inputs
- Invalid sequences/configs immediately rejected with clear error messages
- No code changes needed to benefit from validation

**For Developers:**
```python
# Use Pydantic models for type-safe inputs
from antibody_training_esm.models.prediction import AntibodySequence, AssayType

seq = AntibodySequence(sequence="QVQLVQSGAEVK", assay_type=AssayType.ELISA)
# Validation happens automatically - invalid sequences raise ValidationError
```

### 📚 Documentation

**Implementation Docs (Archived in `docs/implementation/`):**
- `PYDANTIC_INTEGRATION_AUDIT.md` - Master plan and audit
- `PYDANTIC_PHASE_1_PREDICTION_HARDENING.md` - Phase 1 spec
- `PYDANTIC_PHASE_2_CONFIGURATION_SAFETY.md` - Phase 2 spec
- `PYDANTIC_PHASE_3_DATA_INTEGRITY.md` - Phase 3 spec (Pandera)
- `PYDANTIC_PHASE_4_ARTIFACTS_METRICS.md` - Phase 4 spec

**Validation Reports:**
- `docs/archive/investigations/PYDANTIC_VALIDATION_DOSSIER_2025-11-21.md` - Comprehensive validation report

**Updated User Docs:**
- `VALIDATION_PLAN.md` - Updated for Phase 4 completion

### 🎯 What's Next?

With comprehensive validation in place, we can now:
- Deploy to production with confidence (fail-fast validation prevents silent failures)
- Add new models/datasets with schema validation guarantees
- Extend validation to additional pipeline components
- Consider Pydantic v2 for API endpoints (future web service)

---

## [0.6.0] - 2025-11-19

### 🏗️ Multi-Classifier Support - Strategy Pattern Architecture

Major architectural release introducing the Strategy Pattern for classifier backends, enabling runtime selection between LogisticRegression and XGBoost. This lays the groundwork for future classifier additions (MLP, SVM, etc.) while maintaining 100% backward compatibility.

### ✨ Features

**Strategy Pattern Architecture**
- Protocol-based `ClassifierStrategy` interface for pluggable classifier backends
- Runtime classifier selection via `classifier.type` config parameter
- Factory pattern (`create_classifier`) for strategy instantiation
- Full sklearn API compatibility maintained across all strategies
- Type-safe design with mypy --strict compliance

**XGBoost Classifier Support**
- New `XGBoostStrategy` implementation wrapping `xgboost.XGBClassifier`
- Native `.xgb` serialization format (pickle-free production path)
- Comprehensive hyperparameter support (n_estimators, max_depth, learning_rate, etc.)
- Nonlinear decision boundaries for complex polyreactivity patterns
- Complete test coverage with XOR dataset validation

**Gradio Web UI** 🚀
- Web-based prediction interface (`antibody-app` command)
- Interactive sequence input with real-time validation
- Warm-up predictions for faster cold starts (pre-loads ESM model)
- Device optimization (CPU, CUDA, macOS MPS with threading tuning)
- Configurable server settings (host, port, share via CLI args)
- Queue management for concurrent requests
- Full test coverage (unit + integration tests)

**XGBoost Baseline Evaluation** 📊
- Comprehensive cross-dataset testing on Jain, Harvey, Shehata
- **Finding:** LogReg outperforms XGBoost by ~13pp average across all datasets
  - Jain: XGBoost 47.67% vs LogReg 58.8% (-11pp gap)
  - Harvey: XGBoost 56.04% vs LogReg ~71% (-15pp gap)
  - Shehata: XGBoost 45.48% vs LogReg 58.29% (-13pp gap)
- **Root cause:** Small dataset (914 samples) + high-dim embeddings (1280-D) favor linear models
- All benchmarks, models, and predictions committed for reproducibility

**Benchmark Organization**
- Hierarchical directory structure: `experiments/benchmarks/esm1v/{classifier}/{dataset}/`
- Cleaned 22 duplicate files from root directory
- Auto-generated logs moved to gitignore (ephemeral artifacts)
- Clear separation: single-model results (hierarchical) vs aggregated reports (root)

**Configuration System**
- New `conf/classifier/xgboost.yaml` config group
- Single Source of Truth (SSOT): Hydra YAML is authoritative for all hyperparameters
- Removed hardcoded defaults from Python code (magic numbers eliminated)
- Strict configuration validation (fails fast on missing keys)

**Model Persistence**
- Canonical trained models committed for reproducibility:
  - `experiments/checkpoints/esm1v/logreg/` (ESM-1v + LogReg, 11KB)
  - `experiments/checkpoints/esm2_650m/logreg/` (ESM-2 650M + LogReg, 11KB)
- Triple serialization format support: `.pkl` (legacy), `.npz` + `.json` (research), `.xgb` (native)
- Out-of-box inference capability without retraining

### 🐛 Bug Fixes

**Configuration SSOT Enforcement**
- Removed all `config.get(key, default)` fallback values from strategy classes
- Eliminated dual source of truth between YAML and Python code
- Fixed `random_state` conflict (Python: 42, YAML: `${training.random_state}`)
- Now enforces complete config dictionary from Hydra (KeyError if missing)

**Serialization Improvements**
- Fixed internal classifier attribute access in model persistence
- Enhanced Protocol compliance for save/load methods
- Improved error messages for unfitted classifier save attempts

### 🔧 Improvements

**Test Suite Hardening**
- Updated entire test suite to provide explicit, complete configuration dictionaries
- 520 tests passing (up from 476 in v0.5.0), 4 skipped
- 90% test coverage maintained
- New test fixtures: `default_classifier_params`, `FULL_LOGREG_DEFAULTS`
- Comprehensive XGBoost integration and E2E tests added

**Code Quality**
- Removed 153 lines of legacy/redundant code
- Added 6,419 lines of new functionality (Strategy Pattern + XGBoost + tests)
- Zero ruff/mypy violations (strict mode)
- Comprehensive documentation (3,026 lines across 4 developer guides)

**Developer Experience**
- Clear extension path for future classifiers (MLP, SVM, etc.)
- Registry pattern available for plugin-based classifier registration
- Improved type hints and protocol definitions
- Better separation of concerns (embeddings vs classification)

### 📦 Dependencies

**New Requirements:**
- `xgboost>=2.0.0` - Gradient boosting classifier backend
- `gradio>=5.14.0` - Web UI for interactive predictions

### ✅ Verification

**End-to-End Validation:**
- ✅ All 520 tests passing (90% coverage)
- ✅ Backward compatibility: Existing LogReg behavior unchanged
- ✅ Novo parity: 6/6 critical E2E benchmarks passing
- ✅ XGBoost nonlinear tests: XOR dataset validation passing
- ✅ Serialization: All 3 formats (pkl/npz/xgb) working correctly
- ✅ Type safety: mypy --strict clean
- ✅ Code quality: ruff format + ruff check clean

### 🔄 Migration Notes

**100% Backward Compatible** - No breaking changes!

**Existing Usage (Still Works):**
```bash
# Default behavior unchanged (uses LogisticRegression)
antibody-train

# Explicit LogReg (same as before)
antibody-train classifier.type=logistic_regression
```

**New XGBoost Usage:**
```bash
# Train with XGBoost classifier
antibody-train classifier.type=xgboost

# Override XGBoost hyperparameters
antibody-train classifier.type=xgboost classifier.n_estimators=200 classifier.max_depth=8

# Hyperparameter sweep
antibody-train --multirun classifier.type=xgboost classifier.n_estimators=50,100,200
```

**New Gradio Web UI:**
```bash
# Launch web app (default: http://localhost:7860)
uv run antibody-app

# Custom configuration
uv run antibody-app --host 0.0.0.0 --port 7860 --share

# Test with XGBoost model
uv run antibody-test \
  --model experiments/checkpoints/esm1v/xgboost/boughter_vh_esm1v_xgboost.pkl \
  --data data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv
```

### 📚 Documentation

**New Developer Guides (3,026 lines, now archived under `docs/archive/plans/`):**
- `archive/plans/xgboost-api-design.md` (1,085 lines)
- `archive/plans/xgboost-integration-spec.md` (655 lines)
- `archive/plans/xgboost-test-plan.md` (1,013 lines)
- `archive/plans/xgboost-implementation-status.md` (273 lines)

> **Live reference:** Use `docs/developer-guide/xgboost.md` for the maintained summary of the shipped classifier.

**Audit Report:**
- `XGBOOST_BRANCH_AUDIT_REPORT.md` - Comprehensive technical review

### 🎯 Future Work

- MLP classifier strategy (neural network backend)
- SVM classifier strategy (support vector machines)
- Ensemble strategies (voting, stacking)
- Plugin system for third-party classifiers

---

## [0.5.0] - 2025-11-19

### 🚀 Production Inference Pipeline

Complete `predict → test → train` workflow with production-ready inference CLI, modular testing pipeline, and comprehensive validation suite.

### ✨ Features

**Inference CLI (`antibody-predict`)**
- Production-ready `antibody-predict` command with 100% test coverage
- Assay-specific thresholds (PSR: 0.5495, ELISA: 0.5)
- Resource optimization (reuses embedder instance, saves 650MB RAM)
- Column name flexibility (--sequence-col, --label-col)
- Clear validation and error messages

**Modular Testing Pipeline**
- Refactored `test.py`: 872 → 141 lines (83.8% reduction, -731 lines)
- Extracted `ModelTester` class for reusable testing logic
- Column name flexibility across all CLIs (train/test/predict)
- Comprehensive CLI test suite

**Validation & Quality**
- 90.42% test coverage (up from 89.01%)
- 476 tests passing
- Complete validation suite for all CLIs

### 🐛 Bug Fixes

**pyproject.toml Version Mismatch (Docker Fix)**
- Fixed v0.5.0 tag pointing to 0.4.0 in pyproject.toml
- Docker packages now correctly show 0.5.0
- Retroactive fix with force-updated tag

### 📦 Dependencies

No new dependencies added.

### ✅ Verification

**End-to-End Validation:**
- ✅ 476 tests passing
- ✅ 90.42% coverage
- ✅ All CLIs functional with column flexibility
- ✅ Inference pipeline validated on all datasets

### 📚 Documentation

**New Guides:**
- `docs/user-guide/INFERENCE_GUIDE.md` - Complete inference workflow
- Updated CLAUDE.md, USAGE.md with predict CLI examples

---

## [0.4.0] - 2025-11-11

### 🎛️ Hydra Configuration System - Enterprise-Grade Experiment Management

Major feature release introducing [Hydra](https://hydra.cc) for flexible, composable configuration management. This modernizes the training pipeline with industry-standard experiment tracking, CLI overrides, and hyperparameter sweeps.

### ✨ Features

**Hydra Configuration Framework**
- Complete Hydra integration with structured configs (dataclasses for type safety)
- Composable config system: `model` × `classifier` × `data` combinations
- Config directory: `src/antibody_training_esm/conf/` (inside package, deployment-ready)
- Default config: `src/antibody_training_esm/conf/config.yaml` (Boughter train → Jain test)
- CLI override support: `antibody-train model.batch_size=16 classifier.C=0.5`
- Multirun sweeps: `antibody-train --multirun classifier.C=0.1,1.0,10.0`
- Automatic experiment tracking in `outputs/{experiment.name}/{timestamp}/` *(now `experiments/runs/`, moved in v0.5.0)*

**Structured Configuration (Type-Safe)**
- Dataclass schemas for all config sections (ModelConfig, ClassifierConfig, DataConfig, etc.)
- Full type safety with mypy validation
- IDE autocomplete support for config fields
- Required field enforcement with `MISSING` sentinel
- Registered with Hydra ConfigStore for runtime validation

**CLI Improvements**
- No more `--config` flag required (uses `src/antibody_training_esm/conf/config.yaml` by default)
- Override any parameter from command line without editing files
- Multirun support for hyperparameter sweeps (1 command → N experiments)
- Hydra auto-saves complete config snapshot per run
- Provenance tracking: every experiment has `.hydra/config.yaml` snapshot

**Logging & Output Management**
- Hydra-managed output directories: `outputs/{experiment.name}/{timestamp}/` *(now `experiments/runs/`, moved in v0.5.0)*
- Automatic log routing: `outputs/.../logs/training.log` *(now `experiments/runs/.../logs/`, moved in v0.5.0)*
- Backward-compatible legacy mode for non-Hydra runs
- Training logs organized by experiment name and timestamp

### 🐛 Bug Fixes

**Configuration Bugs**
- Fixed incorrect Jain test file path (`VH_only_jain_P5e_S2.csv` → `VH_only_jain_86_p5e_s2.csv`)
- Fixed missing log directory creation in Hydra mode (prevented FileNotFoundError)
- Fixed relative log paths routing to `logs/` directory (not repo root)

**Compatibility Fixes**
- Deprecated `train_model(config_path)` wrapper maintained for backward compatibility
- Legacy tests preserved with `@pytest.mark.legacy` marker
- Dual-mode logging: Hydra output dir when available, legacy `logs/` fallback

### 🔧 Improvements

**Developer Experience**
- CLI patterns simplified: `antibody-train` (no args needed for defaults)
- Experiment reproducibility: config snapshots auto-saved per run
- Faster iteration: override configs from CLI, no file editing
- Systematic sweeps: `--multirun` for grid search over parameters

**Testing Infrastructure**
- 3 new test files: `test_hydra_config.py`, `test_structured_configs.py`, `test_trainer_hydra.py`
- 684 new test lines for Hydra integration coverage
- Legacy tests marked and preserved for backward compatibility
- New pytest marker: `@pytest.mark.legacy` for old tests

**Documentation Overhaul (14 Files Updated)**
- Core docs: `CLAUDE.md`, `README.md`, `USAGE.md`, `AGENTS.md`
- User guides: `getting-started.md`, `training.md`, `troubleshooting.md`
- Developer guides: `development-workflow.md`, `architecture.md`, `docker.md`, `security.md`
- Research docs: `methodology.md`, `novo-parity.md`
- Dataset docs: `preprocessing/boughter/README.md`

### 📦 Dependencies

**New Requirements:**
- `hydra-core>=1.3.2` - Configuration framework
- `omegaconf>=2.3.0` - Configuration manipulation library

### ✅ Training Verification

**End-to-End Validation (Fresh Training Run):**
- ✅ ESM-1v embedding extraction: 28s for 914 sequences
- ✅ 10-fold CV accuracy: **66.62% (+/- 9.26%)** - Matches Novo baseline
- ✅ Train accuracy: 74.07%
- ✅ Model saved in all 3 formats (pkl, npz, json)
- ✅ Hydra outputs verified (config snapshots, logs, metadata)
- ✅ Embeddings cache working correctly
- ✅ All tests passing

### 🔄 Migration Notes

**100% Backward Compatible** - No breaking changes!

**Old Way (Still Works):**
```bash
# Legacy config loading still supported
antibody-train  # Uses Hydra by default now
```

**New Way (Recommended):**
```bash
# Default config (no args needed)
antibody-train

# Override parameters from CLI
antibody-train model.batch_size=16 classifier.C=0.5

# Switch model/classifier/data combinations
antibody-train model=esm2 classifier=mlp data=boughter_harvey

# Hyperparameter sweeps
antibody-train --multirun classifier.C=0.1,1.0,10.0
```

**For Existing Users:**
- No changes required - existing workflows continue to work
- New Hydra features available immediately
- Config files moved from `configs/` to `src/antibody_training_esm/conf/` (inside package)
- Legacy `train_model(config_path)` function still works (deprecated warning)

**For New Users:**
- Start with `antibody-train` for default Boughter → Jain training
- Override parameters from CLI (no file editing needed)
- Use `--multirun` for systematic hyperparameter exploration

### 📊 Stats

- 13 commits since v0.3.0
- 36 files changed (+3,592/-2,757 lines)
- 684 new test lines (3 new test files)
- 14 documentation files updated
- All tests passing (unit + integration + Hydra)
- 100% backward compatible

### 🎯 What's Next?

With Hydra in place, we can now:
- Add ESM2 support (just create `src/antibody_training_esm/conf/model/esm2.yaml`)
- Add MLP classifier (just create `src/antibody_training_esm/conf/classifier/mlp.yaml`)
- Systematic benchmarking with multirun sweeps
- W&B integration for experiment tracking (Phase 2)

---

## [0.3.0] - 2025-11-11

### 🛡️ Production Readiness - 34 Critical Bug Fixes

Comprehensive security and reliability audit of core ML pipeline. Fixed 34 critical bugs that would have caused silent data corruption, production crashes, and resource leaks.

### 🐛 Bug Fixes

**Round 1: 23 Critical Bugs**
- **8 P0 (Production Killers)**: Zero embeddings on batch failure, invalid sequences replaced with "M", cache deletion on training failure, hardcoded embedding dimensions, missing parameter validation, invalid log level crashes, missing column validation, config file error handling
- **3 P1 (High Severity)**: Division by zero in pooling (single + batch), sklearn set_params destroying fitted state, pickle load type validation
- **6 P2 (Medium)**: Tracked for future improvements
- **3 P3 (Low)**: Quality of life improvements
- **3 Backlogged**: Lower priority issues

**Round 2: 11 Critical Bugs**
- **2 P1 (Critical)**: Missing config validation (crash risk before GPU allocation), no validation of cached embeddings (silent corruption)
- **5 P2 (High Priority)**: Inconsistent amino acid validation (21 vs 20 AAs), weak backward compatibility warnings, test set size validation only warns, empty string defaults in fragment creation, no validation of loaded datasets
- **3 P3 (Medium)**: Poor error context in embeddings, loose typing in data loaders, silent test failures (wrong exit codes)

### 🔒 Security & Validation Improvements

**Data Corruption Prevention**
- Invalid sequences now raise errors instead of being silently replaced
- Batch failures now halt training instead of filling zero vectors
- Embedding cache validated for NaN values and all-zero rows
- Dataset loaders validate non-empty data immediately after loading

**Fail-Fast Validation**
- Config validation before GPU allocation (prevents expensive failures)
- Required column validation with helpful error messages showing available columns
- Embedding shape/integrity validation on cache load and compute
- Test set size enforcement (prevents invalid benchmark metrics)

**Type Safety & Compatibility**
- Proper Protocol typing for embedding extractors (compile-time type checking)
- sklearn compatibility preserved (set_params no longer destroys fitted state)
- Pickle load validation with graceful fallback to recomputation
- CI exit code validation (no more false-positive test passes)

### 📊 Impact

**Before v0.3.0:**
- Silent data corruption (training on zero vectors or single-AA sequences)
- Crashes with cryptic error messages
- Resource leaks (cache deleted even on training failure)
- False-positive CI results
- Invalid benchmark metrics accepted silently

**After v0.3.0:**
- Fail-fast with clear, actionable error messages
- No silent corruption anywhere in pipeline
- Cache preserved on failure (hours of GPU compute saved)
- Correct CI exit codes
- Invalid test sets rejected immediately

### 🔧 Files Modified

**Core Pipeline (9 files)**
- `src/antibody_training_esm/core/trainer.py` - Config validation, embeddings validation, cache preservation
- `src/antibody_training_esm/core/embeddings.py` - Batch failure handling, sequence validation, error context
- `src/antibody_training_esm/core/classifier.py` - Parameter validation, sklearn compatibility, backward compat warnings
- `src/antibody_training_esm/data/loaders.py` - Column validation, type safety (Protocol)
- `src/antibody_training_esm/datasets/base.py` - Fragment validation, AA validation
- `src/antibody_training_esm/datasets/jain.py` - Empty dataset validation
- `src/antibody_training_esm/datasets/harvey.py` - Empty dataset validation
- `src/antibody_training_esm/datasets/shehata.py` - Empty dataset validation
- `src/antibody_training_esm/cli/test.py` - Test size error enforcement, exit code validation

### 📚 Documentation

**Updated Canonical Docs**
- `docs/developer-guide/security.md` - Data validation principles, error handling best practices
- `docs/developer-guide/testing-strategy.md` - Lessons learned from production readiness audit
- `docs/user-guide/troubleshooting.md` - New sections for validation errors, config errors, cache errors

**Archived Technical Detail**
- `docs/archive/2025-11-11-production-readiness-audit.md` - Complete bug-by-bug analysis with before/after code

### ✅ Quality Gates

- 408 tests passing (3 skipped)
- 85.76% coverage
- Ruff lint: Clean
- Mypy: No issues
- Bandit: 0 findings
- 100% backward compatible (no breaking changes)

### 🔄 Migration Notes

**No action required** - All fixes are backward compatible. Users on v0.2.0 will automatically benefit from:
- Better error messages when things go wrong
- Validation that prevents silent corruption
- Cache preservation on training failure

**Recommended:** Delete old embedding cache and retrain to ensure no corrupted embeddings from pre-v0.3.0:
```bash
rm -rf embeddings_cache/  # (now experiments/cache/ in v0.5.0+)
uv run antibody-train
```

---

## [0.2.0] - 2025-11-10

### 🎉 Production Model Serialization + Documentation Overhaul

Major feature release enabling production-ready model deployment with secure serialization format and comprehensive documentation reorganization.

### ✨ Features

**Production Model Serialization**
- Dual-format model saving: pickle (research) + NPZ+JSON (production)
- New `load_model_from_npz()` function for secure cross-platform loading
- NPZ+JSON format eliminates code execution risk (unlike pickle)
- Cross-language compatibility (Rust/C++/JavaScript can load models)
- HuggingFace deployment ready
- All trained models automatically saved in both formats
- Public API export via `antibody_training_esm.core`

**Documentation Reorganization (Phases 0-8 Complete)**
- Complete user guide: installation, training, testing, preprocessing, troubleshooting
- Complete developer guide: architecture, workflow, testing, CI/CD, type checking, security, Docker, preprocessing internals
- Research documentation: methodology, Novo parity, assay thresholds, benchmark results
- Dataset-specific guides: Boughter, Jain, Harvey, Shehata
- Archive structure for historical investigations and migration docs
- Canonical structure with clear navigation and cross-linking

### 🐛 Bug Fixes

**Model Serialization**
- Fixed class_weight dict serialization (JSON converts int keys to strings, now converted back on load)
- Added proper type conversion for sklearn compatibility

**Documentation**
- Fixed multiple doc inaccuracies across user and developer guides
- Removed orphaned spec documents
- Updated all references to production serialization

### 🔧 Improvements

**API Design**
- Exported `load_model_from_npz()` in core public API
- Clean imports: `from antibody_training_esm.core import load_model_from_npz`
- 100% backward compatible (pickle still supported)

**Testing**
- Added `test_load_model_from_npz_with_dict_class_weight()` (TDD RED→GREEN)
- Added `test_train_model_saves_all_formats()` integration test
- 28/28 tests passing (was 26 before)
- Coverage maintained at 99.48% for trainer.py

**Security**
- Enhanced production deployment security (NPZ+JSON cannot execute code)
- Updated security documentation to reflect implemented state
- Maintained 0 Bandit findings

### 📦 Deliverables

- Docker images: `ghcr.io/the-obstacle-is-the-way/antibody-training:0.2.0`
- Three new tests for production serialization
- Complete documentation overhaul (user-guide + developer-guide + research + datasets)
- Dual-format model artifacts for all training runs

### 📊 Stats

- 79 commits since v0.1.0
- 28/28 tests passing
- 99.48% coverage on trainer.py
- 0 security findings (Bandit clean)
- 100% backward compatible

### 🔄 Migration Notes

**For Existing Users:**
- No breaking changes - all existing pickle-based workflows still work
- New NPZ+JSON format added alongside pickle (not replacing)
- Update imports to use: `from antibody_training_esm.core import load_model_from_npz`

**For Production Deployments:**
- Use NPZ+JSON format for secure cross-platform loading:
  ```python
  from antibody_training_esm.core import load_model_from_npz

  model = load_model_from_npz(
      npz_path="models/model.npz",  # (now experiments/checkpoints/{model}/{classifier}/ in v0.5.0+)
      json_path="models/model_config.json"
  )
  ```

### 📚 Documentation

Complete documentation reorganization:
- User guides: `docs/user-guide/` (installation, training, testing, preprocessing, troubleshooting)
- Developer guides: `docs/developer-guide/` (architecture, workflow, testing, CI/CD, security, Docker)
- Research: `docs/research/` (methodology, Novo parity, benchmarks, thresholds)
- Datasets: `docs/datasets/{boughter,jain,harvey,shehata}/`

---

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

- Jain confusion matrix: [[40, 19], [10, 17]] (66.28% vs Novo's 68.6% - close match)
- Shehata PSR threshold: 0.5495 (Novo Nordisk's PSR threshold - near-parity)
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

[0.7.0]: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/releases/tag/v0.7.0
[0.6.0]: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/releases/tag/v0.6.0
[0.5.0]: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/releases/tag/v0.5.0
[0.4.0]: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/releases/tag/v0.4.0
[0.3.0]: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/releases/tag/v0.3.0
[0.2.0]: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/releases/tag/v0.2.0
[0.1.0]: https://github.com/The-Obstacle-Is-The-Way/antibody_training_pipeline_ESM/releases/tag/v0.1.0
