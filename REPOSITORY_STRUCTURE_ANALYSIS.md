# Repository Structure Analysis - Current State & Professional Comparison

**Date**: 2025-11-15
**Status**: Post-Phase 1-4 Cleanup, Pre-Phase 5 Reorganization
**Purpose**: Comprehensive diagnosis of repository organization vs professional ML research standards

---

## Executive Summary

**Current State**: ✅ **Functional but Suboptimal**

The repository **works correctly** and Phase 1-4 cleanup (test artifacts, historical archival, CV results persistence) was **successful and commit-ready**. However, the overall directory structure reflects **inherited patterns** from a legacy codebase that diverge from professional ML research repository organization.

**Key Findings**:
- ✅ Core functionality is solid (374/374 tests passing, 82.38% coverage)
- ⚠️ Output artifacts scattered across multiple locations (`outputs/`, `models/`, `embeddings_cache/`, `test_results/`)
- ⚠️ Mixed organization patterns (dataset-centric preprocessing vs task-centric scripts)
- ⚠️ Competing sources of truth for experimental results (`experiments/` vs `test_results/`)

**Recommendation**: Commit Phase 1-4 cleanup, then execute **Phase 5 Repository Reorganization** to consolidate outputs and align with professional standards.

---

## Current Repository Structure (Complete Tree)

```
antibody_training_pipeline_ESM/
├── AGENTS.md
├── CHANGELOG.md
├── CITATIONS.md
├── CLAUDE.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile.dev
├── Dockerfile.prod
├── KNOWN_ISSUES.md
├── LICENSE
├── Makefile
├── PROBLEMS.md                        # 🆕 Structural issues identified
├── README.md
├── ROADMAP.md
├── USAGE.md
├── V0.5.0_CLEANUP_PLAN.md
│
├── assets/                            # Static assets (images)
│   ├── ego.jpg
│   ├── fascinating.jpg
│   ├── leeroy_jenkins.png
│   └── leeroy_jenkins.webp
│
├── configs/                           # ✅ Hydra configuration
│   └── config.yaml
│
├── data/                              # ✅ Dataset storage (good separation)
│   ├── test/
│   │   ├── harvey/
│   │   ├── jain/
│   │   └── shehata/
│   └── train/
│       ├── BOUGHTER_DATA_PROVENANCE.md
│       ├── README.md
│       └── boughter/
│
├── dist/                              # Build artifacts
│   ├── antibody_training_esm-0.3.0-py3-none-any.whl
│   └── antibody_training_esm-0.3.0.tar.gz
│
├── docker-compose.yml
│
├── docs/                              # ✅ Comprehensive documentation
│   ├── ESM1V_ENSEMBLING_INVESTIGATION.md
│   ├── README.md
│   ├── archive/                      # Historical cleanup plans
│   │   ├── 2025-11-11-production-readiness-audit.md
│   │   ├── CLEANUP_COMPLETE_SUMMARY.md
│   │   ├── CLEANUP_PLAN.md
│   │   ├── [... 15 more archived docs ...]
│   │   ├── investigations/
│   │   ├── migrations/
│   │   ├── plans/
│   │   └── summaries/
│   ├── datasets/                     # Dataset documentation
│   │   ├── boughter/
│   │   ├── harvey/
│   │   ├── jain/
│   │   └── shehata/
│   ├── developer-guide/              # Developer documentation
│   │   ├── architecture.md
│   │   ├── ci-cd.md
│   │   ├── development-workflow.md
│   │   ├── directory-organization.md  # ⚠️ Needs update after Phase 5
│   │   ├── docker.md
│   │   ├── preprocessing-internals.md
│   │   ├── security.md
│   │   ├── testing-strategy.md
│   │   └── type-checking.md
│   ├── overview.md
│   ├── research/                     # Research documentation
│   │   ├── assay-thresholds.md
│   │   ├── benchmark-results.md
│   │   ├── methodology.md
│   │   └── novo-parity.md
│   ├── to-be-integrated/             # Pending integration
│   │   ├── CLI_OVERRIDE_BUG_ROOT_CAUSE.md
│   │   ├── ESM2_FEATURE.md
│   │   ├── output_pipeline_architecture.md
│   │   └── training_pipeline_investigation.md
│   └── user-guide/
│       ├── getting-started.md
│       ├── installation.md
│       ├── preprocessing.md
│       ├── testing.md
│       ├── training.md
│       └── troubleshooting.md
│
├── docs_burner/                      # ✅ Working docs (not for commit)
│   ├── CURRENT_STRUCTURE.txt
│   ├── OUTPUT_DIRECTORY_INVESTIGATION.md
│   ├── OUTPUT_ORGANIZATION_FINAL_CLEANUP_PLAN.md
│   ├── POST_MIGRATION_VALIDATION_FINDINGS.md
│   ├── POST_MIGRATION_VALIDATION_PLAN.md
│   ├── POST_MIGRATION_VALIDATION_SUMMARY.md
│   ├── REPOSITORY_CLEANUP_PLAN.md
│   ├── TEST_DATASETS_CONSOLIDATION_PLAN.md
│   ├── TRAIN_DATASETS_CONSOLIDATION_PLAN.md
│   └── implementation/
│       ├── DOCKER_CI_FAILURE_ANALYSIS.md
│       ├── GITHUB_ACTIONS_DISK_SPACE.md
│       ├── HYPERPARAMETER_SWEEP_ARCHIVE_PLAN.md
│       └── TEST_COVERAGE_PLAN.md
│
├── embeddings_cache/                 # ⚠️ ROOT-LEVEL OUTPUT (should be in experiments/)
│   └── train_0e097a6a2cea_embeddings.pkl  # 4.5MB
│
├── experiments/                      # ⚠️ MIXED PURPOSES
│   ├── README.md
│   ├── archive/                      # ✅ Historical results (versioned)
│   │   ├── hyperparameter_sweeps_2025-11-02/
│   │   └── test_results_pre_migration_2025-11-06/
│   ├── hyperparameter_sweeps/        # ❌ EMPTY (no .gitkeep)
│   ├── novo_parity/                  # ✅ Active experiment (well-organized)
│   │   ├── ELISA_THRESHOLD_HYPOTHESIS_TEST.md
│   │   ├── EXACT_MATCH_FOUND.md
│   │   ├── EXPERIMENTS_LOG.md
│   │   ├── FINAL_PERMUTATION_HUNT.md
│   │   ├── MISSION_ACCOMPLISHED.md
│   │   ├── PERMUTATION_TESTING.md
│   │   ├── archive/
│   │   ├── datasets/                # Alternative Jain dataset variants
│   │   ├── results/                 # Test results with metrics
│   │   └── scripts/
│   └── strict_qc_2025-11-04/         # ⚠️ Alternative dataset experiment
│       ├── EXPERIMENT_README.md
│       ├── configs/
│       ├── data/                    # ⚠️ Duplicate data (also in data/train/)
│       ├── docs/
│       └── preprocessing/
│
├── literature/                       # ✅ Reference papers (well-organized)
│   ├── markdown/
│   │   ├── boltzmann_2024_main/
│   │   ├── boughter_2020_main/
│   │   ├── esm_model/
│   │   ├── harvey_2022_main/
│   │   ├── harvey_2022_supplementary/
│   │   ├── jain_2017_main/
│   │   ├── jain_2017_supplementary/
│   │   ├── novo_2025_main/
│   │   ├── novo_2025_supplementary/
│   │   └── shehata_2019_main/
│   └── pdf/
│       ├── [... corresponding PDFs ...]
│
├── logs/                             # ⚠️ ROOT-LEVEL LOGS (should be in experiments/)
│   ├── boughter_retrain_20251106_211513.log
│   ├── boughter_training.log
│   ├── build.log
│   ├── full_test_suite_20251106_211755.log
│   ├── prod-build.log
│   ├── test_harvey_20251106_212635.log
│   └── test_shehata_20251106_212354.log
│
├── models/                           # ⚠️ ROOT-LEVEL MODELS (should be in experiments/)
│   ├── esm1v/                       # ✅ Hierarchical organization (good)
│   │   └── logreg/
│   │       ├── boughter_vh_esm1v_logreg.npz
│   │       ├── boughter_vh_esm1v_logreg.pkl
│   │       └── boughter_vh_esm1v_logreg_config.json
│   └── esm2_650m/
│       └── logreg/
│           ├── boughter_vh_esm2_650m_logreg.npz
│           ├── boughter_vh_esm2_650m_logreg.pkl
│           └── boughter_vh_esm2_650m_logreg_config.json
│   # Total size: 56KB (versioned in git)
│
├── outputs/                          # ✅ HYDRA SCRATCH (gitignored)
│   ├── cv_results_test/
│   │   └── 2025-11-15_20-09-58/
│   ├── cv_yaml_test/
│   │   └── 2025-11-15_20-10-53/
│   └── post_migration_smoke_test/
│       └── 2025-11-15_15-43-37/
│
├── preprocessing/                    # ⚠️ DATASET-CENTRIC (should be task-centric)
│   ├── README.md
│   ├── __init__.py
│   ├── boughter/
│   │   ├── README.md
│   │   ├── audit_training_qc.py
│   │   ├── stage1_dna_translation.py
│   │   ├── stage2_stage3_annotation_qc.py
│   │   ├── train_hyperparameter_sweep.py  # ⚠️ Training script in preprocessing/
│   │   ├── validate_stage1.py
│   │   └── validate_stages2_3.py
│   ├── harvey/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── step1_convert_raw_csvs.py
│   │   ├── step2_extract_fragments.py
│   │   └── test_psr_threshold.py
│   ├── jain/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── step1_convert_excel_to_csv.py
│   │   ├── step2_preprocess_p5e_s2.py
│   │   ├── step3_extract_fragments.py
│   │   ├── test_novo_parity.py
│   │   └── validate_conversion.py
│   └── shehata/
│       ├── README.md
│       ├── __init__.py
│       ├── step1_convert_excel_to_csv.py
│       ├── step2_extract_fragments.py
│       └── validate_conversion.py
│
├── pyproject.toml
├── pytest.ini
│
├── reference_repos/                  # ✅ External reference implementations
│   ├── AIMS/
│   ├── AIMS_manuscripts/
│   ├── abdev-benchmark/
│   ├── esm/
│   ├── harvey_official_repo/
│   └── ludocomito_original/
│
├── scripts/                          # ⚠️ INCOMPLETE (should contain ALL scripts)
│   ├── migrate_model_directories.py
│   ├── migrate_test_datasets_to_data_test.sh
│   ├── migrate_train_datasets_to_data_train.sh
│   ├── testing/
│   │   ├── README.md
│   │   └── demo_assay_specific_thresholds.py
│   └── validation/
│       ├── README.md
│       ├── validate_fragments.py
│       └── validate_jain_csvs.py
│
├── src/                              # ✅ CORE PACKAGE (well-organized)
│   └── antibody_training_esm/
│       ├── __init__.py
│       ├── cli/
│       ├── conf/                    # Hydra configs (duplicates configs/?)
│       ├── core/
│       ├── data/
│       ├── datasets/
│       ├── evaluation/
│       └── utils/
│
├── test_results/                     # ❌ EMPTY + REDUNDANT
│   └── .gitkeep                     # Only file (just added in Phase 2)
│
├── tests/                            # ✅ COMPREHENSIVE TEST SUITE
│   ├── __init__.py
│   ├── conftest.py
│   ├── e2e/
│   │   ├── __init__.py
│   │   ├── test_reproduce_novo.py
│   │   └── test_train_pipeline.py
│   ├── fixtures/
│   │   ├── __init__.py
│   │   ├── mock_datasets/
│   │   ├── mock_models.py
│   │   └── mock_sequences.py
│   ├── integration/
│   │   ├── [... 8 integration tests ...]
│   └── unit/
│       ├── __init__.py
│       ├── cli/
│       ├── core/
│       ├── data/
│       └── datasets/
│           └── conftest.py          # 🆕 Added in Phase 1
│
└── uv.lock

130 directories, 202 files
```

---

## Structural Problems Identified

### **Problem 1: Scattered Output Artifacts** ⚠️

**Issue**: Output artifacts are scattered across **5 different root-level locations**:

| Directory | Purpose | Git Status | Size | Problem |
|-----------|---------|------------|------|---------|
| `outputs/` | Hydra training runs | Gitignored | Varies | ✅ Correct usage |
| `models/` | Trained model checkpoints | **Versioned** | 56KB | ⚠️ Should be in experiments/ |
| `embeddings_cache/` | ESM embedding cache | Gitignored | 4.5MB | ⚠️ Should be in experiments/ |
| `test_results/` | Test evaluation results | Versioned | **Empty** | ❌ Redundant, delete |
| `logs/` | Training/test logs | Versioned | Varies | ⚠️ Should be in experiments/ |

**Impact**:
- Unclear where to find artifacts ("Are models in `models/` or `outputs/{run}/`?")
- Inconsistent gitignore patterns (some outputs versioned, others not)
- Difficult to archive/clean old experiments

**Professional Pattern**: Single `experiments/` directory with clear subdirectories:
```
experiments/
├── runs/         # Hydra outputs (gitignored)
├── checkpoints/  # Models (gitignored or LFS)
├── cache/        # Embeddings, preprocessing (gitignored)
└── benchmarks/   # Published results (versioned)
```

---

### **Problem 2: test_results/ is Empty and Redundant** ❌

**Evidence**:
```bash
$ ls -la test_results/
total 0
drwxr-xr-x@  3 ray  staff    96 Nov 15 17:18 .
-rw-r--r--@  1 ray  staff     0 Nov 15 17:18 .gitkeep
```

**History**:
- Phase 2 cleanup **archived** old test results to `experiments/archive/test_results_pre_migration_2025-11-06/`
- New test results go to `experiments/novo_parity/results/` (86 Jain parity benchmark)
- Directory now serves **no purpose**

**Recommendation**: **DELETE** `test_results/` entirely. Use `experiments/benchmarks/` for published results.

---

### **Problem 3: experiments/ Has Mixed Purposes** ⚠️

**Current Structure**:
```
experiments/
├── archive/                   # ✅ Historical results (good)
├── hyperparameter_sweeps/     # ❌ Empty directory (no .gitkeep)
├── novo_parity/               # ✅ Active experiment (good)
│   ├── datasets/              # Alternative Jain variants
│   ├── results/               # Test results
│   └── scripts/               # Experiment scripts
└── strict_qc_2025-11-04/      # ⚠️ Has data/ and configs/ (duplicates data/train/)
```

**Issues**:
- `strict_qc_2025-11-04/data/` duplicates canonical datasets from `data/train/`
- `hyperparameter_sweeps/` is empty (no .gitkeep, no README)
- No clear distinction between "active experiments" and "published benchmarks"

**Professional Pattern**:
```
experiments/
├── runs/                      # Ephemeral Hydra runs (gitignored)
├── checkpoints/               # Models (gitignored or LFS)
├── cache/                     # Embeddings (gitignored)
└── benchmarks/                # Published results (versioned)
    ├── novo_parity/           # Main Novo replication
    ├── strict_qc/             # Strict QC variant
    ├── ablations/
    └── archive/               # Historical benchmarks
```

---

### **Problem 4: Dataset-Centric Preprocessing** ⚠️

**Current Structure**:
```
preprocessing/
├── boughter/
│   ├── stage1_dna_translation.py
│   ├── stage2_stage3_annotation_qc.py
│   └── train_hyperparameter_sweep.py  # ← Training script!
├── jain/
│   ├── step1_convert_excel_to_csv.py
│   ├── step2_preprocess_p5e_s2.py
│   └── test_novo_parity.py
└── harvey/, shehata/ (similar)
```

**Issues**:
- Training scripts buried in `preprocessing/boughter/`
- Testing scripts buried in `preprocessing/jain/`, `preprocessing/harvey/`
- Hard to find "all preprocessing scripts" or "all training scripts"
- Doesn't scale (what if you preprocess Jain + Harvey together?)

**Professional Pattern**:
```
scripts/
├── preprocessing/
│   ├── preprocess_boughter.py
│   ├── preprocess_jain.py
│   ├── preprocess_harvey.py
│   └── preprocess_shehata.py
├── train.py
├── evaluate.py
└── sweep_hyperparameters.py
```

**Task-centric organization**: Find scripts by **what they do**, not by **which dataset they use**.

---

### **Problem 5: Root-Level Clutter** ⚠️

**Current Root Has**:
- 16 Markdown files (README, CLAUDE, PROBLEMS, ROADMAP, etc.)
- 9 top-level directories for outputs (`outputs/`, `models/`, `embeddings_cache/`, `logs/`, `test_results/`, `experiments/`)
- 4 config files (Dockerfile, docker-compose.yml, Makefile, pyproject.toml)

**Professional Repos Have**:
- Clean root with **6-8 top-level directories max**
- Clear separation: **code** (`src/`, `scripts/`) vs **data** (`data/`) vs **outputs** (`experiments/`)

---

## Professional ML Research Repository Pattern

Based on analysis of AlphaFold, CLIP, ESM, and other Meta/DeepMind/OpenAI repositories:

```
repo_name/
├── README.md                   # Entry point with quickstart
├── pyproject.toml             # Dependencies
├── Makefile                   # Common commands
│
├── data/                       # Dataset storage
│   ├── README.md              # Download instructions
│   ├── train/
│   └── test/
│
├── src/{package}/             # Core library code
│   ├── models/
│   ├── data/
│   └── training/
│
├── scripts/                    # ALL executable scripts
│   ├── preprocessing/         # Data preprocessing
│   ├── train.py               # Training
│   ├── evaluate.py            # Evaluation
│   └── sweep.py               # Hyperparameter search
│
├── configs/                    # Configuration files
│
├── experiments/                # SINGLE source of truth for outputs
│   ├── runs/                  # Ephemeral training runs (gitignored)
│   ├── checkpoints/           # Saved models (gitignored or Git LFS)
│   ├── cache/                 # Intermediate artifacts (gitignored)
│   └── benchmarks/            # Published results (versioned)
│       ├── main/              # Primary benchmark
│       ├── ablations/
│       └── archive/
│
├── docs/                       # Documentation
├── tests/                      # Test suite
└── literature/                 # Reference papers (optional)
```

**Key Principles**:
1. **Single `experiments/` directory** for ALL outputs
2. **Task-centric `scripts/`** (not dataset-centric)
3. **Clean root** with minimal top-level directories
4. **Clear gitignore separation** (ephemeral vs published)

---

## Current State Assessment

### ✅ **What's Working Well**

1. **Core Functionality** (374/374 tests passing, 82.38% coverage)
2. **Data Organization** (`data/train/` and `data/test/` clear separation)
3. **Documentation** (comprehensive `docs/` with research, developer guides, datasets)
4. **Test Suite** (unit, integration, e2e with fixtures)
5. **Phase 1-4 Cleanup** (test artifacts fixed, CV results persistence added, historical results archived)

### ⚠️ **What Needs Improvement**

1. **Scattered Outputs** (`models/`, `embeddings_cache/`, `logs/`, `test_results/` at root)
2. **Empty Redundant Directories** (`test_results/`, `experiments/hyperparameter_sweeps/`)
3. **Mixed Organization Patterns** (dataset-centric preprocessing, incomplete `scripts/`)
4. **Unclear Output Hierarchy** (`outputs/` vs `experiments/` vs `test_results/`)

### ❌ **Immediate Issues**

1. **test_results/** is completely empty (only `.gitkeep`)
2. **No clear SSOT** for where test results should go
3. **Training scripts** buried in `preprocessing/{dataset}/`

---

## Comparison: Current vs Professional

| Aspect | Current State | Professional Pattern | Gap |
|--------|---------------|---------------------|-----|
| **Output Organization** | 5 root-level dirs (outputs, models, embeddings_cache, logs, test_results) | Single experiments/ dir | ⚠️ Major |
| **Script Organization** | Dataset-centric preprocessing/ | Task-centric scripts/ | ⚠️ Moderate |
| **Test Results** | Empty test_results/ + experiments/*/results/ | experiments/benchmarks/ only | ⚠️ Moderate |
| **Models** | models/ at root (versioned, 56KB) | experiments/checkpoints/ (gitignored or LFS) | ⚠️ Minor |
| **Embeddings Cache** | embeddings_cache/ at root (4.5MB) | experiments/cache/ | ⚠️ Minor |
| **Root Clutter** | 16 markdown files, 9 output dirs | 6-8 top-level dirs max | ⚠️ Moderate |

---

## Recommendation: Phase 5 Reorganization

**Verdict**: Phase 1-4 cleanup was **100% successful** and is **commit-ready**. The structural issues identified here are **pre-existing** and should be addressed in a **separate Phase 5 reorganization**.

**Next Steps**:
1. ✅ **Commit Phase 1-4 cleanup** (test artifacts, CV results, historical archival)
2. 📋 **Review & approve** `REPOSITORY_REORGANIZATION_PLAN.md` (Phase 5)
3. 🚀 **Execute Phase 5** in separate commit after senior approval

**Status**: ✅ **Ready for senior review and commit**

---

## Appendix: .gitignore Analysis

**Current .gitignore (excerpt)**:
```
# Ephemeral outputs
outputs/*
models/scratch/
models/ginkgo_*/
embeddings_cache/*

# ⚠️ Issue: test_results/ NOT gitignored (but it's empty)
# ⚠️ Issue: models/ NOT fully gitignored (versioned models at root)
# ⚠️ Issue: logs/ NOT gitignored (versioned logs at root)
```

**Professional Pattern**:
```
# Ephemeral outputs
experiments/runs/*
experiments/checkpoints/*
experiments/cache/*

# Keep directory structure
!experiments/runs/.gitkeep
!experiments/checkpoints/.gitkeep
!experiments/cache/.gitkeep

# Published results are versioned
!experiments/benchmarks/
```

---

**Document Status**: ✅ Complete
**Next Document**: `REPOSITORY_REORGANIZATION_PLAN.md` (Phase 5 implementation plan)
