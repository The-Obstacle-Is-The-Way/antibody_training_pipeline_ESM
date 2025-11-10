# Documentation

This directory contains technical documentation for the antibody training pipeline.

**🆕 New to the project?** Start with the [System Overview](overview.md) to understand what this pipeline does and how it works.

---

## 🎯 **FOR NOVO NORDISK PARITY RESULTS (FINAL)**

**The authoritative reverse engineering results are located in:**

📁 **`experiments/novo_parity/`**

**Key documents:**
- **Executive Summary**: `experiments/novo_parity/MISSION_ACCOMPLISHED.md`
- **Technical Details**: `experiments/novo_parity/EXACT_MATCH_FOUND.md`
- **Final Dataset**: `experiments/novo_parity/datasets/jain_86_p5e_s2.csv`

**Result**: [[40, 19], [10, 17]] ✅ **EXACT MATCH** to Novo Nordisk's confusion matrix (66.28% accuracy)

---

## Current Documentation Structure

### 📁 Developer Guide (`developer-guide/`)

**NEW:** Canonical developer documentation (consolidated from `development/`):

- `architecture.md` - System architecture, core components, design patterns
- `development-workflow.md` - Git workflow, commands, quality gates, common tasks
- `preprocessing-internals.md` - Dataset preprocessing patterns and techniques
- `type-checking.md` - Type safety requirements, mypy configuration, best practices
- `security.md` - Security model, pickle policy, dependency management, scanning

- `docker.md` - Docker development and deployment
- `testing-strategy.md` - Test architecture, patterns, coverage
- `ci-cd.md` - CI/CD pipeline, workflows, enforcement

### 📁 Development Documentation (`development/`)

**Legacy technical docs** (being consolidated into `developer-guide/`):

- **Other**
  - `IMPORT_AND_STRUCTURE_GUIDE.md` - v2.0.0 import structure guide
  - `P0_P1_P2_P3_BLOCKERS.md` - Priority issue tracking

### 📁 Research Documentation (`research/`)

Scientific methodology and validation:

- `novo-parity.md` - Novo Nordisk parity analysis (replication, methodology, QC)
- `methodology.md` - Implementation details, dataset analysis, divergences
- `ASSAY_SPECIFIC_THRESHOLDS.md` - ELISA vs PSR thresholds
- `BENCHMARK_TEST_RESULTS.md` - Cross-dataset validation results
- `COMPLETE_VALIDATION_RESULTS.md` - Comprehensive validation report

### 📁 Archive (`archive/`)

Historical documentation from development process:

- **Completed Plans**
  - `CLEANUP_PLAN.md` - Jain dataset cleanup plan (completed 2025-11-05)
  - `CLEANUP_COMPLETE_SUMMARY.md` - Cleanup execution summary
  - `STRICT_QC_CLEANUP_PLAN.md` - Quality control cleanup plan
  - `TRAINING_SETUP_STATUS.md` - Training setup status report

- **Investigations & Fixes**
  - `FIXES_APPLIED.md` - Bug fixes and corrections log
  - `MPS_MEMORY_LEAK_FIX.md` - MPS memory leak fix (2025-11-03)
  - `P0_SEMAPHORE_LEAK.md` - Semaphore leak investigation
  - `SCRIPTS_AUDIT.md` - Script audit report
  - `RESIDUAL_TYPE_ERRORS.md` - Type error tracking

- **Codebase Reorganization**
  - `CODEBASE_REORGANIZATION_PLAN.md` - v2.0.0 restructuring plan
  - `TEST_DATASETS_REORGANIZATION_PLAN.md` - Test dataset reorganization
  - `REPOSITORY_MODERNIZATION_PLAN.md` - 2025 tooling upgrade plan

- **Audit Reports**
  - `DOCS_AUDIT_STATUS.md` - Documentation audit (pre-reorganization)
  - `PHASE1_TEST_RESULTS.md` - Phase 1 test results

### 📁 Dataset Documentation (`datasets/`)

Dataset-specific preprocessing and validation:

- **`boughter/`** - Training dataset (914 VH sequences, ELISA polyreactivity)
- **`jain/`** - Test dataset (86 clinical antibodies, Novo parity benchmark)
- **`harvey/`** - Test dataset (nanobodies, PSR assay)
- **`shehata/`** - Test dataset (398 antibodies, PSR cross-validation)

Each dataset directory contains preprocessing scripts, validation reports, and data source documentation.

---

**Last Updated**: 2025-11-09
**Branch**: `docs/canonical-structure`
