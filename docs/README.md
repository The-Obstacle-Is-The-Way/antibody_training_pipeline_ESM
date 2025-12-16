# Documentation

This directory contains technical documentation for the antibody training pipeline.

**🆕 New to the project?** Start with the [System Overview](overview.md) to understand what this pipeline does and how it works.

---

## 🎯 **FOR NOVO NORDISK PARITY RESULTS (FINAL)**

**Location:** Archived in the `archive` branch to keep `dev` clean.

```bash
git checkout archive
cd experiments/benchmarks/novo_parity/
```

**Key documents (in archive):**
- **Executive Summary**: `experiments/benchmarks/novo_parity/MISSION_ACCOMPLISHED.md`
- **Technical Details**: `experiments/benchmarks/novo_parity/EXACT_MATCH_FOUND.md`
- **Final Dataset**: `experiments/benchmarks/novo_parity/datasets/jain_86_p5e_s2.csv`

**Our Result**: [[40, 19], [10, 17]], 66.28% (Novo target: [[40, 17], [10, 19]], 68.6% - off by 2 antibodies)

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
- `xgboost.md` - Canonical guide for the XGBoost classifier backend (usage, config, tests)

### 📁 ~~Development Documentation (`development/`)~~

**ARCHIVED** - All files moved to `archive/` (Phase 6 complete):

- **Migrations**
  - ~~`IMPORT_AND_STRUCTURE_GUIDE.md`~~ → `archive/migrations/v2-structure-migration.md`
- **Investigations**
  - ~~`P0_P1_P2_P3_BLOCKERS.md`~~ → `archive/investigations/p0-blockers.md`

### 📁 User Guide (`user-guide/`)

End-user documentation:

- `getting-started.md` - Quick start guide
- `installation.md` - Environment setup
- `training.md` - Model training guide
- `inference.md` - **Comprehensive prediction guide** (NEW)
- `testing.md` - Model evaluation
- `preprocessing.md` - Dataset preparation
- `troubleshooting.md` - Common issues and solutions

### 📁 Research Documentation (`research/`)

Scientific methodology and validation:

- `novo-parity.md` - Novo Nordisk parity analysis (replication, methodology, QC)
- `methodology.md` - Implementation details, dataset analysis, divergences
- `assay-thresholds.md` - ELISA vs PSR thresholds
- `benchmark-results.md` - Cross-dataset validation results (Boughter, Jain, Harvey, Shehata)
- `model-zoo-roadmap.md` - **Future model expansion roadmap** (NEW)

### 📁 Archive (`archive/`)

Historical documentation, fully organized under `docs/archive/`:

- **Audits** (3 files)
  - `archive/audits/2025-11-19-xgboost-branch-audit.md` (NEW)
  - `archive/audits/2025-11-11-production-readiness-audit.md`
  - `archive/audits/2025-11-05-scripts-audit.md`

- **Investigations** (7 files)
  - `archive/investigations/cli-test-refactor-2025-11-18.md` (NEW)
  - `archive/investigations/dataset-column-naming-2025-11-18.md` (NEW)
  - `archive/investigations/2025-11-03-mps-memory-leak.md`
  - `archive/investigations/2025-11-06-p0-semaphore-leak.md`
  - `archive/investigations/2025-11-11-cli-override-bug.md`
  - `archive/investigations/2025-11-11-training-pipeline-fixes.md`
  - `archive/investigations/p0-blockers.md`

- **Summaries** (6 files)
  - `archive/summaries/inference-completion-2025-11-19.md` (NEW)
  - `archive/summaries/2025-11-02-fixes-applied.md`
  - `archive/summaries/2025-11-02-phase1-test-results.md`
  - `archive/summaries/2025-11-02-training-setup-status.md`
  - `archive/summaries/2025-11-06-type-checking-complete.md`
  - `archive/summaries/2025-11-12-esm2-feature.md`

- **Decisions** (1 file)
  - `archive/decisions/preprocessing-location-decision-2025-11-18.md` (NEW)

- **Migrations** (3 files)
  - `archive/migrations/2025-11-05-codebase-reorganization.md`
  - `archive/migrations/2025-11-06-repository-modernization.md`
  - `archive/migrations/v2-structure-migration.md`

- **Plans + Specs** (6 files)
  - `archive/plans/2025-11-11-output-pipeline-architecture.md`
  - `archive/plans/DOCUMENTATION_STRUCTURE_PLAN.md`
  - `archive/plans/xgboost-integration-spec.md`
  - `archive/plans/xgboost-api-design.md`
  - `archive/plans/xgboost-test-plan.md`
  - `archive/plans/xgboost-implementation-status.md`

- **Trash** (2 files)
  - `archive/trash/refactor-test-cli-plan.md` (NEW)
  - `archive/trash/spec-sheet.md` (NEW)

> See `docs/archive/README.md` for archive policy and full listing.

### 📁 Dataset Documentation (`datasets/`)

Dataset-specific preprocessing and validation:

- **`boughter/`** - Training dataset (914 VH sequences, ELISA polyreactivity)
- **`jain/`** - Test dataset (86 clinical antibodies, Novo parity benchmark)
- **`harvey/`** - Test dataset (nanobodies, PSR assay)
- **`shehata/`** - Test dataset (398 antibodies, PSR cross-validation)

Each dataset directory contains preprocessing scripts, validation reports, and data source documentation.

---

### 📁 Needs Integration (`needs_integration/`)

Active tracking documents (work in progress):

- `ARCHITECTURAL_FIXES_PLAN.md` - Ongoing refactoring roadmap (P1-P3 tasks)

---

**Last Updated**: 2025-11-19
**Branch**: `dev`
