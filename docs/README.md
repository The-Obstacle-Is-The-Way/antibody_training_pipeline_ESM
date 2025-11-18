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

### 📁 ~~Development Documentation (`development/`)~~

**ARCHIVED** - All files moved to `archive/` (Phase 6 complete):

- **Migrations**
  - ~~`IMPORT_AND_STRUCTURE_GUIDE.md`~~ → `archive/migrations/v2-structure-migration.md`
- **Investigations**
  - ~~`P0_P1_P2_P3_BLOCKERS.md`~~ → `archive/investigations/p0-blockers.md`

### 📁 Research Documentation (`research/`)

Scientific methodology and validation:

- `novo-parity.md` - Novo Nordisk parity analysis (replication, methodology, QC)
- `methodology.md` - Implementation details, dataset analysis, divergences
- `assay-thresholds.md` - ELISA vs PSR thresholds
- `benchmark-results.md` - Cross-dataset validation results (Boughter, Jain, Harvey, Shehata)

### 📁 Archive (`archive/`)

Historical documentation, fully organized under `docs/archive/`:

- **Audits**
  - `archive/audits/2025-11-11-production-readiness-audit.md`
  - `archive/audits/2025-11-05-scripts-audit.md`

- **Investigations**
  - `archive/investigations/2025-11-03-mps-memory-leak.md`
  - `archive/investigations/2025-11-06-p0-semaphore-leak.md`
  - `archive/investigations/2025-11-11-cli-override-bug.md`
  - `archive/investigations/2025-11-11-training-pipeline-fixes.md`
  - `archive/investigations/p0-blockers.md`

- **Migrations**
  - `archive/migrations/2025-11-05-codebase-reorganization.md`
  - `archive/migrations/2025-11-06-repository-modernization.md`
  - `archive/migrations/v2-structure-migration.md`

- **Plans**
  - `archive/plans/2025-11-11-output-pipeline-architecture.md`
  - `archive/plans/DOCUMENTATION_STRUCTURE_PLAN.md`

- **Summaries**
  - `archive/summaries/2025-11-02-fixes-applied.md`
  - `archive/summaries/2025-11-02-phase1-test-results.md`
  - `archive/summaries/2025-11-02-training-setup-status.md`
  - `archive/summaries/2025-11-06-type-checking-complete.md`
  - `archive/summaries/2025-11-12-esm2-feature.md`

> Trash is currently empty; see `docs/archive/README.md` for archive policy.

### 📁 Dataset Documentation (`datasets/`)

Dataset-specific preprocessing and validation:

- **`boughter/`** - Training dataset (914 VH sequences, ELISA polyreactivity)
- **`jain/`** - Test dataset (86 clinical antibodies, Novo parity benchmark)
- **`harvey/`** - Test dataset (nanobodies, PSR assay)
- **`shehata/`** - Test dataset (398 antibodies, PSR cross-validation)

Each dataset directory contains preprocessing scripts, validation reports, and data source documentation.

---

**Last Updated**: 2025-11-17
**Branch**: `leroy-jenkins/full-send`
