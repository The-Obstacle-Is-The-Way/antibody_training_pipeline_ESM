# Documentation

This directory contains technical documentation for the antibody training pipeline.

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

### 📁 Development Documentation (`development/`)

Technical guides for contributors and developers:

- **CI/CD & Infrastructure**
  - `CICD_SPEC.md` - CI/CD pipeline specification
  - `CICD_SETUP_GUIDE.md` - GitHub Actions setup guide
  - `BRANCH_PROTECTION_WALKTHROUGH.md` - Branch protection configuration
  - `DOCKER_DEPLOYMENT.md` - Docker deployment guide
  - `DOCKER_USAGE.md` - Docker development workflow

- **Code Quality & Security**
  - `TYPE_CHECKING_STRATEGY.md` - mypy type checking strategy
  - `TYPE_HINTING_REMEDIATION_PLAN.md` - Type hint improvements
  - `SECURITY_REMEDIATION_PLAN.md` - Security best practices
  - `CODEQL_FINDINGS.md` - CodeQL security analysis

- **Testing & Coverage**
  - `TEST_SUITE_PLAN.md` - Test suite architecture
  - `TEST_SUITE_REVIEW_CHECKLIST.md` - Testing guidelines
  - `TEST_COVERAGE_GAPS.md` - Coverage improvement tracking

- **Other**
  - `IMPORT_AND_STRUCTURE_GUIDE.md` - v2.0.0 import structure guide
  - `P0_P1_P2_P3_BLOCKERS.md` - Priority issue tracking
  - `excel_to_csv_conversion_methods.md` - Data preprocessing methods

### 📁 Research Documentation (`research/`)

Scientific methodology and validation:

- `METHODOLOGY_AND_DIVERGENCES.md` - Pipeline methodology vs paper
- `NOVO_PARITY_ANALYSIS.md` - Novo Nordisk replication analysis
- `NOVO_REPLICATION_PLAN.md` - Replication strategy
- `NOVO_TRAINING_METHODOLOGY.md` - Training methodology details
- `CODEBASE_AUDIT_VS_NOVO.md` - Implementation audit
- `CRITICAL_IMPLEMENTATION_ANALYSIS.md` - Key implementation details
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
