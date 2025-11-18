# Archive

This directory contains **historical documentation** from the development process. These documents represent completed work, investigations, and planning efforts that provide context for past decisions.

**Archive Policy:** Documents here are time-bound, completed, and serve as historical reference only. For current documentation, see the main `docs/` directory.

---

## Directory Structure

All archive documents are organized into 5 categories:

```
archive/
├── README.md                                              # This file
│
├── audits/                                                # Code audits, security audits, quality audits (2 docs)
│   ├── 2025-11-11-production-readiness-audit.md         # 34 critical bugs fixed
│   └── 2025-11-05-scripts-audit.md                      # 13 scripts analyzed
│
├── investigations/                                        # One-off debugging sessions (3 docs)
│   ├── 2025-11-03-mps-memory-leak.md                    # Apple Silicon MPS fix
│   ├── 2025-11-06-p0-semaphore-leak.md                  # Double model loading bug
│   └── p0-blockers.md                                    # Complete quality audit (P0/P1/P2/P3)
│
├── migrations/                                            # Codebase reorganizations (3 docs)
│   ├── 2025-11-05-codebase-reorganization.md            # v2.0.0 package structure
│   ├── 2025-11-06-repository-modernization.md           # uv, ruff, mypy strict
│   └── v2-structure-migration.md                         # Import conventions guide
│
├── plans/                                                 # Completed planning documents (1 doc)
│   └── DOCUMENTATION_STRUCTURE_PLAN.md                   # Docs reorganization (Phases 0-8)
│
├── summaries/                                             # Completion reports, status summaries (4 docs)
│   ├── 2025-11-02-fixes-applied.md                      # StandardScaler removal (+12.77% accuracy)
│   ├── 2025-11-02-phase1-test-results.md                # Experiment results
│   ├── 2025-11-02-training-setup-status.md              # Pipeline ready milestone
│   └── 2025-11-06-type-checking-complete.md             # 100% type safety achieved
│
└── trash/                                                 # Obsolete/redundant docs (safe to delete)
    ├── CLEANUP_COMPLETE_SUMMARY.md                       # Superseded by canonical docs
    ├── CLEANUP_PLAN.md                                   # Superseded by completion
    ├── DOCS_AUDIT_STATUS.md                              # Superseded by DOCUMENTATION_STRUCTURE_PLAN.md
    ├── STRICT_QC_CLEANUP_PLAN.md                         # Never executed (experimental)
    └── TEST_DATASETS_REORGANIZATION_PLAN.md              # Never executed (obsolete plan)
```

**Total:** 14 active documents + 5 trash (19 total)

---

## Document Categories

### 📋 Audits (2 docs)

Comprehensive code, security, and quality audits:

- **2025-11-11-production-readiness-audit.md** (886 lines)
  Exceptional technical audit documenting 34 critical bugs fixed. Valuable reference for security practices, data validation patterns, and production readiness checklist.

- **2025-11-05-scripts-audit.md** (451 lines)
  Comprehensive audit of 13 scripts across analysis/testing/training/validation. Documents what was experimental vs production.

### 🔍 Investigations (3 docs)

One-off debugging sessions and bug investigations:

- **2025-11-03-mps-memory-leak.md** (203 lines)
  Critical P0 bug fix for Apple Silicon. Documents why Harvey (141k sequences) crashed and the one-line fix. Still referenced in user-guide/troubleshooting.md.

- **2025-11-06-p0-semaphore-leak.md** (265 lines)
  Documents double ESM model loading bug that caused Harvey crashes. Excellent example of object lifecycle debugging.

- **p0-blockers.md** (615 lines)
  Comprehensive P0/P1/P2/P3 blocker tracking. Documents code quality audit with 32 type errors fixed, 90.82% coverage achieved.

### 🚀 Migrations (3 docs)

Codebase reorganizations and structural changes:

- **2025-11-05-codebase-reorganization.md** (419 lines)
  Documents v2.0.0 package structure migration (root files → src/). Critical reference for understanding import patterns and professional package organization.

- **2025-11-06-repository-modernization.md** (918 lines)
  Comprehensive 2025 tooling upgrade plan (uv, ruff, mypy strict). Excellent reference for modern Python best practices.

- **v2-structure-migration.md** (401 lines)
  Import conventions guide for v2.0.0 package structure. Documents breaking changes from root imports.

### 📝 Plans (1 doc)

Completed planning documents:

- **DOCUMENTATION_STRUCTURE_PLAN.md** (935 lines)
  Massive planning document for docs reorganization (Phases 0-8). Successfully executed plan creating user-guide/, developer-guide/, research/ structure.

### ✅ Summaries (4 docs)

Completion reports and status summaries:

- **2025-11-02-fixes-applied.md** (148 lines)
  Documents critical StandardScaler removal that improved Jain accuracy from 55.32% → 68.09%. Historical record of Novo methodology alignment.

- **2025-11-02-phase1-test-results.md** (36 lines)
  Short results summary from StandardScaler removal experiment. Numerical validation of hypothesis.

- **2025-11-02-training-setup-status.md** (164 lines)
  Documents OSS repo being "fully wired and working" with Boughter data. Historical milestone.

- **2025-11-06-type-checking-complete.md** (168 lines)
  Documents achieving 100% type safety (75 errors fixed). Shows systematic type error resolution approach.

### 🗑️ Trash (5 docs - safe to delete)

Obsolete, redundant, or never-executed planning documents:

- **CLEANUP_COMPLETE_SUMMARY.md** - Superseded by JAIN_COMPLETE_GUIDE.md (canonical)
- **CLEANUP_PLAN.md** - Superseded by completion
- **DOCS_AUDIT_STATUS.md** - Superseded by DOCUMENTATION_STRUCTURE_PLAN.md
- **STRICT_QC_CLEANUP_PLAN.md** - Never executed (experimental strict_qc)
- **TEST_DATASETS_REORGANIZATION_PLAN.md** - Never executed (Jain reorg plan)

---

## Archive Criteria

A document belongs in `archive/` if it meets ALL of these criteria:

1. ✅ **Complete** - The work described is finished
2. ✅ **Time-bound** - Represents a specific point in time or completed project
3. ✅ **Historical** - Provides context for past decisions but not current operations
4. ✅ **Superseded** - Information may be outdated or replaced by current practices

**Example:** `2025-11-03-mps-memory-leak.md` is archived because the bug is fixed, the investigation is complete, and the fix is merged. The knowledge is valuable for historical context but not needed for daily operations.

---

## Current Active Documentation

For current, evergreen documentation, see:

- **User Guides:** `docs/user-guide/` (installation, training, testing, troubleshooting)
- **Developer Guides:** `docs/developer-guide/` (architecture, workflow, testing, CI/CD, security)
- **Research Notes:** `docs/research/` (methodology, Novo parity, benchmarks)
- **Dataset Documentation:** `docs/datasets/` (Boughter, Jain, Harvey, Shehata)

---

## Reorganization History

**2025-11-17:** Complete archive reorganization
- Organized 14 root-level files into 5 subdirectories (audits, investigations, migrations, plans, summaries)
- Added date prefixes to all files for chronological clarity
- Moved 5 obsolete/redundant docs to trash/ (26% reduction)
- All documents now follow consistent naming: `YYYY-MM-DD-descriptive-name.md`

**Before:** 14 root files + 5 subdirectory files (19 total)
**After:** 1 root file (README.md) + 14 organized files + 5 trash (19 total)

---

**Last Updated:** 2025-11-17
**Branch:** `leroy-jenkins/full-send`
