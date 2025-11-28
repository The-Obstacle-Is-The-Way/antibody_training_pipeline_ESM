# Archive

> **Note:** Some documents in this archive reference `leroy-jenkins/full-send` which was renamed to `main` on 2025-11-28.

This directory contains **historical documentation** from the development process. These documents represent completed work, investigations, and planning efforts that provide context for past decisions.

**Archive Policy:** Documents here are time-bound, completed, and serve as historical reference only. For current documentation, see the main `docs/` directory.

---

## Directory Structure

All archive documents are organized into 5 categories:

```
archive/
├── README.md                                              # This file
│
├── audits/                                                # Code audits, security audits, quality audits (3 docs)
│   ├── 2025-11-19-xgboost-branch-audit.md               # XGBoost Strategy Pattern integration validation
│   ├── 2025-11-11-production-readiness-audit.md         # 34 critical bugs fixed
│   └── 2025-11-05-scripts-audit.md                      # 13 scripts analyzed
│
├── investigations/                                        # One-off debugging sessions (7 docs)
│   ├── cli-test-refactor-2025-11-18.md                  # CLI test.py modularization validation
│   ├── dataset-column-naming-2025-11-18.md              # Canonical vs fragment files design
│   ├── 2025-11-03-mps-memory-leak.md                    # Apple Silicon MPS fix
│   ├── 2025-11-06-p0-semaphore-leak.md                  # Double model loading bug
│   ├── 2025-11-11-cli-override-bug.md                   # ConfigStore conflicting with YAML configs
│   ├── 2025-11-11-training-pipeline-fixes.md            # Embedding cache + CLI entry point fixes
│   └── p0-blockers.md                                    # Complete quality audit (P0/P1/P2/P3)
│
├── summaries/                                             # Completion reports, status summaries (6 docs)
│   ├── inference-completion-2025-11-19.md                # Inference pipeline production ready
│   ├── 2025-11-02-fixes-applied.md                      # StandardScaler removal (+12.77% accuracy)
│   ├── 2025-11-02-phase1-test-results.md                # Experiment results
│   ├── 2025-11-02-training-setup-status.md              # Pipeline ready milestone
│   ├── 2025-11-06-type-checking-complete.md             # 100% type safety achieved
│   └── 2025-11-12-esm2-feature.md                        # ESM2-650M backbone support
│
├── decisions/                                             # Architectural decisions (1 doc)
│   └── preprocessing-location-decision-2025-11-18.md     # Factory vs Product separation (keep at root)
│
├── migrations/                                            # Codebase reorganizations (3 docs)
│   ├── 2025-11-05-codebase-reorganization.md            # v2.0.0 package structure
│   ├── 2025-11-06-repository-modernization.md           # uv, ruff, mypy strict
│   └── v2-structure-migration.md                         # Import conventions guide
│
├── plans/                                                 # Completed planning documents (2 docs)
│   ├── 2025-11-11-output-pipeline-architecture.md        # Hierarchical output organization design
│   └── DOCUMENTATION_STRUCTURE_PLAN.md                   # Docs reorganization (Phases 0-8)
│
└── trash/                                                 # Obsolete/redundant docs (2 files)
    ├── refactor-test-cli-plan.md                         # Superseded by cli-test-refactor validation
    └── spec-sheet.md                                      # Superseded by inference completion report
```

**Total:** 24 documents (22 active + 2 trash)

---

## Document Categories

### 📋 Audits (3 docs)

Comprehensive code, security, and quality audits:

- **2025-11-19-xgboost-branch-audit.md** (448 lines)
  Technical audit of XGBoost Strategy Pattern integration. Validates 518 passing tests, 90% coverage, complete backward compatibility. **Identified critical issue:** hardcoded default values creating dual source of truth with YAML configs. Documents conditional approval with mandatory fixes before merge.

- **2025-11-11-production-readiness-audit.md** (886 lines)
  Exceptional technical audit documenting 34 critical bugs fixed. Valuable reference for security practices, data validation patterns, and production readiness checklist.

- **2025-11-05-scripts-audit.md** (451 lines)
  Comprehensive audit of 13 scripts across analysis/testing/training/validation. Documents what was experimental vs production.

### 🔍 Investigations (7 docs)

One-off debugging sessions and bug investigations:

- **cli-test-refactor-2025-11-18.md** (328 lines)
  Validates CLI test.py refactoring (872→141 lines, 83.8% reduction). Documents 6-module package structure with comprehensive end-to-end testing. All 468 tests passing, type safety maintained, zero anti-patterns detected. **Verdict:** Production ready, fully validated.

- **dataset-column-naming-2025-11-18.md** (402 lines)
  Root cause analysis of "sequence column not found" error. **Conclusion:** Working as designed - canonical files use `vh_sequence` (research integrity), fragment files use `sequence` (CLI convenience). Documents Factory vs Product separation for dataset types. Added `--sequence-column` CLI flag for flexibility.

- **2025-11-03-mps-memory-leak.md** (203 lines)
  Critical P0 bug fix for Apple Silicon. Documents why Harvey (141k sequences) crashed and the one-line fix. Still referenced in user-guide/troubleshooting.md.

- **2025-11-06-p0-semaphore-leak.md** (265 lines)
  Documents double ESM model loading bug that caused Harvey crashes. Excellent example of object lifecycle debugging.

- **2025-11-11-cli-override-bug.md** (392 lines)
  Root cause analysis of critical bug where `antibody-train model=esm2_650m` was ignored due to ConfigStore/YAML conflicts. Fixed by commenting out ConfigStore registrations. Lessons learned about Hydra's automatic schema matching deprecation.

- **2025-11-11-training-pipeline-fixes.md** (182 lines)
  Multi-issue investigation documenting 4 critical fixes: CLI entry point (config group overrides), embedding cache collision (model metadata in cache key), Hydra deprecation warnings, and log directory creation.

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

### 📝 Plans + Specs (6 docs)

Completed planning documents (kept for historical context):

- **2025-11-11-output-pipeline-architecture.md** (816 lines)
  Hierarchical output organization design and implementation plan. Documents the problem (file collisions when testing multiple models), solution (stratified outputs by backbone/classifier/dataset), and complete implementation via `directory_utils.py`. Critical reference for understanding the experiments/ hierarchy.

- **DOCUMENTATION_STRUCTURE_PLAN.md** (935 lines)
  Massive planning document for docs reorganization (Phases 0-8). Successfully executed plan creating user-guide/, developer-guide/, research/ structure.

- **xgboost-integration-spec.md** (655 lines)
  Technical specification that scoped the XGBoost classifier effort (motivation, risks, success metrics). Superseded by `docs/developer-guide/xgboost.md` now that implementation is complete.

- **xgboost-api-design.md** (1,085 lines)
  Detailed API design for the strategy/factory pattern. Shows UML-style breakdown of protocols, factories, and serialization hooks. Historical blueprint used before implementation.

- **xgboost-test-plan.md** (1,013 lines)
  Exhaustive test coverage plan (unit, integration, E2E, performance). Use for auditing past decisions; live tests reside under `tests/*`.

- **xgboost-implementation-status.md** (273 lines)
  Rolling status tracker for Phase 1 delivery (roadmap tasks, branches, checkpoints). Documentation of the now-complete execution phase.

### ✅ Summaries (6 docs)

Completion reports and status summaries:

- **inference-completion-2025-11-19.md** (44 lines)
  Production readiness certification for `antibody-predict` CLI and `Predictor` core class. Documents critical fixes: documentation accuracy (gap characters NOT supported), error handling improvements, resource optimization (reuse ESM extractor), E2E test markers. **Verdict:** Gucci banger status, ready for Gradio/Web UI integration.

- **2025-11-02-fixes-applied.md** (148 lines)
  Documents critical StandardScaler removal that improved Jain accuracy from 55.32% → 68.09%. Historical record of Novo methodology alignment.

- **2025-11-02-phase1-test-results.md** (36 lines)
  Short results summary from StandardScaler removal experiment. Numerical validation of hypothesis.

- **2025-11-02-training-setup-status.md** (164 lines)
  Documents OSS repo being "fully wired and working" with Boughter data. Historical milestone.

- **2025-11-06-type-checking-complete.md** (168 lines)
  Documents achieving 100% type safety (75 errors fixed). Shows systematic type error resolution approach.

- **2025-11-12-esm2-feature.md** (223 lines)
  Feature completion summary for ESM2-650M backbone support. Documents config file addition (`model/esm2_650m.yaml`), usage via `antibody-train model=esm2_650m`, and benchmark comparison plan (ESM-1v vs ESM2). Includes research context on why ESM-1v likely outperforms ESM2 on antibody tasks.

### 🏛️ Decisions (1 doc)

Architectural decision records (ADRs):

- **preprocessing-location-decision-2025-11-18.md** (477 lines)
  Comprehensive architectural analysis of preprocessing directory location. **Decision:** Keep at root permanently (Factory vs Product separation). Documents why ETL scripts (one-time data manufacturing) should NOT be bundled with runtime code (data usage). Industry pattern analysis, trade-offs, and when to reconsider. Critical reference for understanding why `preprocessing/` lives at project root, not in `src/`.

### 🗑️ Trash (2 files - safe to delete)

Obsolete/redundant planning documents superseded by completion reports:

- **refactor-test-cli-plan.md** (63 lines) - Superseded by cli-test-refactor-2025-11-18.md (complete validation report)
- **spec-sheet.md** (80 lines) - Superseded by inference-completion-2025-11-19.md (production certification)

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

**2025-11-19 (morning):** Documentation cleanup & integration (Phase 4 complete)
- Archived 4 completed docs: XGBoost audit, CLI refactor investigation, column naming investigation, inference completion
- Integrated 1 architectural decision: preprocessing location (Factory vs Product)
- Moved 2 obsolete docs to trash: refactor-test-cli-plan.md, spec-sheet.md
- Created 2 new canonical docs: user-guide/inference.md, research/model-zoo-roadmap.md
- Enhanced developer-guide/architecture.md with preprocessing design section
- **Archive totals:** 22 active documents + 2 trash (24 total) ← **current state**

**2025-11-17 (evening):** Trash cleanup
- Deleted 5 obsolete trash documents (user action)
- **Archive totals:** 17 active documents + 0 trash (17 total)

**2025-11-17 (afternoon):** Integration of to-be-integrated docs
- Moved 4 completed investigation/feature docs from `to-be-integrated/` to archive with Phase 5 annotations
- Integrated evergreen knowledge into canonical documentation:
  - ESM2 model selection → `docs/user-guide/training.md`
  - Config group override troubleshooting → `docs/user-guide/troubleshooting.md`
  - Embedding cache behavior → `docs/developer-guide/architecture.md`
  - Hierarchical output structure → Already documented in `docs/developer-guide/directory-organization.md`
- Deleted `to-be-integrated/` directory after integration
- **Archive totals:** 17 active documents + 5 trash (22 total, +4 from integration)

**2025-11-17 (morning):** Complete archive reorganization
- Organized 14 root-level files into 5 subdirectories (audits, investigations, migrations, plans, summaries)
- Added date prefixes to all files for chronological clarity
- Moved 5 obsolete/redundant docs to trash/ (28% reduction: 18→13 active)
- All documents now follow consistent naming: `YYYY-MM-DD-descriptive-name.md`
- **Archive totals:** 13 active documents + 5 trash (18 total)

---

**Last Updated:** 2025-11-17
**Branch:** `leroy-jenkins/full-send`
