# Documentation Structure Plan

**Created:** 2025-11-09
**Branch:** docs/canonical-structure
**Status:** 📋 Proposed - Awaiting Senior Review

---

## Executive Summary

This document proposes a canonical documentation structure for the antibody training pipeline repository. The goal is to organize ~15,000 lines of existing documentation (across 4 directories + 44 dataset docs) into a logical, discoverable hierarchy that serves multiple audiences: **users**, **developers**, **researchers**, and **maintainers**.

**Current State:**
- 📁 `docs/archive/` - 14 files (~4,300 lines) - historical cleanup/migration plans
- 📁 `docs/development/` - 15 files (~7,100 lines) - CI/CD, security, testing, Docker
- 📁 `docs/research/` - 9 files (~3,150 lines) - Novo parity analysis, methodology
- 📁 `docs/datasets/` - 44 files (4 subdirs) - dataset-specific preprocessing docs
- 📄 `docs/ESM1V_ENSEMBLING_INVESTIGATION.md` - linked externally, must stay in root
- 📄 `docs/README.md` - outdated index

**Problems:**
1. **No clear entry points** - users don't know where to start
2. **Audience confusion** - mixing user guides with internal dev plans
3. **Temporal documentation** - archive docs mixed with evergreen guides
4. **Discoverability** - hard to find CI/CD setup vs research methodology
5. **Redundancy** - overlapping content between development/ and archive/

**Proposed Solution:**
Reorganize into **5 top-level categories** with clear audience separation:

```
docs/
├── README.md                          # Clear navigation hub
├── ESM1V_ENSEMBLING_INVESTIGATION.md  # Keep in root (externally linked)
│
├── user-guide/                        # For end users
│   ├── getting-started.md
│   ├── installation.md
│   ├── training.md
│   ├── testing.md
│   ├── preprocessing.md
│   └── troubleshooting.md
│
├── developer-guide/                   # For contributors
│   ├── architecture.md
│   ├── development-workflow.md
│   ├── testing-strategy.md
│   ├── type-checking.md
│   ├── ci-cd.md
│   ├── docker.md
│   └── security.md
│
├── datasets/                          # Dataset-specific docs (keep as-is)
│   ├── boughter/
│   ├── jain/
│   ├── harvey/
│   └── shehata/
│
├── research/                          # Scientific methodology
│   ├── novo-parity.md
│   ├── methodology.md
│   ├── assay-thresholds.md
│   └── benchmark-results.md
│
└── archive/                           # Historical/completed work
    ├── README.md                      # Archive index
    ├── migrations/                    # Codebase reorganizations
    ├── investigations/                # One-off debugging sessions
    └── plans/                         # Completed planning docs
```

---

## Design Principles

### 1. Audience-First Organization
Each top-level directory serves a distinct persona:

| Directory | Audience | Purpose | Style |
|-----------|----------|---------|-------|
| `user-guide/` | Users running the pipeline | How to accomplish tasks | Task-oriented, imperative |
| `developer-guide/` | Contributors writing code | How to contribute | Conceptual + procedural |
| `datasets/` | Data scientists/bioinformaticians | Dataset provenance + preprocessing | Reference documentation |
| `research/` | Researchers validating methodology | Scientific reproducibility | Academic, analysis-heavy |
| `archive/` | Maintainers/historians | Context for past decisions | Timestamped, read-only |

### 2. Progressive Disclosure
- **Level 1:** `docs/README.md` - navigation hub, 1-2 sentences per category
- **Level 2:** Category READMEs (e.g., `user-guide/README.md`) - overview + quick links
- **Level 3:** Topic-specific guides (e.g., `user-guide/training.md`) - detailed instructions

### 3. DRY (Don't Repeat Yourself)
- **Single source of truth** for each topic
- **Cross-references** instead of duplication (e.g., "See `developer-guide/ci-cd.md` for details")
- **Consolidate overlapping docs** (e.g., merge 3 CI/CD docs into one canonical guide)

### 4. Temporal Separation
- **Evergreen docs** (user/developer/research guides) → live at top level
- **Time-bound docs** (cleanup plans, migration logs) → `archive/`
- **Archive policy:** Document is complete, historical context only

### 5. Discoverability
- **Clear naming:** `getting-started.md` not `SETUP.md`
- **Standardized structure:** All guides follow same sections (Overview, Prerequisites, Steps, Troubleshooting)
- **Cross-linking:** Every doc links to related guides
- **Search-friendly:** Keywords in first paragraph

---

## Proposed Structure (Detailed)

### `docs/README.md` (Navigation Hub)

```markdown
# Documentation

## 🚀 Getting Started
New to the pipeline? Start here:
- [Installation Guide](user-guide/installation.md)
- [Quick Start Tutorial](user-guide/getting-started.md)

## 📖 User Guides
- [Training Models](user-guide/training.md)
- [Testing on New Datasets](user-guide/testing.md)
- [Preprocessing Datasets](user-guide/preprocessing.md)
- [Troubleshooting](user-guide/troubleshooting.md)

## 🛠️ Developer Guides
- [Architecture Overview](developer-guide/architecture.md)
- [Development Workflow](developer-guide/development-workflow.md)
- [CI/CD Pipeline](developer-guide/ci-cd.md)
- [Testing Strategy](developer-guide/testing-strategy.md)
- [Type Checking](developer-guide/type-checking.md)
- [Docker Usage](developer-guide/docker.md)
- [Security Guidelines](developer-guide/security.md)

## 📊 Datasets
- [Boughter (Training)](datasets/boughter/)
- [Jain (Test - Novo Parity)](datasets/jain/)
- [Harvey (Nanobodies)](datasets/harvey/)
- [Shehata (PSR Assay)](datasets/shehata/)

## 🔬 Research Notes
- [Novo Nordisk Parity Analysis](research/novo-parity.md)
- [Methodology & Divergences](research/methodology.md)
- [Assay-Specific Thresholds](research/assay-thresholds.md)
- [Benchmark Results](research/benchmark-results.md)

## 📦 Archive
Historical documentation for completed work:
- [Codebase Migrations](archive/migrations/)
- [Debugging Investigations](archive/investigations/)
- [Completed Plans](archive/plans/)

See [archive/README.md](archive/README.md) for details.

---

**Special Note:** [ESM1V Ensembling Investigation](ESM1V_ENSEMBLING_INVESTIGATION.md) (externally linked)
```

---

### `user-guide/` (6 Files)

**Target Audience:** Users running the pipeline (not modifying code)

**Files to Create:**

1. **`getting-started.md`** (NEW - consolidate from README.md + CLAUDE.md)
   - 5-minute quickstart
   - Single training run
   - Verify installation works

2. **`installation.md`** (NEW - extract from README.md)
   - System requirements
   - uv installation
   - Environment setup
   - Dependency installation

3. **`training.md`** (NEW - extract from CLAUDE.md)
   - Config file structure
   - Running `antibody-train`
   - Hyperparameter tuning
   - Model checkpoints

4. **`testing.md`** (NEW - extract from CLAUDE.md)
   - Running `antibody-test`
   - Evaluating on test sets
   - Fragment-level testing
   - Interpreting results

5. **`preprocessing.md`** (NEW - consolidate from dataset READMEs)
   - Overview of preprocessing pipelines
   - When to preprocess
   - Links to dataset-specific guides

6. **`troubleshooting.md`** (NEW)
   - Common errors + fixes
   - MPS memory issues
   - Cache invalidation
   - Test failures

**Content Sources:**
- Extract from `README.md` (sections: Installation, Developer Workflow)
- Extract from `CLAUDE.md` (sections: Development Commands, Training & Testing)
- Extract from `docs/development/DOCKER_USAGE.md` (user-facing Docker commands)
- Archive `docs/archive/MPS_MEMORY_LEAK_FIX.md` → migrate fix to troubleshooting

---

### `developer-guide/` (7 Files)

**Target Audience:** Contributors modifying/extending the codebase

**Files to Create:**

1. **`architecture.md`** (NEW - extract from CLAUDE.md)
   - Core pipeline flow
   - Module responsibilities
   - Directory structure
   - Design patterns (caching, config, fragments)

2. **`development-workflow.md`** (NEW - consolidate)
   - Git workflow (branching, commits, PRs)
   - Code quality tools (ruff, mypy, pytest)
   - Pre-commit hooks
   - `make` commands
   - **Sources:**
     - `CLAUDE.md` (Git Workflow section)
     - `README.md` (Developer Workflow section)
     - `docs/development/BRANCH_PROTECTION_WALKTHROUGH.md`

3. **`testing-strategy.md`** (CONSOLIDATE from 3 docs)
   - Test markers (unit/integration/e2e)
   - Coverage requirements
   - Mocking strategy
   - Writing new tests
   - **Sources:**
     - `docs/development/TEST_SUITE_PLAN.md`
     - `docs/development/TEST_SUITE_REVIEW_CHECKLIST.md`
     - `docs/development/TEST_COVERAGE_GAPS.md`
     - `CLAUDE.md` (Testing Strategy section)

4. **`type-checking.md`** (CONSOLIDATE from 2 docs)
   - mypy configuration
   - Type annotation requirements
   - Common type errors
   - **Sources:**
     - `docs/development/TYPE_CHECKING_STRATEGY.md`
     - `docs/development/TYPE_HINTING_REMEDIATION_PLAN.md`
     - `CLAUDE.md` (Type Safety section)

5. **`ci-cd.md`** (CONSOLIDATE from 3 docs)
   - CI pipeline overview
   - GitHub Actions workflows
   - Quality gates
   - Branch protection
   - **Sources:**
     - `docs/development/CICD_SPEC.md`
     - `docs/development/CICD_SETUP_GUIDE.md`
     - `docs/development/BRANCH_PROTECTION_WALKTHROUGH.md`

6. **`docker.md`** (CONSOLIDATE from 2 docs)
   - Local Docker usage
   - CI Docker builds
   - GHCR publishing
   - **Sources:**
     - `docs/development/DOCKER_USAGE.md`
     - `docs/development/DOCKER_DEPLOYMENT.md`

7. **`security.md`** (CONSOLIDATE from 2 docs)
   - Pickle usage policy
   - Bandit scanning
   - CodeQL findings
   - Security best practices
   - **Sources:**
     - `docs/development/SECURITY_REMEDIATION_PLAN.md`
     - `docs/development/CODEQL_FINDINGS.md`
     - `README.md` (Security section)

**Files to Archive:**
- `docs/development/IMPORT_AND_STRUCTURE_GUIDE.md` → `archive/migrations/v2-structure-migration.md`
- `docs/development/P0_P1_P2_P3_BLOCKERS.md` → `archive/investigations/p0-blockers.md` (if completed)
- `docs/development/excel_to_csv_conversion_methods.md` → `datasets/` (if dataset-specific) OR `developer-guide/preprocessing-internals.md`

---

### `datasets/` (Keep Existing Structure)

**Target Audience:** Data scientists, bioinformaticians

**Action:** Keep as-is (already well-organized by dataset)

**Structure:**
```
datasets/
├── boughter/
│   ├── README.md
│   └── [dataset-specific docs]
├── jain/
│   ├── README.md
│   └── [dataset-specific docs]
├── harvey/
│   ├── README.md
│   └── [dataset-specific docs]
└── shehata/
    ├── README.md
    └── [dataset-specific docs]
```

**Minor Improvements:**
- Ensure each dataset has a `README.md` with:
  - Data source + citation
  - Preprocessing steps
  - File locations
  - Known issues
- Move `archive/` subdirectories inside datasets to `datasets/{name}/archive/`

---

### `research/` (4 Canonical Files)

**Target Audience:** Researchers validating methodology

**Files to Create:**

1. **`novo-parity.md`** (CONSOLIDATE from 3 docs)
   - Executive summary of Novo Nordisk reverse engineering
   - Exact match results ([[40, 19], [10, 17]])
   - Links to experiments/novo_parity/
   - **Sources:**
     - `docs/research/NOVO_PARITY_ANALYSIS.md`
     - `docs/research/CODEBASE_AUDIT_VS_NOVO.md`
     - `docs/research/NOVO_REPLICATION_PLAN.md`

2. **`methodology.md`** (CONSOLIDATE from 3 docs)
   - Pipeline methodology (ESM → LogReg)
   - Training procedure (10-fold CV)
   - Testing procedure (hold-out sets)
   - Divergences from paper
   - **Sources:**
     - `docs/research/METHODOLOGY_AND_DIVERGENCES.md`
     - `docs/research/NOVO_TRAINING_METHODOLOGY.md`
     - `docs/research/CRITICAL_IMPLEMENTATION_ANALYSIS.md`

3. **`assay-thresholds.md`** (KEEP)
   - ELISA vs PSR thresholds
   - Novo Nordisk exact parity (0.5495)
   - **Source:** `docs/research/ASSAY_SPECIFIC_THRESHOLDS.md`

4. **`benchmark-results.md`** (CONSOLIDATE from 2 docs)
   - Cross-dataset validation results
   - Fragment-level performance
   - Comparison to paper
   - **Sources:**
     - `docs/research/BENCHMARK_TEST_RESULTS.md`
     - `docs/research/COMPLETE_VALIDATION_RESULTS.md`

**Files to Archive:**
- All original files move to `archive/research/` with timestamps

---

### `archive/` (Reorganize by Type)

**Target Audience:** Maintainers, historians

**Proposed Structure:**
```
archive/
├── README.md                                    # Archive index with timestamps
│
├── migrations/                                  # Codebase reorganizations
│   ├── v2-structure-migration.md               # (from IMPORT_AND_STRUCTURE_GUIDE.md)
│   ├── codebase-reorganization-2025-11-05.md   # (from CODEBASE_REORGANIZATION_PLAN.md)
│   ├── repository-modernization-2025-11-06.md  # (from REPOSITORY_MODERNIZATION_PLAN.md)
│   └── test-datasets-reorganization.md         # (from TEST_DATASETS_REORGANIZATION_PLAN.md)
│
├── investigations/                              # One-off debugging sessions
│   ├── mps-memory-leak-2025-11-03.md           # (from MPS_MEMORY_LEAK_FIX.md)
│   ├── p0-semaphore-leak-2025-11-05.md         # (from P0_SEMAPHORE_LEAK.md)
│   ├── p0-p1-p2-p3-blockers.md                 # (from P0_P1_P2_P3_BLOCKERS.md)
│   ├── residual-type-errors.md                 # (from RESIDUAL_TYPE_ERRORS.md)
│   └── scripts-audit-2025-11-05.md             # (from SCRIPTS_AUDIT.md)
│
├── plans/                                       # Completed planning docs
│   ├── cleanup-plan-2025-11-05.md              # (from CLEANUP_PLAN.md)
│   ├── strict-qc-cleanup-plan.md               # (from STRICT_QC_CLEANUP_PLAN.md)
│   └── training-setup-status.md                # (from TRAINING_SETUP_STATUS.md)
│
└── summaries/                                   # Completion reports
    ├── cleanup-complete-2025-11-05.md          # (from CLEANUP_COMPLETE_SUMMARY.md)
    ├── docs-audit-status.md                    # (from DOCS_AUDIT_STATUS.md)
    ├── fixes-applied.md                        # (from FIXES_APPLIED.md)
    └── phase1-test-results.md                  # (from PHASE1_TEST_RESULTS.md)
```

**Archive Policy:**
- Document must be **complete** (not in-progress)
- Document is **time-bound** (planning/investigation that ended)
- Document has **historical value** (explains past decisions)
- All archive files **must have dates in filename or header**

**Files to Delete (If Redundant):**
- Consider deleting `DOCS_AUDIT_STATUS.md` (superseded by this plan)

---

## Industry Standards Reference

### Documentation Best Practices (2025)

This plan follows modern documentation frameworks:

1. **Diátaxis Framework** (https://diataxis.fr/)
   - **Tutorials** → `user-guide/getting-started.md`
   - **How-To Guides** → `user-guide/training.md`, `user-guide/testing.md`
   - **Reference** → `datasets/`, API docs (future)
   - **Explanation** → `research/methodology.md`, `developer-guide/architecture.md`

2. **Write the Docs Best Practices**
   - Progressive disclosure (README → category → topic)
   - Task-oriented organization
   - Clear navigation
   - Avoid jargon in user guides

3. **Example: Well-Documented ML Projects**
   - **HuggingFace Transformers:** Clear user/dev split, quickstart + deep dives
   - **PyTorch:** Tutorials, API reference, community (3 top-level categories)
   - **FastAPI:** Tutorial, User Guide, Advanced (progressive disclosure)

---

## Implementation Roadmap

### Phase 1: Prepare Archive (1-2 hours)
- [ ] Create `archive/migrations/`, `archive/investigations/`, `archive/plans/`, `archive/summaries/`
- [ ] Move 14 archive docs to appropriate subdirectories with date prefixes
- [ ] Write `archive/README.md` with index + archive policy

### Phase 2: Create User Guide (3-4 hours)
- [ ] Extract installation steps from `README.md` → `user-guide/installation.md`
- [ ] Extract quickstart from `README.md` + `CLAUDE.md` → `user-guide/getting-started.md`
- [ ] Extract training workflow from `CLAUDE.md` → `user-guide/training.md`
- [ ] Extract testing workflow from `CLAUDE.md` → `user-guide/testing.md`
- [ ] Create `user-guide/preprocessing.md` (overview + links to dataset docs)
- [ ] Create `user-guide/troubleshooting.md` (MPS leak, common errors)

### Phase 3: Create Developer Guide (4-5 hours)
- [ ] Extract architecture from `CLAUDE.md` → `developer-guide/architecture.md`
- [ ] Consolidate git workflow docs → `developer-guide/development-workflow.md`
- [ ] Merge 3 testing docs → `developer-guide/testing-strategy.md`
- [ ] Merge 2 type checking docs → `developer-guide/type-checking.md`
- [ ] Merge 3 CI/CD docs → `developer-guide/ci-cd.md`
- [ ] Merge 2 Docker docs → `developer-guide/docker.md`
- [ ] Merge 2 security docs → `developer-guide/security.md`

### Phase 4: Consolidate Research (2-3 hours)
- [ ] Merge Novo parity docs → `research/novo-parity.md`
- [ ] Merge methodology docs → `research/methodology.md`
- [ ] Keep `research/assay-thresholds.md` as-is
- [ ] Merge benchmark docs → `research/benchmark-results.md`
- [ ] Move originals to `archive/research/`

### Phase 5: Update Navigation (1 hour)
- [ ] Rewrite `docs/README.md` with clear navigation hub (see template above)
- [ ] Update root `README.md` to link to `docs/` correctly
- [ ] Update `CLAUDE.md` to reference new docs paths
- [ ] Add cross-links between related guides

### Phase 6: Validation (1 hour)
- [ ] Verify all links work (use `markdown-link-check` or manual)
- [ ] Test navigation from user perspective (can I find training guide in <30s?)
- [ ] Ensure no broken references to old docs paths
- [ ] Run `make lint` on markdown files

**Total Estimated Effort:** 12-16 hours

---

## Success Metrics

### Quantitative
- ✅ **Reduce top-level doc count:** 40 docs → 6 categories + ESM1V
- ✅ **Consolidation ratio:** 15 development docs → 7 canonical guides (53% reduction)
- ✅ **Archive coverage:** 14/14 archive docs organized by type with dates
- ✅ **Navigation depth:** ≤3 clicks from docs/README.md to any guide

### Qualitative
- ✅ **New user can find installation in <30 seconds**
- ✅ **Contributor can find CI/CD guide without reading 3 separate docs**
- ✅ **Researcher can verify Novo parity methodology in single doc**
- ✅ **Maintainer can understand past decisions from archive**

### Maintainability
- ✅ **Single source of truth for each topic** (no duplicate CI/CD docs)
- ✅ **Clear ownership:** Each guide has a primary audience
- ✅ **Temporal separation:** Archive docs won't clutter active guides

---

## Open Questions for Senior Review

1. **Developer Guide Scope:**
   - Should `excel_to_csv_conversion_methods.md` go in `developer-guide/` or `datasets/`?
   - Current recommendation: If general preprocessing internals → `developer-guide/preprocessing-internals.md`
   - If Boughter-specific → move to `datasets/boughter/`

2. **Archive Deletion:**
   - Should we delete `DOCS_AUDIT_STATUS.md` (superseded by this plan)?
   - Recommendation: Yes, delete (redundant)

3. **Research Consolidation:**
   - Keep 4 research docs or merge into 2 (novo-parity + methodology)?
   - Current recommendation: 4 docs (clear separation of concerns)

4. **Dataset Archive Handling:**
   - Move `datasets/{name}/archive/` to top-level `archive/datasets/{name}/`?
   - Current recommendation: Keep in-place (dataset-specific context)

5. **External Links:**
   - Should `ESM1V_ENSEMBLING_INVESTIGATION.md` eventually move to `research/`?
   - Current answer: No, must stay in root (externally linked, breaking link would cause issues)

---

## Next Steps

1. **Senior Review:** Approve/modify this plan
2. **Create GitHub Issue:** Track implementation with checklist
3. **Branch:** Use `docs/canonical-structure` (already created)
4. **Execute Phases 1-6:** Follow roadmap
5. **PR Review:** Ensure navigation works before merge
6. **Announce:** Update team on new doc structure

---

## Appendix: File Mapping

### From `docs/development/` (15 files)
| Old File | New Location | Action |
|----------|--------------|--------|
| `BRANCH_PROTECTION_WALKTHROUGH.md` | `developer-guide/ci-cd.md` | Merge |
| `CICD_SETUP_GUIDE.md` | `developer-guide/ci-cd.md` | Merge |
| `CICD_SPEC.md` | `developer-guide/ci-cd.md` | Merge |
| `CODEQL_FINDINGS.md` | `developer-guide/security.md` | Merge |
| `DOCKER_DEPLOYMENT.md` | `developer-guide/docker.md` | Merge |
| `DOCKER_USAGE.md` | `developer-guide/docker.md` | Merge |
| `IMPORT_AND_STRUCTURE_GUIDE.md` | `archive/migrations/v2-structure-migration.md` | Archive |
| `P0_P1_P2_P3_BLOCKERS.md` | `archive/investigations/p0-blockers.md` | Archive |
| `SECURITY_REMEDIATION_PLAN.md` | `developer-guide/security.md` | Merge |
| `TEST_COVERAGE_GAPS.md` | `developer-guide/testing-strategy.md` | Merge |
| `TEST_SUITE_PLAN.md` | `developer-guide/testing-strategy.md` | Merge |
| `TEST_SUITE_REVIEW_CHECKLIST.md` | `developer-guide/testing-strategy.md` | Merge |
| `TYPE_CHECKING_STRATEGY.md` | `developer-guide/type-checking.md` | Merge |
| `TYPE_HINTING_REMEDIATION_PLAN.md` | `developer-guide/type-checking.md` | Merge |
| `excel_to_csv_conversion_methods.md` | `developer-guide/preprocessing-internals.md` OR `datasets/boughter/` | TBD |

### From `docs/archive/` (14 files)
| Old File | New Location | Action |
|----------|--------------|--------|
| `CLEANUP_COMPLETE_SUMMARY.md` | `archive/summaries/cleanup-complete-2025-11-05.md` | Rename |
| `CLEANUP_PLAN.md` | `archive/plans/cleanup-plan-2025-11-05.md` | Rename |
| `CODEBASE_REORGANIZATION_PLAN.md` | `archive/migrations/codebase-reorganization-2025-11-05.md` | Rename |
| `DOCS_AUDIT_STATUS.md` | DELETE (superseded) | Delete |
| `FIXES_APPLIED.md` | `archive/summaries/fixes-applied.md` | Rename |
| `MPS_MEMORY_LEAK_FIX.md` | `archive/investigations/mps-memory-leak-2025-11-03.md` | Rename |
| `P0_SEMAPHORE_LEAK.md` | `archive/investigations/p0-semaphore-leak-2025-11-05.md` | Rename |
| `PHASE1_TEST_RESULTS.md` | `archive/summaries/phase1-test-results.md` | Rename |
| `REPOSITORY_MODERNIZATION_PLAN.md` | `archive/migrations/repository-modernization-2025-11-06.md` | Rename |
| `RESIDUAL_TYPE_ERRORS.md` | `archive/investigations/residual-type-errors.md` | Rename |
| `SCRIPTS_AUDIT.md` | `archive/investigations/scripts-audit-2025-11-05.md` | Rename |
| `STRICT_QC_CLEANUP_PLAN.md` | `archive/plans/strict-qc-cleanup-plan.md` | Rename |
| `TEST_DATASETS_REORGANIZATION_PLAN.md` | `archive/migrations/test-datasets-reorganization.md` | Rename |
| `TRAINING_SETUP_STATUS.md` | `archive/plans/training-setup-status.md` | Rename |

### From `docs/research/` (9 files)
| Old File | New Location | Action |
|----------|--------------|--------|
| `ASSAY_SPECIFIC_THRESHOLDS.md` | `research/assay-thresholds.md` | Rename (lowercase) |
| `BENCHMARK_TEST_RESULTS.md` | `research/benchmark-results.md` | Merge |
| `CODEBASE_AUDIT_VS_NOVO.md` | `research/novo-parity.md` | Merge |
| `COMPLETE_VALIDATION_RESULTS.md` | `research/benchmark-results.md` | Merge |
| `CRITICAL_IMPLEMENTATION_ANALYSIS.md` | `research/methodology.md` | Merge |
| `METHODOLOGY_AND_DIVERGENCES.md` | `research/methodology.md` | Merge |
| `NOVO_PARITY_ANALYSIS.md` | `research/novo-parity.md` | Merge |
| `NOVO_REPLICATION_PLAN.md` | `research/novo-parity.md` | Merge |
| `NOVO_TRAINING_METHODOLOGY.md` | `research/methodology.md` | Merge |

---

**End of Plan**

**Approval Checklist:**
- [ ] Senior review complete
- [ ] Open questions resolved
- [ ] Implementation roadmap approved
- [ ] Ready to execute
