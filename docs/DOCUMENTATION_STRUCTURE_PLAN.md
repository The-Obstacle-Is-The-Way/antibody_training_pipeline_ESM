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
Reorganize into **6 top-level categories** with clear audience separation + system overview:

```text
docs/
├── README.md                          # Navigation hub
├── overview.md                        # NEW: "What is this system?" (architecture, components)
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
│   ├── security.md
│   └── preprocessing-internals.md     # NEW: excel_to_csv + general preprocessing
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
    ├── README.md                      # Archive index (CREATE FIRST)
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
| `overview.md` | Everyone (first-time readers) | System architecture + components | High-level, conceptual |
| `user-guide/` | Users running the pipeline | How to accomplish tasks | Task-oriented, imperative |
| `developer-guide/` | Contributors writing code | How to contribute | Conceptual + procedural |
| `datasets/` | Data scientists/bioinformaticians | Dataset provenance + preprocessing | Reference documentation |
| `research/` | Researchers validating methodology | Scientific reproducibility | Academic, analysis-heavy |
| `archive/` | Maintainers/historians | Context for past decisions | Timestamped, read-only |

### 2. Progressive Disclosure

- **Level 0:** `docs/overview.md` - "What is this system?" (5-minute read, architecture diagram)
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

### 6. Additive-First Migration

- **Write new content first** - Create new canonical guides before archiving old docs
- **Parallel operation** - Old and new structures coexist during migration
- **Link updates last** - Only update cross-references after new content is stable
- **Archive in one commit** - Move all old files at once to avoid partial states

---

## Proposed Structure (Detailed)

### `docs/README.md` (Navigation Hub)

```markdown
# Documentation

## 🎯 Overview

**New to the pipeline?** Start with [System Overview](overview.md) to understand the architecture and components.

## 🚀 Getting Started

After reading the overview:

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

### `docs/overview.md` (NEW - Cross-Cutting System Document)

**Target Audience:** Everyone (first-time readers, potential users, researchers evaluating the tool)

**Purpose:** Answer "What is this system?" before diving into how-to guides

**Content:**

1. **Problem Statement** (2 paragraphs)
   - What is antibody non-specificity (polyreactivity)?
   - Why is prediction important for drug development?

2. **Solution Architecture** (visual diagram + 3 paragraphs)
   - High-level pipeline: Sequence → ESM-1v → Embeddings → LogisticRegression → Prediction
   - Key components (see CLAUDE.md architecture section)
   - How components interact

3. **Key Capabilities** (bulleted list)
   - Train models on Boughter dataset (914 VH sequences)
   - Test on multiple benchmarks (Jain, Harvey, Shehata)
   - Fragment-level predictions (VH, CDRs, FWRs)
   - Assay-specific thresholds (ELISA vs PSR)
   - Docker deployment + CI/CD

4. **Technology Stack** (table)
   - ESM-1v (HuggingFace transformers)
   - scikit-learn (LogisticRegression)
   - Python 3.12 + uv package manager
   - pytest + mypy + ruff (quality tooling)

5. **Quick Navigation**
   - → Users: See [Installation Guide](user-guide/installation.md)
   - → Developers: See [Architecture Deep Dive](developer-guide/architecture.md)
   - → Researchers: See [Methodology](research/methodology.md)

**Content Sources:**
- Root `README.md` (Project Description + Model Architecture sections)
- `CLAUDE.md` (Project Overview + Core Pipeline Flow)

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

8. **`preprocessing-internals.md`** (NEW - MOVE from docs/development/)
   - Excel to CSV conversion methods (pandas, openpyxl, CLI tools)
   - Validation techniques (SHA-256 checksums, multi-method cross-checking)
   - Threshold derivation (Jain Table 1, Shehata PSR scores)
   - General preprocessing patterns across datasets
   - **Source:**
     - `docs/development/excel_to_csv_conversion_methods.md` (covers Shehata + Jain, not Boughter-specific)

**Files to Archive:**

- `docs/development/IMPORT_AND_STRUCTURE_GUIDE.md` → `archive/migrations/v2-structure-migration.md`
- `docs/development/P0_P1_P2_P3_BLOCKERS.md` → `archive/investigations/p0-blockers.md` (if completed)
- `docs/development/excel_to_csv_conversion_methods.md` → Move to `developer-guide/preprocessing-internals.md` (not archive)

**DECISION: excel_to_csv_conversion_methods.md Ownership**

✅ **Resolved:** `developer-guide/preprocessing-internals.md`

**Reasoning:**

- File covers **Shehata + Jain preprocessing** (not Boughter-specific)
- Describes **general techniques** (pandas, openpyxl, CLI tools, validation)
- Useful for developers implementing new dataset preprocessing pipelines
- If it were Boughter-only, it would go in `datasets/boughter/README.md`

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

### Phase 0: Fix Accuracy Issues in Current Docs (CRITICAL - DO FIRST)

**Priority:** P0 - Must complete before reorganization to establish trust

**Estimated Time:** 2-3 hours

**Rationale:** Writing a reorganization plan is useless if current docs are outdated. Fix inaccuracies first.

**Tasks:**

- [ ] **Fix `docs/README.md`** (lines 22-60 are severely outdated)
  - Remove references to moved files (`excel_to_csv_conversion_methods.md`, `FIXES_APPLIED.md`, etc.)
  - Update to reflect current directory structure (`development/`, `research/`, `archive/`, `datasets/`)
  - Remove references to non-existent subdirectories (`failed_attempts/`, `p5_close_attempt/`)
  - Update "Last Updated" date and branch name

- [ ] **Fix `docs/development/CICD_SPEC.md`** (lines 32-53 contradict reality)
  - **Current:** Says "No Python environment CI (only Docker)"
  - **Reality:** We have `.github/workflows/ci.yml` with quality gates (ruff, mypy, bandit), unit tests, integration tests, coverage tracking (90.79%)
  - **Fix:** Rewrite "Current State" section to reflect 5 workflows (ci.yml, docker-ci.yml, codeql.yml, dependencies.yml, benchmark.yml)

- [ ] **Fix `docs/development/TYPE_CHECKING_STRATEGY.md`** (lines 5-65 are contradictory)
  - **Current:** Says "Total Errors: 75" AND "ALL FIXED ✅" (contradictory)
  - **Reality:** Type errors are fixed (mypy passes in CI)
  - **Fix:** Remove error lists (lines 13-65), keep only "Status: COMPLETE" section with summary

- [ ] **Fix `docs/development/TEST_SUITE_PLAN.md`** (lines 97-100 are wildly outdated)
  - **Current:** Says "Current Coverage: ~5% (integration tests only)"
  - **Reality:** Coverage is 90.79% (enforced in CI at line 116)
  - **Fix:** Update "Executive Summary" to reflect 400+ tests, 90.79% coverage, comprehensive unit/integration/e2e suite

**Validation:**

After fixes, verify:

- [ ] All file references in `docs/README.md` point to existing files
- [ ] CICD_SPEC.md "Current State" matches `.github/workflows/*.yml` files
- [ ] TYPE_CHECKING_STRATEGY.md doesn't claim errors exist and are fixed simultaneously
- [ ] TEST_SUITE_PLAN.md coverage numbers match `ci.yml` line 116

---

### Phase 1: Create Archive Structure (1 hour)

**Priority:** P1 - Create before moving any files (discoverability)

**Rationale:** Archive README must exist BEFORE we start moving docs, so historical context is discoverable

- [ ] Create `archive/migrations/`, `archive/investigations/`, `archive/plans/`, `archive/summaries/`
- [ ] Write `archive/README.md` with:
  - Index of all archived docs (table: filename, date, category, summary)
  - Archive policy (what gets archived, naming conventions)
  - Cross-references to related active docs
- [ ] DO NOT move files yet (just create structure)

### Phase 2: Write New Overview Document (1-2 hours)

**Priority:** P1 - Foundational "what is this?" doc

**Rationale:** Newcomers need high-level context before diving into installation/training guides

- [ ] Create `docs/overview.md` (see template in "Proposed Structure" section above)
- [ ] Extract content from:
  - Root `README.md` (Project Description, Model Architecture)
  - `CLAUDE.md` (Project Overview, Core Pipeline Flow)
- [ ] Include architecture diagram (ESM → embeddings → classifier → predictions)
- [ ] Add navigation links to user/dev/research guides
- [ ] Get senior review before proceeding

---

### Phase 3: Create User Guide (Additive - Don't Move Old Files Yet)

**Priority:** P2 - User-facing docs

**Strategy:** Write new guides WITHOUT archiving old docs (parallel operation to avoid broken links)

**Estimated Time:** 3-4 hours

- [ ] Write `user-guide/installation.md` (extract from `README.md`)
- [ ] Write `user-guide/getting-started.md` (extract from `README.md` + `CLAUDE.md`)
- [ ] Write `user-guide/training.md` (extract from `CLAUDE.md`)
- [ ] Write `user-guide/testing.md` (extract from `CLAUDE.md`)
- [ ] Write `user-guide/preprocessing.md` (overview + links to dataset docs)
- [ ] Write `user-guide/troubleshooting.md` (MPS leak from archive, common errors)
- [ ] Link from `docs/README.md` to new guides (update navigation hub)
- [ ] DO NOT delete or move old files yet

### Phase 4: Create Developer Guide (Additive - Consolidate Without Deleting)

**Priority:** P2 - Contributor-facing docs

**Strategy:** Write consolidated guides WITHOUT archiving sources (validate new content first)

**Estimated Time:** 4-5 hours

- [ ] Write `developer-guide/architecture.md` (extract from `CLAUDE.md`)
- [ ] Write `developer-guide/development-workflow.md` (consolidate git workflow + pre-commit hooks)
- [ ] Write `developer-guide/testing-strategy.md` (merge 3 testing docs into single canonical guide)
- [ ] Write `developer-guide/type-checking.md` (merge 2 type checking docs, remove error lists)
- [ ] Write `developer-guide/ci-cd.md` (merge 3 CI/CD docs, ensure accuracy after Phase 0 fix)
- [ ] Write `developer-guide/docker.md` (merge 2 Docker docs)
- [ ] Write `developer-guide/security.md` (merge 2 security docs)
- [ ] Write `developer-guide/preprocessing-internals.md` (move `excel_to_csv_conversion_methods.md`)
- [ ] Link from `docs/README.md` to new guides
- [ ] DO NOT delete old files yet (parallel operation)

### Phase 5: Consolidate Research (Additive)

**Priority:** P2 - Scientific validation docs

**Strategy:** Write consolidated research docs WITHOUT archiving sources

**Estimated Time:** 2-3 hours

- [ ] Write `research/novo-parity.md` (merge 3 Novo parity docs)
- [ ] Write `research/methodology.md` (merge 3 methodology docs)
- [ ] Rename `research/ASSAY_SPECIFIC_THRESHOLDS.md` → `research/assay-thresholds.md` (lowercase)
- [ ] Write `research/benchmark-results.md` (merge 2 benchmark docs)
- [ ] Link from `docs/README.md` to new research guides
- [ ] DO NOT archive old files yet

### Phase 6: Archive Old Documentation (1-2 hours)

**Priority:** P3 - Final cleanup after new content is validated

**Strategy:** Archive in ONE COMMIT to avoid partial states

**Rationale:** Only archive after new guides are live and linked (reduces risk of broken references)

- [ ] Move 14 archive docs to appropriate subdirectories with date prefixes (see Appendix)
- [ ] Move original development docs to `archive/` (sources for consolidated guides)
- [ ] Move original research docs to `archive/` (sources for consolidated guides)
- [ ] Update `archive/README.md` index with all archived files
- [ ] Verify no broken inbound links to archived files
- [ ] Commit all moves in single atomic commit

---

### Phase 7: Update Navigation & Cross-Links (1 hour)

**Priority:** P3 - Final polish

- [ ] Finalize `docs/README.md` with complete navigation hub (see template above)
- [ ] Update root `README.md` to link to new docs structure
- [ ] Update `CLAUDE.md` to reference new docs paths (architecture → `docs/overview.md`, etc.)
- [ ] Add cross-links between related guides (user ↔ dev ↔ research)
- [ ] Verify all internal links work

### Phase 8: Validation (1 hour)

**Priority:** P3 - Quality assurance

- [ ] Run `npx markdown-link-check docs/**/*.md` (validate all internal links)
- [ ] Test navigation from user perspective:
  - Can first-time reader find overview in <10s?
  - Can user find training guide in <30s?
  - Can developer find CI/CD guide in <30s?
- [ ] Verify no broken references to archived docs (search for old paths in active docs)
- [ ] Run linter on markdown files (if available)
- [ ] Get senior review approval

**Total Estimated Effort:** 14-18 hours (increased from 12-16h due to Phase 0 accuracy fixes)

---

## Success Metrics

### Quantitative

- ✅ **Reduce top-level doc count:** 40 docs → 7 items (overview.md + 6 categories)
- ✅ **Consolidation ratio:** 15 development docs → 8 canonical guides (47% reduction)
- ✅ **Archive coverage:** 14/14 archive docs organized by type with dates
- ✅ **Navigation depth:** ≤3 clicks from docs/README.md to any guide
- ✅ **Accuracy baseline:** 4/4 outdated docs fixed in Phase 0 (README, CICD_SPEC, TYPE_CHECKING, TEST_SUITE_PLAN)

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

## Open Questions for Senior Review (RESOLVED)

All open questions from original plan have been resolved:

### 1. ✅ excel_to_csv_conversion_methods.md Ownership

**Decision:** `developer-guide/preprocessing-internals.md`

**Reasoning:** File covers Shehata + Jain (not Boughter-specific), describes general techniques useful for implementing new dataset preprocessing pipelines.

### 2. ✅ Archive Deletion Policy

**Decision:** Delete `DOCS_AUDIT_STATUS.md` (superseded by this plan)

**Reasoning:** Redundant audit report that will be outdated once reorganization completes.

### 3. ✅ Research Consolidation

**Decision:** Keep 4 research docs (novo-parity, methodology, assay-thresholds, benchmark-results)

**Reasoning:** Clear separation of concerns, easier to navigate specific topics.

### 4. ✅ Dataset Archive Handling

**Decision:** Keep `datasets/{name}/archive/` in-place

**Reasoning:** Dataset-specific context should live with dataset docs (SSOT principle).

### 5. ✅ External Links (ESM1V doc)

**Decision:** Keep `ESM1V_ENSEMBLING_INVESTIGATION.md` in root permanently

**Reasoning:** Externally linked, moving would break external references (not worth the risk).

### 6. ✅ NEW - Overview Document Placement

**Decision:** `docs/overview.md` at root level (not in `user-guide/`)

**Reasoning:** Overview serves ALL audiences (users, devs, researchers), should be at top level for maximum visibility.

### 7. ✅ NEW - Additive vs Move-First Strategy

**Decision:** Additive-first (write new guides, then archive old files in one commit)

**Reasoning:** Reduces risk of broken links, allows parallel testing of new structure, classic blue-green deployment pattern.

---

## NEW REQUIREMENT: Phase 0 Accuracy Fixes

**Critical Addition:** Added Phase 0 to fix severely outdated docs BEFORE reorganization:

1. `docs/README.md` (lines 22-60 reference moved/non-existent files)
2. `docs/development/CICD_SPEC.md` (claims no CI when we have 5 workflows)
3. `docs/development/TYPE_CHECKING_STRATEGY.md` (contradictory: lists errors + claims fixed)
4. `docs/development/TEST_SUITE_PLAN.md` (claims 5% coverage when we have 90.79%)

**Rationale:** Establishing trust through accuracy is prerequisite for reorganization. Writing a plan to organize outdated docs is counterproductive.

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
| `excel_to_csv_conversion_methods.md` | `developer-guide/preprocessing-internals.md` | Move (not Boughter-specific, covers Shehata + Jain) |

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

---

## Revision History

| Date | Version | Changes | Approver |
|------|---------|---------|----------|
| 2025-11-09 | v1.0 | Initial plan (5 categories, 12-16h effort) | - |
| 2025-11-09 | v2.0 | **REVISED** - Incorporated senior feedback:<br>• Added Phase 0 (accuracy fixes for 4 outdated docs)<br>• Added `overview.md` (cross-cutting system doc)<br>• Added `developer-guide/preprocessing-internals.md`<br>• Changed strategy to additive-first (write new, then archive)<br>• Resolved all open questions<br>• Increased effort to 14-18h | Awaiting approval |

**Approval Checklist:**

- [ ] Senior review complete
- [x] Open questions resolved (all 7 questions answered)
- [x] Phase 0 accuracy fixes validated (4 outdated docs confirmed)
- [x] Additive-first strategy approved (reduces risk)
- [ ] Implementation roadmap approved
- [ ] Ready to execute
