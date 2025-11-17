# Documentation Inventory - Complete Baseline Audit

**Created**: 2025-11-17
**Phase**: Phase 0 (Baseline Inventory) of DOCUMENTATION_UPDATE_ROADMAP.md
**Purpose**: Complete catalog of all markdown documentation in repository
**Total Files**: 212

---

## 📊 Summary Statistics

| Category | Count | Audit Priority | Notes |
|----------|-------|----------------|-------|
| **Root Documentation** | 10 | 🔴 P0 - Critical | User-facing, must be 100% accurate |
| **Canonical Docs (`docs/`)** | 89 | 🔴 P0 - Critical | Official documentation |
| **Burner Docs (Gold)** | 7 | 🟡 P1 - High | Need accuracy audit before integration |
| **Burner Docs (Trash)** | 13 | ⚪ P3 - Low | Obsolete, moved to trash |
| **Experiments Benchmarks** | 15 | 🟢 P2 - Medium | Versioned results documentation |
| **Dataset Documentation** | 19 | 🟢 P2 - Medium | Data pipeline docs |
| **Preprocessing Docs** | 5 | 🟢 P2 - Medium | Pipeline-specific docs |
| **Scripts Documentation** | 2 | 🟢 P2 - Medium | Utility script docs |
| **Literature (Papers)** | 10 | ⚪ P3 - Low | Reference papers (static) |
| **Reference Repos** | 41 | ⚪ P3 - Low | External repo mirrors (static) |
| **Other (.pytest_cache)** | 1 | ⚪ P3 - Low | Auto-generated |
| **TOTAL** | **212** | | |

---

## 🎯 Audit Priority Targets (Phase 1-2)

### 🔴 **P0 - Critical** (99 files - MUST be 100% accurate)

These docs are user-facing or primary developer references. Any stale paths/commands will break user workflows.

**Root Documentation (10 files):**
- `README.md` - Project quickstart
- `CLAUDE.md` - AI assistant instructions
- `USAGE.md` - Command reference
- `ROADMAP.md` - Project roadmap
- `CHANGELOG.md` - Version history
- `CITATIONS.md` - Dataset attributions
- `CONTRIBUTING.md` - Contribution guidelines
- `CODE_OF_CONDUCT.md` - Community standards
- `DOCUMENTATION_UPDATE_ROADMAP.md` ✅ - This audit plan (just created)
- `PENDING_MODEL_VALIDATION.md` ✅ - Model zoo expansion plan (just created)

**Canonical Documentation (89 files in `docs/`):**
- User guides (`docs/user-guide/`)
- Developer guides (`docs/developer-guide/`)
- Research documentation (`docs/research/`)
- Dataset documentation (`docs/datasets/`)
- Architecture docs (`docs/architecture/`)
- Full file list: See "Canonical Documentation Files" section below

---

### 🟡 **P1 - High** (7 files - Audit BEFORE integration)

Burner docs marked as "gold" - need accuracy audit for stale references before integrating into canonical docs.

**Burner Docs (Gold) - `docs_burner/implementation/`:**
1. `VALIDATION_ROADMAP.md` - Comprehensive validation plan
2. `REPOSITORY_REORGANIZATION_PLAN.md` - Phase 5 reorganization plan
3. `TEST_COVERAGE_PLAN.md` - Testing strategy
4. `V0.5.0_CLEANUP_PLAN.md` - Legacy cleanup plan
5. `REPOSITORY_STRUCTURE_ANALYSIS.md` - Repository evolution
6. `VALIDATION_REALITY.md` - Reality check on validation commands
7. `OUTPUT_DIRECTORY_INVESTIGATION.md` - Hierarchical routing documentation

**Known Staleness Risks:**
- May reference `configs/config.yaml` ❌ (DELETED in Phase 5)
- May reference root-level `models/` ❌ (moved to `experiments/checkpoints/`)
- May reference root-level `embeddings_cache/` ❌ (moved to `experiments/cache/`)
- May reference root-level `logs/` ❌ (moved to `experiments/runs/logs/`)
- May reference root-level `outputs/` ❌ (moved to `experiments/runs/`)

---

### 🟢 **P2 - Medium** (41 files - Standard audit)

Supporting documentation that likely needs path updates but won't break critical workflows.

**Experiments Documentation (15 files):**
- Benchmark results READMEs in `experiments/benchmarks/`
- Archive documentation

**Dataset Documentation (19 files):**
- Dataset READMEs in `data/train/` and `data/test/`
- Pipeline-specific documentation

**Preprocessing Documentation (5 files):**
- Stage-specific preprocessing docs in `preprocessing/{dataset}/`

**Scripts Documentation (2 files):**
- Utility script documentation in `scripts/`

---

### ⚪ **P3 - Low** (65 files - No audit needed)

Static reference materials, obsolete docs, or auto-generated files.

**Burner Docs (Trash) - `docs_burner/trash/` (13 files):**
- Obsolete migration plans (already executed)
- One-off validation summaries (already completed)
- Known issues (already fixed)
- These are preserved for historical reference but excluded from audits

**Literature (10 files):**
- Research paper markdown conversions in `literature/markdown/`
- Static reference material (no updates needed)

**Reference Repos (41 files):**
- External repository mirrors in `reference_repos/`
- Static snapshots (no updates needed)

**Auto-generated (1 file):**
- `.pytest_cache/README.md` - Pytest auto-generated

---

## 📋 Complete File Lists

### Root Documentation (10 files)

```
./CHANGELOG.md
./CITATIONS.md
./CLAUDE.md
./CODE_OF_CONDUCT.md
./CONTRIBUTING.md
./DOCUMENTATION_UPDATE_ROADMAP.md ✅ NEW
./PENDING_MODEL_VALIDATION.md ✅ NEW
./README.md
./ROADMAP.md
./USAGE.md
```

---

### Canonical Documentation (89 files)

**Full listing:**
```bash
# To regenerate: find docs -name "*.md" | sort
```

**Key subsections:**
- `docs/user-guide/` - User-facing guides (installation, training, testing, preprocessing)
- `docs/developer-guide/` - Developer workflow, architecture, testing strategy
- `docs/research/` - Methodology, Novo parity analysis, benchmark results
- `docs/datasets/` - Boughter, Jain, Harvey, Shehata dataset documentation
- `docs/architecture/` - System architecture and design docs

**High-priority files to audit (Phase 2):**
- `docs/developer-guide/directory-organization.md` ⚠️ - Likely references old structure
- `docs/user-guide/training.md` ⚠️ - May reference old paths
- `docs/user-guide/testing.md` ⚠️ - May reference old paths
- `docs/developer-guide/development-workflow.md` ⚠️ - May reference old commands
- `docs/architecture/*.md` ⚠️ - May need Phase 5 structure updates

---

### Burner Docs - Gold (7 files)

**Location: `docs_burner/implementation/`**

```
./docs_burner/implementation/OUTPUT_DIRECTORY_INVESTIGATION.md
./docs_burner/implementation/REPOSITORY_REORGANIZATION_PLAN.md
./docs_burner/implementation/REPOSITORY_STRUCTURE_ANALYSIS.md
./docs_burner/implementation/TEST_COVERAGE_PLAN.md
./docs_burner/implementation/V0.5.0_CLEANUP_PLAN.md
./docs_burner/implementation/VALIDATION_REALITY.md
./docs_burner/implementation/VALIDATION_ROADMAP.md
```

**Integration Plan (Phase 5 of roadmap):**
- `VALIDATION_ROADMAP.md` → `docs/developer-guide/validation-checklist.md`
- `REPOSITORY_REORGANIZATION_PLAN.md` → `docs/architecture/experiments-structure.md`
- `TEST_COVERAGE_PLAN.md` → `docs/developer-guide/testing-strategy.md` (append)
- `V0.5.0_CLEANUP_PLAN.md` → `docs/archive/migrations/v0.5.0-legacy-removal.md`
- `REPOSITORY_STRUCTURE_ANALYSIS.md` → `docs/architecture/repository-evolution.md`
- `VALIDATION_REALITY.md` → Merge into `validation-checklist.md`
- `OUTPUT_DIRECTORY_INVESTIGATION.md` → Extract sections into preprocessing docs

---

### Burner Docs - Trash (13 files)

**Location: `docs_burner/trash/`**

```
./docs_burner/trash/CLEANUP_BEFORE_VALIDATION_DECISION.md
./docs_burner/trash/DOCKER_CI_FAILURE_ANALYSIS.md
./docs_burner/trash/GITHUB_ACTIONS_DISK_SPACE.md
./docs_burner/trash/HYPERPARAMETER_SWEEP_ARCHIVE_PLAN.md
./docs_burner/trash/KNOWN_ISSUES.md ⚠️ REVERTED (configs/config.yaml issue already fixed)
./docs_burner/trash/OUTPUT_ORGANIZATION_FINAL_CLEANUP_PLAN.md
./docs_burner/trash/POST_MIGRATION_VALIDATION_FINDINGS.md
./docs_burner/trash/POST_MIGRATION_VALIDATION_PLAN.md
./docs_burner/trash/POST_MIGRATION_VALIDATION_SUMMARY.md
./docs_burner/trash/REPOSITORY_CLEANUP_PLAN.md
./docs_burner/trash/TEST_DATASETS_CONSOLIDATION_PLAN.md
./docs_burner/trash/TEST_OUTPUT_DIRECTORY_ISSUE.md
./docs_burner/trash/TRAIN_DATASETS_CONSOLIDATION_PLAN.md
```

**Status**: Obsolete (already executed or issues already fixed). Preserved for historical reference.

---

### Experiments Documentation (15 files)

**Location: `experiments/benchmarks/`**

Versioned benchmark result documentation. These docs are tied to specific benchmark runs and should remain accurate to the snapshot they document.

**Sample:**
```
./experiments/README.md
./experiments/benchmarks/README.md
./experiments/benchmarks/archive/hyperparameter_sweeps_2025-11-02/README.md
./experiments/benchmarks/archive/test_results_pre_migration_2025-11-06/README.md
./experiments/benchmarks/esm1v/logreg/VH_only_jain_86_p5e_s2/README.md
./experiments/benchmarks/esm1v/logreg/VH_only_shehata/README.md
./experiments/benchmarks/esm1v/logreg/VHH_only_harvey/README.md
./experiments/benchmarks/novo_parity/README.md
... (7 more)
```

---

### Dataset Documentation (19 files)

**Location: `data/train/` and `data/test/`**

Dataset-specific READMEs documenting data sources, preprocessing stages, and file organization.

**Sample:**
```
./data/train/README.md
./data/train/BOUGHTER_DATA_PROVENANCE.md
./data/train/boughter/README.md
./data/train/boughter/canonical/README.md
./data/test/jain/README.md
./data/test/harvey/README.md
./data/test/shehata/README.md
... (12 more stage-specific READMEs)
```

---

### Preprocessing Documentation (5 files)

**Location: `preprocessing/{dataset}/`**

Pipeline-specific documentation for each dataset's preprocessing stages.

```
./preprocessing/README.md
./preprocessing/boughter/README.md
./preprocessing/harvey/README.md
./preprocessing/jain/README.md
./preprocessing/shehata/README.md
```

---

### Scripts Documentation (2 files)

**Location: `scripts/`**

```
./scripts/README.md
./scripts/prepare_combined_datasets.md
```

---

### Literature (10 files - Static)

**Location: `literature/markdown/`**

Markdown conversions of research papers. No updates needed (static reference material).

```
./literature/markdown/boltzmann_2024_main/jiao-et-al-2024-boltzmann-aligned-inverse-folding-model.md
./literature/markdown/boughter_2020_main/bougher.md
./literature/markdown/esm_model/esm_model/esm_model.md
./literature/markdown/harvey_2022_main/harvey-et-al-2022-in-silico-method-to-assess-antibody-fragment-polyreactivity.md
./literature/markdown/harvey_2022_supplementary/harvey-et-al-2022-supplementary-information.md
./literature/markdown/jain_2017_main/jain-et-al-2017-biophysical-properties-of-the-clinical-stage-antibody-landscape.md
./literature/markdown/jain_2017_supplementary/pnas.201616408si.md
./literature/markdown/novo_2025_main/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.md
./literature/markdown/novo_2025_supplementary/novo-media-1/novo-media-1.md
./literature/markdown/shehata_2019_main/shehata_2019_main/shehata_2019_main.md
```

---

### Reference Repos (41 files - Static)

**Location: `reference_repos/`**

External repository mirrors (Harvey official repo, Ludocomito original). No updates needed (static snapshots).

**Sample:**
```
./reference_repos/harvey_official_repo/LICENSE.md
./reference_repos/ludocomito_original/USAGE.md
... (39 more files)
```

---

### Auto-generated (1 file)

```
./.pytest_cache/README.md
```

---

## 🚦 Next Steps (Phase 1 - Burner Docs Audit)

**Immediate Next Action**: Execute Phase 1 of DOCUMENTATION_UPDATE_ROADMAP.md

**Phase 1 Tasks** (2-3 hours):
1. Audit 7 gold burner docs for stale references
2. Check each file for:
   - ❌ References to `configs/config.yaml` (DELETED)
   - ❌ References to root-level `models/` (moved to `experiments/checkpoints/`)
   - ❌ References to root-level `embeddings_cache/` (moved to `experiments/cache/`)
   - ❌ References to root-level `logs/` (moved to `experiments/runs/logs/`)
   - ❌ References to root-level `outputs/` (moved to `experiments/runs/`)
3. Verify all file paths exist in current codebase
4. Verify all commands still work
5. Create `BURNER_DOCS_AUDIT_REPORT.md` with findings

**Commands to run**:
```bash
# For each burner doc file:
grep -n "configs/config.yaml\|^models/\|^embeddings_cache/\|^logs/\|^outputs/" docs_burner/implementation/<file>
```

---

## 📝 Deliverable Status

✅ **COMPLETE** - Phase 0: Baseline Inventory (this document)
⏸️ **PENDING** - Phase 1: Burner Docs Audit Report
⏸️ **PENDING** - Phase 2: Canonical Docs Audit Report
⏸️ **PENDING** - Phase 3: Documentation Update Plan
⏸️ **PENDING** - Phase 4: Execute Updates
⏸️ **PENDING** - Phase 5: Integrate Burner Docs
⏸️ **PENDING** - Phase 6: Final Verification Report

---

**Inventory Created**: 2025-11-17
**Created By**: Claude Code (systematic baseline audit)
**Next Action**: Execute Phase 1 (Burner Docs Audit)
**Reference**: DOCUMENTATION_UPDATE_ROADMAP.md
