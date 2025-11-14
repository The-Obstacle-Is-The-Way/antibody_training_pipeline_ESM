# Directory Structure Cleanup & Modernization Proposal

**Branch**: `chore/directory-structure-cleanup`
**Status**: PLANNING - Awaiting Senior Approval
**Goal**: Transform inherited codebase into a professionally-organized, industry-standard repository

---

## Executive Summary

We inherited a forked codebase from master students (now 800+ commits ahead). Time to clean up the slop and establish a canonical, maintainable directory structure that would impress engineers at Google DeepMind or any world-class research lab.

**Key Issues Identified:**
1. `test_datasets/` and `train_datasets/` should be unified under `data/`
2. Fragmented experiment artifacts scattered across root
3. Inconsistent naming conventions (snake_case vs kebab-case)
4. Unclear separation between research artifacts and production code
5. Cache directories polluting root (`.mypy_cache`, `.uv_cache`)

---

## Current Structure Analysis

### Root-Level Directories (27 total)

#### Production Code & Config ✅
- `src/` - Main package (clean)
- `tests/` - Test suite (clean)
- `configs/` - Hydra configs (REDUNDANT with `src/antibody_training_esm/conf/`)
- `.github/` - CI/CD workflows (clean)

#### Data Directories ⚠️ NEEDS CONSOLIDATION
- `train_datasets/` - Training data (Boughter only)
- `test_datasets/` - Test data (Harvey, Jain, Shehata)
- `external_datasets/` - External data (currently empty, git-ignored)

**PROPOSAL**: Consolidate into `data/` with subdirectories:
```
data/
├── train/
│   └── boughter/
├── test/
│   ├── harvey/
│   ├── jain/
│   └── shehata/
└── external/  (git-ignored)
```

#### Experiment Artifacts ⚠️ NEEDS CLEANUP
- `experiments/` - Versioned experiments (novo_parity, strict_qc) ✅ KEEP
- `outputs/` - Training outputs (git-ignored) ✅ KEEP
- `models/` - Trained models (hierarchical: esm1v/logreg/, esm2_650m/logreg/) ✅ KEEP
- `test_results/` - Testing outputs (hierarchical) ✅ KEEP
- `hyperparameter_sweep_results/` - Legacy sweep CSVs ❌ MOVE to `experiments/archive/`
- `logs/` - Training logs (git-ignored) ✅ KEEP

#### Documentation 📚 MOSTLY CLEAN
- `docs/` - Comprehensive documentation (well-organized) ✅ KEEP
- `docs_burner/` - Temporary planning docs (git-ignored) ✅ KEEP
- `literature/` - Research PDFs + markdown (git-ignored) ✅ KEEP

#### Reference Code 📖
- `reference_repos/` - Upstream repos (git-ignored) ✅ KEEP
- `preprocessing/` - Dataset-specific preprocessing ✅ KEEP
- `scripts/` - Utility scripts (validation, testing, migration) ✅ KEEP

#### Build Artifacts & Caches ⚠️ POLLUTION
- `dist/` - Build artifacts (git-ignored) ✅ KEEP
- `.mypy_cache/` - Type checking cache ❌ SHOULD BE GIT-IGNORED
- `.uv_cache/` - UV package cache ❌ SHOULD BE GIT-IGNORED (already is)
- `.benchmarks/` - Pytest benchmark cache ✅ KEEP (git-ignored)
- `embeddings_cache/` - ESM embeddings (git-ignored) ✅ KEEP

#### Assets 🎨
- `assets/` - Images (ego.jpg, leeroy_jenkins.png) ✅ KEEP

---

## Proposed Changes

### Priority 1: Data Directory Consolidation

**CURRENT (Confusing):**
```
train_datasets/
├── boughter/
test_datasets/
├── harvey/
├── jain/
└── shehata/
external_datasets/  (empty)
```

**PROPOSED (Clear):**
```
data/
├── train/
│   └── boughter/
│       ├── raw/
│       ├── processed/
│       ├── annotated/
│       └── canonical/
├── test/
│   ├── harvey/
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── canonical/
│   │   └── fragments/
│   ├── jain/
│   │   └── [same structure]
│   └── shehata/
│       └── [same structure]
└── external/  (git-ignored for large downloads)
```

**Rationale:**
- Industry standard: `data/` is universal (see: TensorFlow, PyTorch, Hugging Face)
- Semantic clarity: `train/` vs `test/` instead of `train_datasets/` vs `test_datasets/`
- Single source of truth for all data assets
- Easier to explain to collaborators
- Matches Google's style guide for ML repos

**Files to Update:**
- `src/antibody_training_esm/datasets/default_paths.py` - All path constants
- `preprocessing/**/README.md` - Update documentation
- `CLAUDE.md` - Update directory structure section
- `docs/developer-guide/directory-organization.md` - Update reference
- All test fixtures referencing data paths
- `.gitignore` - Update ignore patterns

### Priority 2: Consolidate Hydra Configs

**CURRENT (Redundant):**
```
configs/
└── config.yaml  (ROOT LEVEL - confusing)
src/antibody_training_esm/conf/
└── config.yaml  (PACKAGED - correct)
```

**PROPOSED:**
```
(Remove root configs/)
src/antibody_training_esm/conf/
└── config.yaml  (Single source of truth)
```

**Rationale:**
- Hydra configs should be packaged with the application
- Root-level `configs/` is a legacy artifact
- Single source of truth principle

**Files to Update:**
- Delete `configs/config.yaml`
- Verify no code references `configs/`
- Update documentation

### Priority 3: Archive Legacy Experiments

**CURRENT:**
```
hyperparameter_sweep_results/  (22 CSV files from Nov 2, 2025)
preprocessing/boughter/train_hyperparameter_sweep.py (hardcoded output path)
```

**PROPOSED:**
```
experiments/archive/
└── hyperparameter_sweeps_2025-11-02/
    └── [22 CSV files moved from root]

preprocessing/boughter/train_hyperparameter_sweep.py
  - Update default: output_dir = "experiments/archive/hyperparameter_sweeps"
```

**Rationale:**
- These are historical artifacts from completed Nov 2, 2025 sweep
- `experiments/` directory already exists for versioned work
- Keeps root clean
- **CRITICAL**: Must update script BEFORE moving files to prevent future sweep failures

### Priority 4: Cache Directory Hygiene (VERIFIED CORRECT)

**VERIFIED git-ignore status:**
- `.mypy_cache/` - ✅ **ALREADY git-ignored** (verified via `git ls-files`)
- `.uv_cache/` - ✅ Already git-ignored
- `.benchmarks/` - ✅ Already git-ignored

**FINDING**: All cache directories are correctly git-ignored. No action needed.

**ACTION:**
- ✅ **NO CHANGES REQUIRED** - Repository is already compliant with best practices

---

## Impact Analysis

### Breaking Changes 🔴
1. **Data path changes** - All code referencing `train_datasets/` or `test_datasets/` must be updated
2. **Sweep output path** - Future hyperparameter sweeps write to new location (acceptable)

### Non-Breaking Changes 🟢
1. Moving `hyperparameter_sweep_results/` to `experiments/archive/` (historical data only)
2. Cache directories already correctly configured

### Deferred Changes (v0.5.0+) ⏸️
1. Removing `configs/` directory (requires deprecation period)
2. Migrating all scripts to Hydra
3. Removing legacy `train_model()` function

### Migration Complexity
- **Data directory move**: MEDIUM - Requires systematic path updates across codebase
- **Sweep archive**: LOW - Update 1 default parameter, move files
- **Config deprecation**: DEFERRED - Needs 1-2 release deprecation period

---

## Implementation Plan

### Phase 1: Preparation (No Code Changes)
1. ✅ Create branch `chore/directory-structure-cleanup`
2. ✅ Generate current structure documentation
3. ⏸️ Write this proposal document
4. ⏸️ Get senior approval
5. ⏸️ Audit all path references in codebase

### Phase 2: Data Directory Migration
1. Create new `data/` directory structure
2. Update `default_paths.py` constants
3. Update all preprocessing scripts
4. Update test fixtures
5. Move files from `train_datasets/` → `data/train/`
6. Move files from `test_datasets/` → `data/test/`
7. Run full test suite (expect failures, fix iteratively)
8. Update documentation

### Phase 3: Archive Cleanup (CORRECTED)
1. **Update script first**: `preprocessing/boughter/train_hyperparameter_sweep.py` default path
2. Create `experiments/archive/hyperparameter_sweeps_2025-11-02/`
3. Move 22 CSV files with `git mv` from `hyperparameter_sweep_results/`
4. Delete empty `hyperparameter_sweep_results/`
5. Add README.md in archive explaining provenance
6. Test script still works with new default

### Phase 4: Config Deprecation (DEFERRED TO v0.5.0)
1. Migrate `train_hyperparameter_sweep.py` to use Hydra
2. Add deprecation warning to `configs/config.yaml`
3. Wait 1-2 release cycles
4. Delete `configs/` directory
5. Remove legacy `train_model()` function
6. Update all documentation

### Phase 5: Verification
1. Run full test suite (`make test`)
2. Run type checking (`make typecheck`)
3. Run linting (`make lint`)
4. Verify Docker builds work
5. Test training pipeline end-to-end
6. Update CLAUDE.md with new structure

---

## Alternatives Considered

### Alternative 1: Leave train_datasets/ and test_datasets/ separate
**Rejected**: Not industry standard, confusing for new contributors

### Alternative 2: Use datasets/ instead of data/
**Rejected**: `data/` is more universal (used by PyTorch, TensorFlow, sklearn)

### Alternative 3: Keep configs/ at root
**Rejected**: Hydra best practice is to package configs with application

---

## Risk Mitigation

1. **All work in feature branch** - No impact on main/dev until approved
2. **Iterative testing** - Run test suite after each phase
3. **Git history preserved** - Use `git mv` to maintain file history
4. **Rollback plan** - Revert commits if issues arise
5. **Comprehensive documentation** - Update all docs before merge

---

## Success Criteria

✅ All tests pass
✅ No hardcoded paths remain
✅ Directory structure matches industry standards
✅ Documentation updated
✅ Docker builds work
✅ Training pipeline works end-to-end
✅ Code review approval from senior engineer (you)

---

## Questions for Senior Approval

1. **Data directory naming**: Approve `data/train/` and `data/test/` structure?
2. **Timeline**: Should we do this incrementally or all at once?
3. **Breaking changes**: Acceptable for internal research repo?
4. **Additional cleanup**: Any other directories that need reorganization?

---

## Current Directory Tree

See `CURRENT_STRUCTURE.txt` for full tree output (238 directories, 577 files)

**Key Stats:**
- Root-level directories: 27
- Python source files: ~150
- Test files: ~100
- Documentation files: ~80
- Git-ignored cache/output directories: ~10

---

**Next Steps:**
1. Review this proposal
2. Approve/modify Phase 1-5 plan
3. Begin implementation systematically
4. Test at each step
5. Merge when everything works

**Estimated Time (CORRECTED):**
- Phase 1 (Audit): 1 hour ✅
- Phase 2 (Data migration): 8-10 hours (deferred to next release)
- Phase 3 (Archive cleanup): 30 mins (THIS release)
- Phase 4 (Config deprecation): 2-3 hours (deferred to v0.5.0)
- Phase 5 (Verification): 1-2 hours

**Recommended THIS Release**: Phase 1 + Phase 3 only (~1.5 hours)
**Defer to NEXT Release**: Phase 2 (data consolidation)
**Defer to v0.5.0+**: Phase 4 (config removal)
