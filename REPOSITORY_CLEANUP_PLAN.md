# Repository Cleanup Plan - Single Source of Truth

**Branch**: `chore/directory-structure-cleanup`
**Date**: 2025-11-14
**Status**: ✅ EXECUTION COMPLETE (Phases 1 & 2 finished 2025-11-15)

---

## Goals

1. **Data Consolidation**: `train_datasets/` + `test_datasets/` → `data/train/` + `data/test/` (Phase 1 ✅ complete)
2. **Config Cleanup**: Remove redundant root `configs/` directory (defer to v0.5.0)
3. **Archive Legacy**: Move `hyperparameter_sweep_results/` → `experiments/archive/`

---

## Current State Snapshot

**Reference**: See `CURRENT_STRUCTURE.txt` for full tree (238 dirs, 577 files)

**Key findings**:
- ✅ All datasets consolidated under `data/{train,test}/`
- ✅ Zero `train_datasets/` references remain in production code (docs retain historical context only)
- 2 active dependencies on `configs/config.yaml` (to be removed in v0.5.0)
- 1 reference to `hyperparameter_sweep_results/` (pending archive)
- All cache directories (`.mypy_cache/`, `.uv_cache/`, `.benchmarks/`) already correctly git-ignored

---

## Problem 1: Data Directory Consolidation

> **Note:** The following subsections capture the original execution plan. Both phases completed successfully on 2025-11-15 and are preserved here for historical reference.

### Pre-Migration State (Before Phase 1)
```
train_datasets/
├── boughter/
    ├── raw/
    ├── processed/
    ├── annotated/
    └── canonical/

test_datasets/           # ✅ Migrated to data/test/ (Phase 1 complete)
├── harvey/
│   ├── raw/
│   ├── processed/
│   ├── canonical/
│   └── fragments/
├── jain/
│   ├── raw/
│   ├── processed/
│   └── canonical/
└── shehata/
    ├── raw/
    ├── processed/
    └── canonical/
```

### Current State (After Phases 1 & 2)
```
data/train/              # ✅ Phase 2 complete (commit 0d4605d)
└── boughter/
    ├── raw/
    ├── processed/
    ├── annotated/
    └── canonical/

data/test/               # ✅ Phase 1 complete (commit 288905c)
├── harvey/
│   ├── raw/
│   ├── processed/
│   ├── canonical/
│   └── fragments/
├── jain/
│   ├── raw/
│   ├── processed/
│   └── canonical/
└── shehata/
    ├── raw/
    ├── processed/
    └── canonical/
```

### Historical Target (achieved 2025-11-15)
*(Retained for provenance; see “Current State” above for actual layout.)*
```
data/
├── train/
│   └── boughter/
│       ├── raw/
│       ├── processed/
│       ├── annotated/
│       └── canonical/
└── test/
    ├── harvey/
    │   ├── raw/
    │   ├── processed/
    │   ├── canonical/
    │   └── fragments/
    ├── jain/
    │   ├── raw/
    │   ├── processed/
    │   └── canonical/
    └── shehata/
        ├── raw/
        ├── processed/
        └── canonical/
```

### Complexity Assessment

**125 hardcoded path references** across:

#### Category A: Centralized Path Constants (EASY - 1 file)
- `src/antibody_training_esm/datasets/default_paths.py` - Single source of truth for data paths

#### Category B: Hydra Configs (EASY - 2 files)
- `src/antibody_training_esm/conf/config.yaml` - Modern Hydra config
- `configs/config.yaml` - Legacy config (will be removed in v0.5.0)

#### Category C: Test Files (MEDIUM - ~20 files)
- `tests/unit/datasets/test_boughter.py:52`
- `tests/unit/datasets/test_harvey.py:60`
- `tests/unit/datasets/test_jain.py`
- `tests/unit/datasets/test_shehata.py`
- `tests/integration/test_boughter_embedding_compatibility.py:47`
- `tests/e2e/test_reproduce_novo.py:63`
- All test fixtures in `tests/fixtures/`

#### Category D: Preprocessing Scripts (MEDIUM - ~15 files)
- `preprocessing/boughter/validate_stage1.py:16-150`
- `preprocessing/boughter/validate_stages2_3.py`
- `preprocessing/harvey/step2_extract_fragments.py`
- `preprocessing/jain/step2_preprocess_p5e_s2.py`
- `preprocessing/shehata/step1_convert_excel_to_csv.py:285`
- All `preprocessing/*/README.md` files

#### Category E: CLI & Source (HARD - scattered)
- `src/antibody_training_esm/cli/test.py:218`
- Documentation in `docs/`
- Docker files (if any)
- CI/CD workflows in `.github/workflows/`

### Execution Plan for Data Consolidation

#### Phase 1: Full Path Audit (3 hours)
```bash
# Generate comprehensive reference list
grep -rn "train_datasets\|test_datasets" \
  --include="*.py" \
  --include="*.md" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="*.sh" \
  --include="Dockerfile*" \
  src/ tests/ preprocessing/ scripts/ docs/ .github/ \
  > /tmp/data_path_references.txt

# Categorize by risk/complexity
# - default_paths.py: 1 line change
# - configs: 2 files
# - tests: ~20 files
# - preprocessing: ~15 files
# - CLI/docs: scattered
```

#### Phase 2: Create Migration Script (3 hours)

**Phase 1 Used:** `scripts/migrate_test_datasets_to_data_test.sh` (bash, grep + sed)
- ✅ Successfully migrated 135 files (commit 288905c)
- ✅ Preserved as documentation-only (exits immediately, no re-run risk)
- ✅ Excludes plan documents and self-references

**Phase 2 Recommendation:** Copy and adapt the shell script pattern:
```bash
# For Phase 2, copy the Phase 1 script:
cp scripts/migrate_test_datasets_to_data_test.sh scripts/migrate_train_datasets_to_data_train.sh

# Then modify patterns:
# - Change test_datasets/ → train_datasets/
# - Change data/test/ → data/train/
# - Update exclusions for Phase 2 plan docs
# - Add code-only file filters (*.py, *.yaml, not *.md except docs/)
```

Alternative (Python, if preferred):
```python
# scripts/migrate_data_directories.py
# Similar to scripts/migrate_model_directories.py which worked successfully

import re
from pathlib import Path

def update_file(file_path: Path, old_pattern: str, new_pattern: str) -> None:
    """Update all occurrences of data paths in a file."""
    content = file_path.read_text()
    updated = re.sub(old_pattern, new_pattern, content)
    if content != updated:
        file_path.write_text(updated)
        print(f"Updated: {file_path}")

# Pattern: train_datasets/boughter → data/train/boughter
```

#### Phase 3: Execute Migration (4 hours)

**Step 1: Update centralized constants**
```python
# src/antibody_training_esm/datasets/default_paths.py
# FROM:
BOUGHTER_ANNOTATED_DIR = Path("train_datasets/boughter/annotated")
# TO:
BOUGHTER_ANNOTATED_DIR = Path("data/train/boughter/annotated")
```

**Step 2: Update configs**
```yaml
# src/antibody_training_esm/conf/config.yaml
# configs/config.yaml (legacy)
data:
  train_file: data/train/boughter/canonical/boughter_canonical.csv
  test_file: data/test/jain/canonical/jain_canonical.csv
```

**Step 3: Run migration script**

Phase 1 (✅ complete):
```bash
# Used: scripts/migrate_test_datasets_to_data_test.sh
# Migrated 135 files via grep + sed
# Now preserved as documentation-only (exits immediately)
```

Phase 2 (historical runbook):
```bash
# Option A: Bash (recommended - proven pattern from Phase 1)
cp scripts/migrate_test_datasets_to_data_test.sh scripts/migrate_train_datasets_to_data_train.sh
# Modify patterns: test_datasets → train_datasets, data/test → data/train
chmod +x scripts/migrate_train_datasets_to_data_train.sh
./scripts/migrate_train_datasets_to_data_train.sh

# Option B: Python (if preferred)
python scripts/migrate_data_directories.py --dry-run  # Preview changes
python scripts/migrate_data_directories.py            # Execute (must create first)
```

**Step 4: Move directories with git mv**

Phase 1 (✅ complete - commit 288905c):
```bash
mkdir -p data/test
git mv test_datasets/harvey data/test/
git mv test_datasets/jain data/test/
git mv test_datasets/shehata data/test/
rmdir test_datasets/
```

Phase 2 (historical runbook):
```bash
mkdir -p data/train
git mv train_datasets/boughter data/train/
rmdir train_datasets/
```

**Step 5: Update .gitignore**
```bash
# .gitignore
# Old patterns:
-train_datasets/boughter/raw/
-data/test/*/raw/

# New patterns:
+data/train/*/raw/
+data/test/*/raw/
```

#### Phase 4: Verification (3 hours)

**Run all tests**
```bash
make test              # Full test suite must pass
make typecheck         # Type safety must pass
make lint              # Linting must pass
```

**Manual verification**
```bash
# Test training pipeline
uv run antibody-train experiment.name=data_migration_test

# Test preprocessing scripts
python preprocessing/boughter/validate_stage1.py
python preprocessing/harvey/step2_extract_fragments.py

# Test CLI
uv run antibody-test --model models/esm1v/logreg/model.pkl --dataset jain
```

**Docker/CI verification**
```bash
# Check Docker builds (if applicable)
docker build -t test .

# Check CI workflows
# Review .github/workflows/*.yml for hardcoded paths
```

### Estimated Time: 13-15 hours

---

## Problem 2: Config Directory Removal

### Current State

**Active dependencies**:
1. `src/antibody_training_esm/core/trainer.py:793` - `train_model(config_path: str = "configs/config.yaml")`
2. `preprocessing/boughter/train_hyperparameter_sweep.py:279` - `main(config_path: str | Path = "configs/config.yaml")`

**Files**:
```
configs/
└── config.yaml  (82 lines, legacy pre-Hydra config)

src/antibody_training_esm/conf/
└── config.yaml  (Modern Hydra config)
```

**Key difference**: These are DIFFERENT files
- Legacy: Flat YAML with hardcoded paths
- Modern: Hydra composition with defaults

### Target State

```
(Remove root configs/ directory)

src/antibody_training_esm/conf/
└── config.yaml  (Single source of truth)
```

### Execution Plan for Config Removal

#### Phase 1: Migrate Dependencies (2 hours)

**Step 1: Update train_hyperparameter_sweep.py**
```python
# preprocessing/boughter/train_hyperparameter_sweep.py:279
# FROM:
def main(config_path: str | Path = "configs/config.yaml") -> int:

# TO:
@hydra.main(version_base=None, config_path="../../src/antibody_training_esm/conf", config_name="config")
def main(cfg: DictConfig) -> int:
```

**Step 2: Mark legacy train_model() as deprecated**
```python
# src/antibody_training_esm/core/trainer.py:793
def train_model(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    DEPRECATED: This function is deprecated and will be removed in v0.5.0.
    Use train_pipeline(cfg) with Hydra instead.
    """
    warnings.warn(
        "train_model() is deprecated. Use train_pipeline(cfg) with Hydra.",
        DeprecationWarning,
        stacklevel=2,
    )
```

#### Phase 2: Deprecation Period (v0.4.0)

**Add deprecation notice to configs/config.yaml**
```yaml
# configs/config.yaml
# DEPRECATED: This config file is deprecated and will be removed in v0.5.0
# Use src/antibody_training_esm/conf/config.yaml with Hydra instead
```

**Update documentation**
- Add migration guide in `docs/migration/v0.4.0-to-v0.5.0.md`
- Update `CLAUDE.md` to reflect deprecation
- Add CHANGELOG entry

#### Phase 3: Removal (v0.5.0)

**Delete configs/ directory**
```bash
git rm -r configs/
```

**Remove legacy train_model() function**
```python
# Delete src/antibody_training_esm/core/trainer.py:793-850
```

**Update tests**
```bash
# Remove any tests still using configs/config.yaml
# Update tests to use Hydra configs
```

### Estimated Time: 2-3 hours (across multiple releases)

---

## Problem 3: Archive Hyperparameter Sweep Results

### Current State

**Files**:
```
hyperparameter_sweep_results/
├── best_config_20251102_170516.yaml
├── best_config_20251102_182542.yaml
├── final_sweep_results_20251102_170516.csv
├── final_sweep_results_20251102_182542.csv
└── sweep_results_20251102_*.csv (18 files)
```

**Reference**: `preprocessing/boughter/train_hyperparameter_sweep.py:168`
```python
def run_sweep(
    self, output_dir: str = "hyperparameter_sweep_results"
) -> pd.DataFrame:
```

### Target State

```
experiments/archive/
└── hyperparameter_sweeps_2025-11-02/
    ├── README.md
    └── [22 files moved from root]

preprocessing/boughter/train_hyperparameter_sweep.py:168
# Updated default path
def run_sweep(
    self, output_dir: str = "experiments/archive/hyperparameter_sweeps"
) -> pd.DataFrame:
```

### Execution Plan for Sweep Archive

#### Step 1: Update Code Reference (5 min)
```python
# preprocessing/boughter/train_hyperparameter_sweep.py
# Line 168: Change default parameter

FROM:
    def run_sweep(
        self, output_dir: str = "hyperparameter_sweep_results"
    ) -> pd.DataFrame:

TO:
    def run_sweep(
        self, output_dir: str = "experiments/archive/hyperparameter_sweeps"
    ) -> pd.DataFrame:
```

**Verification**:
```bash
grep "hyperparameter_sweep_results" preprocessing/boughter/train_hyperparameter_sweep.py
# Should return empty
```

#### Step 2: Create Archive Directory (2 min)
```bash
mkdir -p experiments/archive/hyperparameter_sweeps_2025-11-02
```

#### Step 3: Move Files with Git (5 min)
```bash
# Move all 22 files preserving history
for file in hyperparameter_sweep_results/*; do
    git mv "$file" experiments/archive/hyperparameter_sweeps_2025-11-02/
done

# Remove empty directory
rmdir hyperparameter_sweep_results/
```

#### Step 4: Add Archive README (5 min)
```bash
cat > experiments/archive/hyperparameter_sweeps_2025-11-02/README.md <<'EOF'
# Hyperparameter Sweep Results - November 2, 2025

## Provenance
- **Created**: 2025-11-02
- **Script**: `preprocessing/boughter/train_hyperparameter_sweep.py`
- **Dataset**: Boughter VH training data
- **Model**: ESM-1v + Logistic Regression

## Contents
- `sweep_results_*.csv` - Individual fold results (18 files)
- `final_sweep_results_*.csv` - Aggregated results (2 files)
- `best_config_*.yaml` - Optimal hyperparameters (2 files)

## Results
- **Best C**: 1.0 (optimal regularization)
- **Best penalty**: L2
- **Best solver**: lbfgs
- **Cross-validation accuracy**: ~66% on Jain test set

## Archive Location
These files were moved from root `hyperparameter_sweep_results/` during
repository cleanup (2025-11-14) to improve organization.

Original commit history preserved via `git mv`.
EOF
```

#### Step 5: Test Script (3 min)
```bash
# Dry run to verify new default path works
python preprocessing/boughter/train_hyperparameter_sweep.py --help
```

#### Step 6: Commit (5 min)
```bash
git add preprocessing/boughter/train_hyperparameter_sweep.py
git add experiments/archive/hyperparameter_sweeps_2025-11-02/
git commit -m "chore: Archive hyperparameter sweep results to experiments/

- Move 22 sweep result files from root to experiments/archive/
- Update train_hyperparameter_sweep.py default output path
- Add README documenting provenance and results
- Preserves git history via git mv

This cleanup improves root directory organization while maintaining
full historical context of November 2025 sweep experiments."
```

### Estimated Time: 30 minutes

---

## Recommended Execution Order

### Immediate (This Session)
1. ✅ **Commit this planning document** - Single source of truth established
2. ⏸️ **Archive hyperparameter sweeps** - Easiest win (30 min)

### Next Sprint (Dedicated Time Block)
3. ⏸️ **Data directory consolidation** - Largest effort (13-15 hours)
   - Requires full day or 2-day sprint
   - Needs careful testing at each step

### Future Releases
4. ⏸️ **Config deprecation cycle** - Multi-release effort
   - v0.4.0: Add deprecation warnings
   - v0.5.0: Remove configs/ directory

---

## Success Criteria

### For This Session
- [x] Single planning document created
- [x] Hyperparameter sweeps archived (commit 683abde, Nov 14)
- [x] Tests pass (`make test` - 466 passed, 3 skipped, 90.66% coverage)
- [x] No breaking changes

### For Data Consolidation Sprint
- [x] Phase 1 complete: `test_datasets/` → `data/test/` (135 files, commit 288905c)
- [x] Phase 2 complete: `train_datasets/` → `data/train/` (commit 0d4605d)
- [x] Migration script created and tested (`/tmp/migrate_test_datasets.sh`)
- [x] Full test suite passes (133 tests, mypy, ruff)
- [x] Git history preserved (verified via `git log --follow`)
- [x] Documentation updated (TEST_DATASETS_CONSOLIDATION_PLAN.md)

### For Config Removal (v0.5.0)
- [ ] All dependencies migrated to Hydra
- [ ] Deprecation warnings added
- [ ] 1-2 release cycles elapsed
- [ ] configs/ directory removed
- [ ] Legacy train_model() removed

---

## Rollback Plan

If anything breaks:

```bash
# Revert all changes
git reset --hard HEAD~N

# Or revert specific commits
git revert <commit-hash>

# Restore moved files
git mv experiments/archive/hyperparameter_sweeps_2025-11-02/* hyperparameter_sweep_results/
git mv data/train/boughter train_datasets/
git mv data/test/* test_datasets/
```

---

## Questions for Decision

1. **Start with sweep archive?** (30 min, zero risk)
2. **Schedule data consolidation sprint?** (13-15 hours, dedicated time)
3. **Approve config deprecation timeline?** (v0.4.0 → v0.5.0)

---

**Ready to Execute** - Awaiting approval to proceed with sweep archive or data consolidation.
