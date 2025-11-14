# Repository Restructuring: Iron-Clad Action Plan

**Branch**: `chore/directory-structure-cleanup`
**Date**: 2025-11-14
**Status**: READY TO EXECUTE

---

## Executive Summary

After thorough audit, here's what we CAN and CANNOT safely do:

### Safe Actions (This Session)
1. ✅ **Archive hyperparameter sweeps** (30 min, 1 file change)
2. ✅ **Commit planning documents** (5 min)

### Deferred Actions (Future Releases)
3. ⏸️ **Data consolidation** - BLOCKED by 125 hardcoded path references
4. ⏸️ **Config removal** - BLOCKED by active dependencies, needs deprecation

---

## Action 1: Archive Hyperparameter Sweep Results ✅

### Verification
```bash
# Single code reference confirmed:
$ grep -r "hyperparameter_sweep_results" --include="*.py" .
preprocessing/boughter/train_hyperparameter_sweep.py:168

# 22 git-tracked files to move:
$ git ls-files hyperparameter_sweep_results/ | wc -l
22
```

### Steps (In Order)

#### Step 1: Update Code Reference
```python
# File: preprocessing/boughter/train_hyperparameter_sweep.py
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

**Verification**: `grep "hyperparameter_sweep_results" preprocessing/boughter/train_hyperparameter_sweep.py` returns empty

#### Step 2: Create Archive Directory
```bash
mkdir -p experiments/archive/hyperparameter_sweeps_2025-11-02
```

#### Step 3: Move Files with Git
```bash
# Move all 22 files preserving history
for file in hyperparameter_sweep_results/*; do
    git mv "$file" experiments/archive/hyperparameter_sweeps_2025-11-02/
done

# Remove empty directory
rmdir hyperparameter_sweep_results/
```

#### Step 4: Add Archive README
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

#### Step 5: Test Script Still Works
```bash
# Dry run to verify new default path works
python preprocessing/boughter/train_hyperparameter_sweep.py --help
```

#### Step 6: Commit
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

### Risk Assessment
- **Breaking changes**: None (existing files moved, code updated)
- **Test impact**: None (no tests reference this directory)
- **User impact**: Future sweeps write to new organized location (improvement)

**Verdict**: ✅ **SAFE TO EXECUTE**

---

## Action 2: Commit Planning Documents ✅

### Files to Commit
1. `CURRENT_STRUCTURE.txt` - Full tree snapshot (238 dirs, 577 files)
2. `DIRECTORY_CLEANUP_AUDIT.md` - Safety audit with verification
3. `DIRECTORY_STRUCTURE_PROPOSAL.md` - Long-term vision
4. `RESTRUCTURING_ACTION_PLAN.md` - This file

### Commit Message
```bash
git add CURRENT_STRUCTURE.txt DIRECTORY_CLEANUP_AUDIT.md \
        DIRECTORY_STRUCTURE_PROPOSAL.md RESTRUCTURING_ACTION_PLAN.md

git commit -m "docs: Add comprehensive directory restructuring plan

Planning documents for systematic repository cleanup:

- CURRENT_STRUCTURE.txt: Complete directory tree snapshot
- DIRECTORY_CLEANUP_AUDIT.md: Safety audit of proposed changes
- DIRECTORY_STRUCTURE_PROPOSAL.md: Long-term restructuring vision
- RESTRUCTURING_ACTION_PLAN.md: Immediate actionable steps

Key findings:
- Data consolidation blocked by 125 hardcoded path references
- Config removal requires deprecation period (defer to v0.5.0)
- Hyperparameter sweep archive: safe to execute (Action 1)

This establishes foundation for incremental, safe cleanup."
```

**Verdict**: ✅ **SAFE TO EXECUTE**

---

## Deferred Action 3: Data Directory Consolidation ⏸️

### Why Deferred
```bash
$ grep -r "train_datasets\|test_datasets" --include="*.py" src/ tests/ preprocessing/ scripts/ | wc -l
125
```

**125 hardcoded references** across:
- Source code (`src/`)
- Test suites (`tests/`)
- Preprocessing scripts (`preprocessing/`)
- Utility scripts (`scripts/`)
- Config files (`configs/`, `src/.../conf/`)

### Required Before Execution
1. **Full path audit** - Categorize all 125 references
2. **Migration script** - Automated refactoring tool
3. **Test strategy** - Incremental validation approach
4. **Docker updates** - Volume mounts, CI/CD workflows
5. **Documentation sweep** - Update all examples

### Estimated Effort
- Audit: 3 hours
- Script: 3 hours
- Execution: 4 hours
- Testing: 3 hours
- **Total: 13-15 hours**

**Recommendation**: Schedule for dedicated sprint, not ad-hoc cleanup

---

## Deferred Action 4: Config Directory Removal ⏸️

### Why Deferred
Active dependencies:
1. `src/antibody_training_esm/core/trainer.py:793` - `train_model()` legacy function
2. `preprocessing/boughter/train_hyperparameter_sweep.py:279` - `main()` default parameter

### Deprecation Path (v0.5.0)
1. **Now**: Add deprecation warning to `configs/config.yaml`
2. **v0.4.0**: Migrate scripts to Hydra, emit warnings
3. **v0.5.0**: Remove `configs/` and legacy `train_model()`

**Recommendation**: Follow proper deprecation cycle, don't rush

---

## Success Criteria

### For This Session
- [x] Iron-clad plan documented
- [ ] Hyperparameter sweeps archived (Action 1)
- [ ] Planning docs committed (Action 2)
- [ ] Tests pass (`make test`)
- [ ] No breaking changes

### Long-Term
- [ ] Data consolidation completed (future sprint)
- [ ] Config deprecation cycle started (v0.4.0)
- [ ] Config removal completed (v0.5.0)

---

## Execution Checklist

### Pre-Flight
- [x] Branch created: `chore/directory-structure-cleanup`
- [x] Full audit completed
- [x] Risk assessment documented
- [ ] Senior approval obtained

### Action 1: Hyperparameter Sweeps
- [ ] Code updated (train_hyperparameter_sweep.py:168)
- [ ] Archive directory created
- [ ] Files moved with `git mv`
- [ ] README added
- [ ] Script tested
- [ ] Changes committed

### Action 2: Planning Docs
- [ ] Documents reviewed
- [ ] Commit message drafted
- [ ] Changes committed

### Post-Flight
- [ ] Tests pass
- [ ] Git history clean
- [ ] No untracked files
- [ ] Branch ready for review

---

## Rollback Plan

If anything breaks:

```bash
# Revert all changes
git reset --hard HEAD~2

# Or revert specific commits
git revert <commit-hash>

# Restore moved files
git mv experiments/archive/hyperparameter_sweeps_2025-11-02/* hyperparameter_sweep_results/
```

---

## Questions & Answers

**Q: Why not do data consolidation now?**
A: 125 hardcoded references. Needs dedicated sprint, not ad-hoc cleanup.

**Q: Why not delete configs/ now?**
A: Active dependencies in production code. Needs deprecation cycle.

**Q: Why archive hyperparameter sweeps?**
A: Only 1 reference, clean history preservation, improves organization.

**Q: Will this break anything?**
A: No. We update code BEFORE moving files, tests verify correctness.

---

**READY TO EXECUTE** ✅

Senior approval to proceed with Actions 1 & 2?
