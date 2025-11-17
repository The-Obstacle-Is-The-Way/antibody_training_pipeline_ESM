# Cleanup Before Validation - Decision Document

**Date**: 2025-11-16
**Decision**: ✅ **CLEAN FIRST, THEN VALIDATE**

## Executive Summary

We will execute V0.5.0_CLEANUP_PLAN.md Problem 1 (remove `configs/config.yaml` + `train_model()`) BEFORE running end-to-end validation.

**Rationale**: Validation should test the FINAL production state, not a transitional state with legacy artifacts.

---

## Evidence-Based Analysis

### What Validation Actually Uses

**Training** (VALIDATION_ROADMAP.md:312-321):
```bash
uv run antibody-train  # Uses Hydra: src/antibody_training_esm/conf/config.yaml
```
- ✅ Uses Hydra configs in `src/antibody_training_esm/conf/`
- ❌ Does NOT use `configs/config.yaml`
- ❌ Does NOT call `train_model()` function

**Testing** (VALIDATION_ROADMAP.md:365-449):
```bash
uv run antibody-test --config configs/testing/jain_p5e_s2.yaml  # Uses TestConfig
uv run antibody-test --model <pkl> --data <csv>  # Uses TestConfig
```
- ✅ Uses `TestConfig` class from CLI args or YAML
- ❌ Does NOT use `configs/config.yaml`

### What Will Be Removed

**File**: `configs/config.yaml` (3,324 bytes)
- **Status**: Legacy, not used by current CLI tools
- **References**: Only in deprecated code paths

**Function**: `train_model(config_path="configs/config.yaml")` (src/antibody_training_esm/core/trainer.py:870)
- **Status**: DEPRECATED since v0.4.0
- **Replacement**: `train_pipeline(cfg)` (Hydra-based)
- **Used by**:
  - ❌ `tests/e2e/test_train_pipeline.py` (2 tests)
  - ❌ `tests/unit/core/test_trainer.py` (deprecation warning test)
  - ✅ **NOT used by validation commands**

**Function**: `main(config_path="configs/config.yaml")` (preprocessing/boughter/train_hyperparameter_sweep.py:279)
- **Status**: Legacy preprocessing script
- **Used by**: Manual hyperparameter sweeps (not part of validation)

---

## Decision Matrix

| Factor | Clean First | Clean After |
|--------|-------------|-------------|
| **Validates production state** | ✅ YES | ❌ NO (validates legacy state) |
| **Risk of breaking validation** | ✅ LOW (validation doesn't use legacy code) | ⚠️ MEDIUM (legacy code could interfere) |
| **Total validation runs needed** | ✅ ONE (validate clean state) | ❌ TWO (before + after cleanup) |
| **Clarity for users** | ✅ HIGH (test what ships) | ⚠️ LOW (what's the point of validating old state?) |
| **Debugging complexity** | ⚠️ MEDIUM (if failure, is it cleanup?) | ✅ LOW (know it's not cleanup) |
| **Total time investment** | ✅ LOWER (validate once) | ❌ HIGHER (validate twice) |

**Winner**: Clean First (4 advantages vs 1 disadvantage)

---

## Execution Order

### Step 1: Execute V0.5.0_CLEANUP_PLAN.md Problem 1

```bash
# 1a. Remove legacy config file
git rm configs/config.yaml

# 1b. Remove train_model() from trainer.py
#     (Keep train_pipeline() - the Hydra version)

# 1c. Update tests to use train_pipeline()
#     - tests/e2e/test_train_pipeline.py
#     - tests/unit/core/test_trainer.py (remove deprecation test)

# 1d. Update preprocessing scripts
#     - preprocessing/boughter/train_hyperparameter_sweep.py

# 1e. Commit cleanup
git add -A
git commit -m "refactor: Remove legacy configs/config.yaml and train_model() API

- Remove configs/config.yaml (replaced by Hydra configs in src/)
- Remove train_model() function (replaced by train_pipeline())
- Update tests to use Hydra-based train_pipeline()
- Update preprocessing scripts to use Hydra

Breaking change: train_model() API removed (deprecated since v0.4.0)
Migration: Use 'uv run antibody-train' with Hydra configs"
```

### Step 2: Create Jain Test Config

```bash
mkdir -p configs/testing
cat > configs/testing/jain_p5e_s2.yaml << 'EOF'
model_paths: [experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl]
data_paths: [data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv]
sequence_column: vh_sequence
label_column: label
output_dir: experiments/runs/tests/jain
device: cpu
batch_size: 8
EOF

git add configs/testing/jain_p5e_s2.yaml
git commit -m "config: Add Jain test config for vh_sequence column mapping"
```

### Step 3: Run Full Validation

```bash
# Follow VALIDATION_ROADMAP.md exactly:
# - Phase 2: Training Pipeline (Tasks 2.1-2.3)
# - Phase 3: Testing Pipeline (Tasks 3.1-3.3)
# - Phase 4: Hyperparameter Sweeps (Task 4.1)
# - Phase 5: Documentation (Task 5.1)
# - Phase 6: Fresh Clone Test (Task 6.1)
```

---

## Risk Mitigation

### If Cleanup Breaks Something

**Problem**: Test failures after removing legacy code
**Solution**:
1. Check git status - cleanup is in its own commit
2. Revert cleanup commit: `git revert HEAD`
3. Investigate what actually depends on legacy code
4. Update dependencies to use Hydra
5. Re-apply cleanup

**Backup Plan**: We can always restore `configs/config.yaml` from git history if truly needed

### If Validation Finds Unrelated Issues

**Problem**: Validation fails for reasons unrelated to cleanup
**Solution**:
1. This is GOOD - we found issues before release
2. Fix validation issues
3. Re-run validation
4. Cleanup was correct - issues were pre-existing

---

## Conclusion

**Decision**: ✅ **CLEAN FIRST, THEN VALIDATE**

**Next Steps**:
1. Execute V0.5.0_CLEANUP_PLAN.md Problem 1 (remove legacy code)
2. Create `configs/testing/jain_p5e_s2.yaml`
3. Run VALIDATION_ROADMAP.md end-to-end
4. Document results in POST_VALIDATION_SUMMARY.md

**Why This Is Correct**:
- Validation commands use Hydra/TestConfig, not legacy code
- Validating production state (what users get) is the right approach
- Lower total effort (validate once vs twice)
- Clearer signal if something breaks (we know immediately)
