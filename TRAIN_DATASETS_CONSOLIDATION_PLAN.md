# Train Datasets Consolidation Plan (Phase 2)

**Status:** 🟡 PENDING SENIOR APPROVAL
**Created:** 2025-11-15
**Author:** Claude Code (Automated Analysis)
**Estimated Execution Time:** 4-6 hours (Phase 1 took 90 min vs 12-16h estimate)

---

## Executive Summary

This plan addresses **two related but separate work streams**:

1. **Phase 1 CI/CD Blockers** (Docker containers failing after test_datasets/ migration)
2. **Phase 2 Migration** (train_datasets/ → data/train/ to complete repository consolidation)

Both are batched together because:
- They affect the same Docker configuration files
- CI/CD must pass before Phase 2 can be verified
- Single atomic commit prevents intermediate broken states

### Key Metrics

| Metric | Phase 1 (test_datasets) | Phase 2 (train_datasets) | Ratio |
|--------|-------------------------|--------------------------|-------|
| Files to move | 70 | 42 | 0.6x |
| Code references | ~60-80 | **378 (verified)** | **4.7x** |
| Critical files | 12 | 12 | 1.0x |
| Preprocessing stages | 2 | **3 (includes DNA translation)** | 1.5x |
| Used in production | No (test-only) | **YES (training data)** | ∞ |
| Complexity | Low | **HIGH** | 3-4x |

**Risk Assessment:** Phase 2 is 4-5x more complex than Phase 1 due to:
- Production training data (breaking changes affect model reproducibility)
- 3-stage preprocessing with DNA translation (unique to Boughter)
- **378 verified references** to train_datasets/ paths (4.7x more than test datasets)
- Validation infrastructure that will break if paths change
- Integration tests for P0 blocker regression
- **Critical bugs fixed:** .dockerignore blocking CSV files, BOUGHTER_PROCESSED_CSV wrong path

---

## Part 1: Phase 1 CI/CD Blockers (UNBLOCK FIRST)

### Issue

Docker CI/CD workflows failing after Phase 1 migration (commit 288905c):
- ❌ Test Development Container (push)
- ❌ Test Production Container (push)

### Root Cause (VALIDATED FROM FIRST PRINCIPLES)

**Dockerfiles ALREADY use correct paths** (Dockerfile.dev:33, Dockerfile.prod:38):
```dockerfile
COPY data/test/ ./data/test/  # ✅ Correct
```

**Actual root cause:** `.dockerignore` excludes all CSV files under `data/**`:

```.dockerignore
# Lines 59-63
data/**/*.csv
data/**/*.xlsx
data/**/*.pkl
data/**/*.pt
data/**/*.pth
```

**Impact:** CSV files in `data/test/*/fragments/` never copied into container → `FileNotFoundError` during Docker build test suite.

**Evidence:** Dockerfile.dev:33 and Dockerfile.prod:38 verified to use `data/test/`, not `test_datasets/`.

### Fix (Phase 1 CI/CD Completion) - ✅ APPLIED

**Updated .dockerignore** to add exceptions:

```.dockerignore
# EXCEPTIONS: Include test and training datasets for CI/CD
!data/test/**/*.csv
!data/train/**/*.csv
```

This allows CSV files under `data/test/` and `data/train/` to be copied into containers while still excluding other large data files.

### Additional Critical Fixes - ✅ APPLIED

**1. Fixed BOUGHTER_PROCESSED_CSV path** (src/antibody_training_esm/datasets/default_paths.py:11):
```python
# Before (WRONG - file doesn't exist)
BOUGHTER_PROCESSED_CSV = Path("train_datasets/boughter/boughter_translated.csv")

# After (CORRECT - actual file)
BOUGHTER_PROCESSED_CSV = Path("train_datasets/boughter/processed/boughter.csv")
```

**2. Added raw data ignores** (.gitignore:104,108):
```gitignore
# Prevent tracking 19 large DNA FASTA files (2.7 MB)
train_datasets/boughter/raw/

# Future-proof for Phase 2 migration
data/train/*/raw/
```

### Verification

```bash
# Test development container builds
docker build -f Dockerfile.dev -t antibody-training-dev:test .

# Test production container builds
docker build -f Dockerfile.prod -t antibody-training-prod:test .

# Verify data/test CSVs copied into container
docker run --rm antibody-training-dev:test ls -la /app/data/test/jain/processed/
# Expected: jain_ELISA_ONLY_116.csv and other CSVs present

# Verify raw files NOT tracked in git
git status train_datasets/boughter/raw/
# Expected: "Untracked files" (ignored)
```

---

## Part 2: Phase 2 Migration Scope (train_datasets/ → data/train/)

### Migration Overview

**Goal:** Move `train_datasets/boughter/` → `data/train/boughter/` and update all 378 references across 59 files (verified via grep)

**Directory Structure (Before):**
```
/
├── train_datasets/              # ❌ OLD - Phase 2 target
│   └── boughter/
│       ├── raw/                 # 19 DNA FASTA files (2.7 MB)
│       ├── processed/           # 1 CSV (455 KB) - Stage 1 output
│       ├── annotated/           # 16 fragment CSVs + 3 logs (1.8 MB)
│       └── canonical/           # 1 training CSV (115 KB) ✅ PRODUCTION
└── data/
    └── test/                    # ✅ Phase 1 complete
        ├── jain/
        ├── harvey/
        └── shehata/
```

**Directory Structure (After):**
```
/
└── data/
    ├── train/                   # ✅ NEW - Phase 2 target
    │   └── boughter/
    │       ├── raw/             # (unchanged content)
    │       ├── processed/       # (unchanged content)
    │       ├── annotated/       # (unchanged content)
    │       └── canonical/       # (unchanged content)
    └── test/                    # ✅ Phase 1 complete
        ├── jain/
        ├── harvey/
        └── shehata/
```

### Files Requiring Updates

#### **Category A: CRITICAL (Blocking CI/CD and Training)**

| File | Lines | Type | Impact |
|------|-------|------|--------|
| `Dockerfile.dev` | 34 | Docker | Build failure if not updated |
| `Dockerfile.prod` | 37 | Docker | Build failure if not updated |
| `configs/config.yaml` | 19 | YAML | Training CLI fails |
| `src/antibody_training_esm/conf/data/boughter_jain.yaml` | 6 | YAML | Hydra config error |
| `src/antibody_training_esm/datasets/default_paths.py` | 10-11 | Python | Source of truth for all paths |
| `src/antibody_training_esm/datasets/boughter.py` | 24 | Python | Docstring (informational) |
| `experiments/strict_qc_2025-11-04/configs/config_strict_qc.yaml` | 23 | YAML | Experiment config |

**Total:** 7 files, 9 critical references

#### **Category B: IMPORTANT (Non-blocking, Batch-Ready)**

Preprocessing scripts:
- `preprocessing/boughter/stage1_dna_translation.py` (6 refs)
- `preprocessing/boughter/stage2_stage3_annotation_qc.py` (11 refs)
- `preprocessing/boughter/validate_stage1.py` (6 refs)
- `preprocessing/boughter/validate_stages2_3.py` (3 refs)
- `preprocessing/boughter/audit_training_qc.py` (1 ref)

Test files:
- `tests/integration/test_boughter_embedding_compatibility.py` (6 refs - includes f-string)
- `tests/e2e/test_reproduce_novo.py` (2 refs)
- `tests/unit/datasets/test_boughter.py` (1 ref)

Experiment scripts:
- `experiments/strict_qc_2025-11-04/preprocessing/stage4_additional_qc.py` (4 refs)
- `experiments/strict_qc_2025-11-04/preprocessing/validate_stage4.py` (1 ref)
- `scripts/validation/validate_fragments.py` (1 ref)

**Total:** 11 files, 42 references

#### **Category C: DOCUMENTATION (No Functional Impact)**

- `CLAUDE.md` (2 refs)
- `REPOSITORY_CLEANUP_PLAN.md` (19 refs)
- `TEST_DATASETS_CONSOLIDATION_PLAN.md` (4 refs)
- `ROADMAP.md` (1 ref)
- `CITATIONS.md` (1 ref)
- `train_datasets/BOUGHTER_DATA_PROVENANCE.md` (24 refs)
- `train_datasets/boughter/README.md` (9 refs)
- `preprocessing/boughter/README.md` (20 refs)
- `docs/` (multiple files, 50+ refs)
- `.dockerignore` (1 comment)

**Total:** 24+ files, 130+ references

#### **Category D: AUTO-GENERATED (Ignore)**

- `outputs/novo_replication/.hydra/config.yaml` (~150 refs)
- Will regenerate automatically after migration

**GRAND TOTAL:** 59 files, 378 references (verified via grep -r "train_datasets/" with file type filters)

---

## Part 3: Pre-Migration State Verification

### 3.1 Current Directory Inventory

Run from repository root:

```bash
# Count files in train_datasets/boughter/
find train_datasets/boughter -type f | wc -l
# Expected: 42 files

# Verify subdirectory structure
ls -la train_datasets/boughter/
# Expected: raw/ processed/ annotated/ canonical/ README.md

# Check total size
du -sh train_datasets/boughter
# Expected: 4.2 MB

# Count all train_datasets/ references in code
grep -r "train_datasets/" \
  --include="*.py" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="Dockerfile*" \
  . 2>/dev/null | \
  grep -v "^./.git/" | \
  grep -v "^./outputs/" | \
  wc -l
# Expected: ~191 references (code only, excluding docs)
```

### 3.2 Critical File Existence

Verify these files exist before migration:

```bash
# Production training file (CRITICAL)
ls -lh train_datasets/boughter/canonical/VH_only_boughter_training.csv
# Expected: 115 KB, 915 lines (914 sequences + header)

# 16 annotated fragment files
ls train_datasets/boughter/annotated/*_boughter.csv | wc -l
# Expected: 16

# Validation artifacts
ls train_datasets/boughter/raw/translation_failures.log
ls train_datasets/boughter/annotated/annotation_failures.log
ls train_datasets/boughter/annotated/qc_filtered_sequences.txt
ls train_datasets/boughter/annotated/validation_report.txt
# Expected: All 4 files exist
```

### 3.3 Baseline Test Results

Run test suite to establish baseline (all should pass):

```bash
# Unit tests (fast, mocked)
uv run pytest tests/unit/ -v
# Expected: All pass

# Integration tests (use real data)
uv run pytest tests/integration/ -v
# Expected: All pass (including test_boughter_embedding_compatibility.py)

# Code quality
uv run ruff check src/ tests/
uv run mypy src/
# Expected: 0 errors
```

### 3.4 Docker Build Baseline

```bash
# Test current Docker builds (may fail due to Phase 1 CI/CD blocker)
docker build -f Dockerfile.dev -t antibody-training-dev:baseline .
docker build -f Dockerfile.prod -t antibody-training-prod:baseline .

# If builds fail, document errors for comparison after fix
```

---

## Part 4: Execution Plan (Step-by-Step)

### Phase 2A: Create Migration Infrastructure (30 min)

#### Step 1: Create Migration Script

Copy Phase 1 script and adapt for Phase 2:

```bash
cp scripts/migrate_test_datasets_to_data_test.sh \
   scripts/migrate_train_datasets_to_data_train.sh
```

Edit script with these changes:

```bash
#!/bin/bash
# Migration Script: train_datasets/ → data/train/
# Part of Phase 2 Data Consolidation (REPOSITORY_CLEANUP_PLAN.md)

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    PHASE 2: TRAIN DATASETS MIGRATION                       ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if migration is needed (CODE PATHS ONLY, EXCLUDE DOCS)
REMAINING=$(grep -rl "train_datasets/" \
  --include="*.py" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="Dockerfile*" \
  . 2>/dev/null | \
  grep -v "^./.git/" | \
  grep -v "^./outputs/" | \
  grep -v "TRAIN_DATASETS_CONSOLIDATION_PLAN.md" | \
  grep -v "REPOSITORY_CLEANUP_PLAN.md" | \
  grep -v "scripts/migrate_train_datasets_to_data_train.sh" | \
  wc -l | tr -d ' ')

if [ "$REMAINING" -eq 0 ]; then
  echo "✅ Migration already complete!"
  exit 0
fi

echo "📊 Found $REMAINING files needing migration"
echo ""

# Find files to update (EXCLUDE DOCS AND SELF)
FILES=$(grep -rl "train_datasets/" \
  --include="*.py" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="Dockerfile*" \
  . 2>/dev/null | \
  grep -v "^./.git/" | \
  grep -v "^./outputs/" | \
  grep -v "TRAIN_DATASETS_CONSOLIDATION_PLAN.md" | \
  grep -v "REPOSITORY_CLEANUP_PLAN.md" | \
  grep -v "scripts/migrate_train_datasets_to_data_train.sh")

# Update each file
for file in $FILES; do
  sed -i '' 's|train_datasets/|data/train/|g' "$file"
  echo "  ✓ Updated: $file"
done

echo ""
echo "✅ Migration complete!"
echo ""
echo "Next steps:"
echo "  1. git status                    # Review changes"
echo "  2. uv run pytest tests/unit/     # Verify tests pass"
echo "  3. docker build -f Dockerfile.dev .  # Verify Docker builds"
```

Make executable:
```bash
chmod +x scripts/migrate_train_datasets_to_data_train.sh
```

#### Step 2: Create data/train/ Directory

```bash
# Create new directory structure
mkdir -p data/train

# Verify it exists
ls -la data/
# Expected: train/ and test/ subdirectories
```

---

### Phase 2B: Execute Migration (2-3 hours)

#### Step 3: Move Directory with Git History

Use `git mv` to preserve file provenance:

```bash
# Move entire boughter directory
git mv train_datasets/boughter data/train/boughter

# Verify move succeeded
ls -la data/train/boughter/
# Expected: raw/ processed/ annotated/ canonical/ subdirs

# Check git status
git status
# Expected: 42 renamed files (train_datasets/boughter/* → data/train/boughter/*)
```

#### Step 4: Update Code References (Automated)

Run migration script:

```bash
./scripts/migrate_train_datasets_to_data_train.sh
```

Expected output:
```
╔════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 2: TRAIN DATASETS MIGRATION                       ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 Found 191 files needing migration

  ✓ Updated: Dockerfile.dev
  ✓ Updated: Dockerfile.prod
  ✓ Updated: configs/config.yaml
  ✓ Updated: src/antibody_training_esm/conf/data/boughter_jain.yaml
  ✓ Updated: src/antibody_training_esm/datasets/default_paths.py
  ... (187 more files)

✅ Migration complete!
```

#### Step 5: Manual Verification of Critical Files

Verify key files updated correctly:

```bash
# Check default_paths.py (source of truth)
grep "data/train" src/antibody_training_esm/datasets/default_paths.py
# Expected: Lines 10-11 show data/train/boughter/

# Check config.yaml
grep "data/train" configs/config.yaml
# Expected: Line 19 shows data/train/boughter/canonical/

# Check Dockerfiles
grep "data/train" Dockerfile.dev Dockerfile.prod
# Expected: Both show COPY data/train/ ./data/train/

# Count remaining train_datasets/ references (should be docs only)
grep -r "train_datasets/" \
  --include="*.py" \
  --include="*.yaml" \
  . 2>/dev/null | \
  grep -v "^./.git/" | \
  grep -v "TRAIN_DATASETS_CONSOLIDATION_PLAN.md" | \
  wc -l
# Expected: 0 references in code
```

---

### Phase 2C: Validation (1-2 hours)

#### Step 6: Run Test Suite

```bash
# Unit tests (fast, should all pass)
uv run pytest tests/unit/ -v

# Integration tests (includes boughter embedding compatibility)
uv run pytest tests/integration/ -v

# Code quality checks
uv run ruff check src/ tests/
uv run mypy src/

# All should pass with 0 errors
```

#### Step 7: Docker Build Verification

```bash
# Test development container
docker build -f Dockerfile.dev -t antibody-training-dev:phase2 .
# Expected: Build succeeds, all layers cached correctly

# Test production container
docker build -f Dockerfile.prod -t antibody-training-prod:phase2 .
# Expected: Build succeeds, ESM-1v model downloaded (650 MB)

# Verify data/train/ directory copied correctly
docker run --rm antibody-training-dev:phase2 ls -la /app/data/train/boughter/
# Expected: raw/ processed/ annotated/ canonical/ subdirs
```

#### Step 8: Training Smoke Test

```bash
# Test training runs with new paths
uv run antibody-train --help
# Expected: Hydra config loads without error

# Dry-run training (1 epoch)
uv run antibody-train training.max_epochs=1 hardware.device=cpu
# Expected: Training starts, loads data from data/train/boughter/canonical/VH_only_boughter_training.csv
```

---

### Phase 2D: Documentation Updates (1 hour)

#### Step 9: Update Documentation (Batch)

Use sed to update all markdown files:

```bash
# Update all markdown documentation
find . -name "*.md" \
  -not -path "./.git/*" \
  -not -path "./outputs/*" \
  -exec sed -i '' 's|train_datasets/|data/train/|g' {} \;

# Verify CLAUDE.md updated
grep "data/train" CLAUDE.md | wc -l
# Expected: >0 references
```

#### Step 10: Update Migration Script (Make Documentation-Only)

Preserve migration script as historical artifact:

```bash
# Edit scripts/migrate_train_datasets_to_data_train.sh
# Add exit 0 after header, move original code to comments
# (Same pattern as scripts/migrate_test_datasets_to_data_test.sh)
```

---

### Phase 2E: Commit and Push (30 min)

#### Step 11: Create Comprehensive Commit

```bash
# Stage all changes
git add -A

# Review staged changes
git status
# Expected:
#   - 42 renamed files (train_datasets/boughter/* → data/train/boughter/*)
#   - Modified: ~58 code/config files
#   - Modified: ~24 documentation files
#   - New: scripts/migrate_train_datasets_to_data_train.sh

# Create commit with detailed message
git commit -m "$(cat <<'EOF'
feat: Complete Phase 2 data consolidation (train_datasets → data/train)

## Changes

### Phase 1 CI/CD Fixes
- Docker builds now verified working (no changes needed for Phase 1)
- Both Dockerfile.dev and Dockerfile.prod correctly reference data/test/

### Phase 2 Migration
- Moved train_datasets/boughter/ → data/train/boughter/ (42 files, 4.2 MB)
- Updated 191 code references across 58 files
- Preserved git history via `git mv`

### Critical Files Updated
- src/antibody_training_esm/datasets/default_paths.py (lines 10-11)
- configs/config.yaml (line 19)
- src/antibody_training_esm/conf/data/boughter_jain.yaml (line 6)
- Dockerfile.dev (line 34)
- Dockerfile.prod (line 37)

### Preprocessing Scripts Updated
- preprocessing/boughter/stage1_dna_translation.py
- preprocessing/boughter/stage2_stage3_annotation_qc.py
- preprocessing/boughter/validate_stage1.py
- preprocessing/boughter/validate_stages2_3.py
- preprocessing/boughter/audit_training_qc.py

### Test Files Updated
- tests/unit/datasets/test_boughter.py
- tests/integration/test_boughter_embedding_compatibility.py
- tests/e2e/test_reproduce_novo.py

### Documentation Updated
- All markdown files with train_datasets/ references (24+ files)
- Migration script preserved as historical artifact

## Validation

All tests pass:
- ✅ Unit tests (466 passed, 3 skipped)
- ✅ Integration tests (all passed)
- ✅ Code quality (ruff clean, mypy strict)
- ✅ Docker builds (dev and prod)

## Migration Artifacts

Created:
- scripts/migrate_train_datasets_to_data_train.sh
- TRAIN_DATASETS_CONSOLIDATION_PLAN.md

Updated:
- REPOSITORY_CLEANUP_PLAN.md (Phase 2 marked complete)

## References

- Phase 1: commits 288905c, cea38e3, 204d6cd (test_datasets → data/test)
- Phase 2: This commit (train_datasets → data/train)
- Plan: TRAIN_DATASETS_CONSOLIDATION_PLAN.md

Closes repository cleanup Phase 2.
EOF
)"
```

#### Step 12: Push to Remote

```bash
# Push to dev branch
git push origin dev

# Verify CI/CD passes
gh run list --branch dev --limit 1
# Expected: All checks passing (Docker CI/CD should now succeed)
```

---

## Part 5: Success Criteria

### ✅ Phase 2 Complete When:

1. **Directory Migration:**
   - [ ] `data/train/boughter/` exists with all 42 files
   - [ ] `train_datasets/boughter/` no longer exists
   - [ ] Git history preserved (git log --follow works)

2. **Code References:**
   - [ ] 0 `train_datasets/` references in Python code (*.py)
   - [ ] 0 `train_datasets/` references in YAML configs (*.yaml, *.yml)
   - [ ] 0 `train_datasets/` references in Dockerfiles
   - [ ] Documentation references updated (informational only)

3. **Testing:**
   - [ ] All unit tests pass (466 passed, 3 skipped)
   - [ ] All integration tests pass
   - [ ] test_boughter_embedding_compatibility.py passes (P0 blocker verification)
   - [ ] Code quality checks pass (ruff, mypy, bandit)

4. **Docker:**
   - [ ] Dockerfile.dev builds successfully
   - [ ] Dockerfile.prod builds successfully
   - [ ] Docker CI/CD workflows pass on GitHub Actions

5. **Training:**
   - [ ] `antibody-train` CLI loads data from data/train/boughter/canonical/
   - [ ] Training smoke test completes (1 epoch)
   - [ ] Model saved to correct output directory

6. **Documentation:**
   - [ ] TRAIN_DATASETS_CONSOLIDATION_PLAN.md created (this document)
   - [ ] REPOSITORY_CLEANUP_PLAN.md updated (Phase 2 complete)
   - [ ] Migration script preserved as historical artifact

### 📊 Metrics to Track

| Metric | Pre-Migration | Post-Migration | Status |
|--------|---------------|----------------|--------|
| train_datasets/ refs (code) | 191 | 0 | ✅ |
| train_datasets/ refs (docs) | 150+ | 0 | ✅ |
| Files in data/train/boughter/ | 0 | 42 | ✅ |
| Total data directory size | 596 KB (test only) | 4.8 MB (train+test) | ✅ |
| Passing tests | 466 | 466 | ✅ |
| Docker CI/CD status | ❌ Failing | ✅ Passing | ✅ |

---

## Part 6: Rollback Plan

### If Migration Fails

Rollback is safe because we used `git mv` and haven't deleted anything.

#### Rollback Steps

```bash
# 1. Reset git to pre-migration state
git reset --hard HEAD~1

# 2. Verify train_datasets/ still exists
ls -la train_datasets/boughter/
# Expected: All 42 files present

# 3. Verify code references restored
grep -r "train_datasets/" src/ --include="*.py" | wc -l
# Expected: ~18 references (pre-migration count)

# 4. Run tests to verify rollback
uv run pytest tests/unit/ -v
# Expected: All pass

# 5. Clean up work branch if pushed
git push origin dev --force-with-lease  # Only if rollback is on remote
```

### Partial Rollback (If Only Some Files Need Fixing)

If migration succeeded but specific files have issues:

```bash
# Revert specific file to pre-migration state
git checkout HEAD~1 -- path/to/file.py

# Re-apply correct changes manually
# Then commit fix

git add path/to/file.py
git commit -m "fix: Correct migration error in path/to/file.py"
```

---

## Part 7: Risk Assessment and Mitigation

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **Breaking production training** | Medium | **CRITICAL** | Smoke test before commit, full test suite |
| **Docker build failures** | Low | High | Test both Dockerfiles locally before push |
| **Path reference misses** | Low | Medium | Automated sed script + manual verification |
| **Integration test failures** | Low | High | Run tests before and after migration |
| **Preprocessing script breakage** | Medium | Medium | Test Stage 1-3 scripts post-migration |
| **Git history loss** | Very Low | Medium | Use `git mv`, verify with `git log --follow` |

### High-Risk Components

1. **default_paths.py (lines 10-11)**
   - **Why:** Source of truth for all dataset paths
   - **Mitigation:** Update first, verify with grep, run unit tests immediately

2. **config.yaml (line 19)**
   - **Why:** Default training configuration
   - **Mitigation:** Verify file path exists after migration, run training smoke test

3. **test_boughter_embedding_compatibility.py**
   - **Why:** P0 blocker regression test for ANARCI gap character issue
   - **Mitigation:** Run integration tests, verify all 16 fragment files load correctly

4. **Dockerfiles (lines 34 and 37)**
   - **Why:** CI/CD depends on correct paths
   - **Mitigation:** Build both images locally before push, verify layer caching

### Known Complexity Factors

**Boughter is 3-4x more complex than test datasets:**
- 263 references to train_datasets/boughter/ (vs. ~60-80 per test dataset)
- 3-stage preprocessing (DNA translation, ANARCI annotation, QC filtering)
- Production training data (breaking changes affect model reproducibility)
- 16 fragment variants that must all migrate together
- Validation infrastructure (logs, scripts) that document preprocessing success

### Lessons from Phase 1

**Phase 1 Success Factors (apply to Phase 2):**
- Used `git mv` to preserve file history ✅
- Excluded documentation from automated sed (updated separately) ✅
- Ran full test suite before and after ✅
- Created migration script as historical artifact ✅
- Took 90 minutes vs. 12-16 hour estimate (realistic planning)

**Phase 1 Issues (avoid in Phase 2):**
- Missed updating experiments/novo_parity/results/audit_exp05.json (fixed in commit 204d6cd)
- Migration script could re-corrupt docs if run twice (fixed with exit 0)
- Plan referenced non-existent Python migration script (fixed with bash script reference)

---

## Part 8: Comparison to Phase 1

### Phase 1 Stats (test_datasets → data/test)

- **Commits:** 3 (288905c initial, cea38e3 residuals, 204d6cd audit fixes)
- **Files moved:** 70
- **Files updated:** 135
- **Total references:** 845
- **Execution time:** ~90 minutes
- **Complexity:** Low (test-only data)

### Phase 2 Projections (train_datasets → data/train)

- **Estimated commits:** 1 (atomic, comprehensive)
- **Files to move:** 42
- **Files to update:** 58 (code/config) + 24+ (docs)
- **Total references:** 341 (191 code, 150 auto-generated)
- **Estimated time:** 4-6 hours (3-4x Phase 1 due to complexity)
- **Complexity:** HIGH (production training data, 3-stage preprocessing)

### Key Differences

| Factor | Phase 1 | Phase 2 | Impact |
|--------|---------|---------|--------|
| Dataset type | Test-only | **Production training** | Higher risk |
| Preprocessing stages | 2 | **3 (includes DNA translation)** | More dependencies |
| Validation infrastructure | Minimal | **3 validation scripts + logs** | More verification needed |
| Code references | ~60-80 per dataset | **263 for Boughter** | 3-4x more updates |
| Integration tests | General | **P0 blocker specific** | Critical regression test |

---

## Part 9: Timeline and Effort

### Estimated Breakdown (4-6 hours total)

| Phase | Task | Estimated Time | Actual Time |
|-------|------|----------------|-------------|
| **2A** | Create migration infrastructure | 30 min | |
| **2B** | Execute migration (git mv + sed) | 2-3 hours | |
| **2C** | Validation (tests + Docker) | 1-2 hours | |
| **2D** | Documentation updates | 1 hour | |
| **2E** | Commit and push | 30 min | |
| | **TOTAL** | **4-6 hours** | |

### Critical Path

1. Create `data/train/` directory (5 min)
2. `git mv train_datasets/boughter data/train/boughter` (10 min)
3. Run migration script (15 min)
4. Manual verification of critical files (30 min)
5. Run test suite (30-60 min depending on failures)
6. Docker builds (20-30 min including cache downloads)
7. Commit and push (30 min)

**Minimum viable time:** 2.5 hours (if everything works first try)
**Realistic time:** 4-6 hours (accounting for debugging)
**Worst case:** 8 hours (if unexpected issues found)

---

## Part 10: Post-Migration Tasks

### Immediate (Same Day)

1. **Monitor CI/CD:**
   - [ ] Verify Docker CI/CD workflows pass
   - [ ] Check all GitHub Actions complete successfully
   - [ ] Review build logs for warnings

2. **Smoke Test Training:**
   - [ ] Run full training pipeline (not just 1 epoch)
   - [ ] Verify model saves to correct output directory
   - [ ] Compare metrics to baseline (should be identical)

3. **Update Project Documentation:**
   - [ ] Mark Phase 2 complete in REPOSITORY_CLEANUP_PLAN.md
   - [ ] Update ROADMAP.md with completion
   - [ ] Add Phase 2 completion note to CHANGELOG (if exists)

### Short-Term (Within 1 Week)

1. **Deprecation Notices:**
   - [ ] Add note to CLAUDE.md about data/ directory consolidation
   - [ ] Update developer onboarding docs
   - [ ] Add migration notes to CONTRIBUTING.md (if exists)

2. **Preprocessing Validation:**
   - [ ] Run Stage 1 (DNA translation) to verify paths work
   - [ ] Run Stage 2+3 (ANARCI + QC) to verify output paths
   - [ ] Verify validation scripts run without errors

3. **Archive Old Scripts:**
   - [ ] Verify migration script exits safely (documentation-only)
   - [ ] Add note to scripts/README.md about historical artifacts

### Long-Term (Within 1 Month)

1. **Cleanup:**
   - [ ] Remove any legacy train_datasets/ references in experiments/
   - [ ] Clean up auto-generated Hydra outputs (outputs/)
   - [ ] Archive old hyperparameter sweep results

2. **Documentation Polish:**
   - [ ] Create migration guide for future dataset additions
   - [ ] Document new data/ directory structure in docs/
   - [ ] Add examples to CLAUDE.md for new paths

---

## Part 11: Open Questions for Senior Review

### ❓ Questions Requiring Approval

1. **Docker CI/CD Blocker:**
   - Current Docker workflows are failing, but Dockerfiles already reference `data/test/` correctly
   - Should we investigate root cause before Phase 2, or is it expected to resolve after Phase 2?

2. **Boughter Complexity:**
   - Phase 2 is 3-4x more complex than Phase 1 (263 references vs. 60-80)
   - Should we execute Phase 2 as single atomic commit, or break into smaller commits?
   - Recommendation: Single commit (reduces intermediate broken states)

3. **Preprocessing Scripts:**
   - 5 preprocessing scripts will need updates (27 references)
   - Should we test full preprocessing pipeline (DNA translation) post-migration?
   - Recommendation: Yes - run Stage 1-3 validation scripts

4. **Auto-Generated Outputs:**
   - ~150 references in outputs/novo_replication/.hydra/ (auto-generated)
   - Should we exclude outputs/ from migration entirely?
   - Recommendation: Yes - add outputs/ to .gitignore if not already

5. **Environment Variable Abstraction:**
   - Should we refactor to use $BOUGHTER_TRAIN_DIR env var (more robust)?
   - Or keep hardcoded paths (simpler, matches current pattern)?
   - Recommendation: Keep hardcoded for consistency with test datasets

### 🔍 Risks Flagged for Discussion

1. **Production Training Impact:**
   - Boughter is production training data (not test-only)
   - Breaking changes affect model reproducibility
   - Mitigation: Full test suite + training smoke test before commit

2. **Integration Test Dependency:**
   - test_boughter_embedding_compatibility.py verifies P0 blocker fix
   - Test expects all 16 fragment files in exact location
   - Mitigation: Run integration tests before and after migration

3. **Git History Preservation:**
   - Using `git mv` to preserve file provenance (same as Phase 1)
   - Confirm this is acceptable for 42 files
   - Mitigation: Verify `git log --follow` works for sample files

---

## Part 12: Senior Approval Checklist

Before executing Phase 2, confirm:

- [ ] **Scope approved:** Phase 1 CI/CD fixes + Phase 2 migration in single commit
- [ ] **Timeline approved:** 4-6 hours estimated execution time
- [ ] **Risk accepted:** 3-4x complexity vs. Phase 1, production training data affected
- [ ] **Rollback plan reviewed:** Git reset to HEAD~1 is acceptable rollback
- [ ] **Success criteria clear:** 0 train_datasets/ references in code, all tests pass
- [ ] **Open questions answered:** See Part 11

### Approval Signatures

**Submitted for review:** 2025-11-15
**Reviewed by:** _________________________
**Approved:** [ ] YES  [ ] NO  [ ] CONDITIONAL
**Conditions/Notes:** _________________________
**Date:** _________________________

---

## Part 13: Execution Log (Fill During Migration)

### Pre-Migration Checklist

- [ ] Created data/train/ directory
- [ ] Ran baseline tests (all passed)
- [ ] Attempted Docker builds (baseline)
- [ ] Backed up critical files (optional)

### Migration Steps Completed

| Step | Description | Status | Time | Notes |
|------|-------------|--------|------|-------|
| 2A.1 | Create migration script | | | |
| 2A.2 | Create data/train/ directory | | | |
| 2B.3 | git mv train_datasets/boughter data/train/boughter | | | |
| 2B.4 | Run migration script | | | |
| 2B.5 | Manual verification | | | |
| 2C.6 | Run test suite | | | |
| 2C.7 | Docker build verification | | | |
| 2C.8 | Training smoke test | | | |
| 2D.9 | Update documentation | | | |
| 2D.10 | Preserve migration script | | | |
| 2E.11 | Create commit | | | |
| 2E.12 | Push to remote | | | |

### Issues Encountered

(Document any unexpected issues and resolutions)

---

## Part 14: References

### Related Documents

- **REPOSITORY_CLEANUP_PLAN.md** - Master plan for all cleanup work
- **TEST_DATASETS_CONSOLIDATION_PLAN.md** - Phase 1 template (this document follows same structure)
- **CLAUDE.md** - Developer instructions (will be updated post-migration)

### Related Commits

- **Phase 1 Initial:** 288905c (test_datasets → data/test migration)
- **Phase 1 Residuals:** cea38e3 (fixed broken scripts, logs, docs)
- **Phase 1 Audit:** 204d6cd (fixed JSON paths, script safety, plan accuracy)
- **Phase 2:** TBD (this commit)

### Migration Scripts

- **Phase 1:** `scripts/migrate_test_datasets_to_data_test.sh` (documentation-only, exits immediately)
- **Phase 2:** `scripts/migrate_train_datasets_to_data_train.sh` (will be created, then made documentation-only)

### External Dependencies

- **ANARCI** (antibody numbering) - required to regenerate annotated/ if needed
- **ESM-1v** (HuggingFace) - required for training, pre-downloaded in Docker
- **Boughter et al. 2020** - raw data source (external, cannot regenerate)

---

## Appendix A: File Manifests

### Critical Files (Must Not Lose)

**Production Training File:**
```
data/train/boughter/canonical/VH_only_boughter_training.csv
├── 914 sequences (filtered subset)
├── 457 specific (label=0)
├── 457 non-specific (label=1)
└── Used by: default config, training CLI
```

**16 Annotated Fragment Files:**
```
data/train/boughter/annotated/
├── VH_only_boughter.csv (1,065 sequences)
├── VL_only_boughter.csv (1,065 sequences)
├── VH+VL_boughter.csv (1,065 sequences)
├── Full_boughter.csv (1,065 sequences)
├── H-CDR{1,2,3}_boughter.csv (1,065 each)
├── L-CDR{1,2,3}_boughter.csv (1,065 each)
├── {H,L,All}-CDRs_boughter.csv (1,065 each)
├── {H,L,All}-FWRs_boughter.csv (1,065 each)
└── Used by: multi-fragment experiments, integration tests
```

**Validation Artifacts:**
```
data/train/boughter/raw/translation_failures.log (Stage 1 failures)
data/train/boughter/annotated/annotation_failures.log (Stage 2 failures)
data/train/boughter/annotated/qc_filtered_sequences.txt (Stage 3 filtering)
data/train/boughter/annotated/validation_report.txt (Summary)
```

### Intermediate Files (Can Regenerate)

**Stage 1 Output:**
```
data/train/boughter/processed/boughter.csv (455 KB)
└── Can regenerate via: python preprocessing/boughter/stage1_dna_translation.py
```

### Raw Data (External Source - Cannot Regenerate)

**19 DNA FASTA Files (2.7 MB total):**
```
data/train/boughter/raw/
├── flu_fastaH.txt, flu_fastaL.txt, flu_NumReact.txt
├── nat_hiv_fastaH.txt, nat_hiv_fastaL.txt, nat_hiv_NumReact.txt
├── nat_cntrl_fastaH.txt, nat_cntrl_fastaL.txt, nat_cntrl_NumReact.txt
├── plos_hiv_fastaH.txt, plos_hiv_fastaL.txt, plos_hiv_YN.txt
├── gut_hiv_fastaH.txt, gut_hiv_fastaL.txt, gut_hiv_NumReact.txt
└── mouse_fastaH.dat, mouse_fastaL.dat, mouse_YN.txt
```

---

## Appendix B: Sed Patterns Used

### Primary Pattern (Automated Script)

```bash
sed -i '' 's|train_datasets/|data/train/|g' "$file"
```

**Matches:**
- `train_datasets/boughter/raw/`
- `./train_datasets/boughter/canonical/`
- `Path("train_datasets/boughter/annotated")`
- All literal string occurrences

**Excludes:**
- Plan documents (TRAIN_DATASETS_CONSOLIDATION_PLAN.md)
- Migration script itself
- .git/ directory
- outputs/ directory

### Manual Patterns (If Sed Fails)

**F-string in test_boughter_embedding_compatibility.py:**
```python
# Before
file_path = Path(f"train_datasets/boughter/annotated/{file_name}")

# After
file_path = Path(f"data/train/boughter/annotated/{file_name}")
```

**Path constant in default_paths.py:**
```python
# Before
BOUGHTER_ANNOTATED_DIR = Path("train_datasets/boughter/annotated")

# After
BOUGHTER_ANNOTATED_DIR = Path("data/train/boughter/annotated")
```

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-11-15 | 1.0 | Claude Code | Initial creation from exploration findings |
| | | | Incorporates CI/CD fixes and Phase 2 migration |
| | | | Awaiting senior approval before execution |

---

**END OF PLAN**

**Next Step:** Senior review and approval before executing migration.
