# Data Directory Consolidation - Phase 1: Test Datasets

**Created:** 2025-11-14
**Revised:** 2025-11-15 (v3 - post-migration documentation)
**Status:** ✅ COMPLETED
**Execution Time:** ~90 minutes
**Commit:** 288905c

---

## Executive Summary

**Problem:** Dataset paths were hardcoded across 250+ files with inconsistent naming (`train_datasets/`, `test_datasets/`). This created friction for contributors, complicated Docker/CI, and made dataset additions error-prone.

**Solution:** Migrated `test_datasets/` → `data/test/` (Phase 1 complete), then `train_datasets/` → `data/train/` (Phase 2 pending).

**Why Phase 1 first?**
- Smaller scope: 135 files vs 250+ for full migration
- Lower risk: Test data changes don't affect production training
- Proves pattern: Validates migration before tackling `train_datasets/`
- Immediate CI/CD value: Standardized `data/test/` paths

---

## What Was Done (Completed 2025-11-15)

### Filesystem Migration ✅
- Created `data/test/` directory
- Moved 70 files via `git mv` preserving history:
  - `test_datasets/harvey/` → `data/test/harvey/`
  - `test_datasets/jain/` → `data/test/jain/`
  - `test_datasets/shehata/` → `data/test/shehata/`
- Removed empty `test_datasets/` directory

### Automated Path Updates ✅
- Updated 135 unique files across codebase:
  - Python: 30 files (src/, tests/, preprocessing/, scripts/)
  - Markdown: 84 files (docs/, experiments/, root)
  - YAML: 19 files (configs/, experiments/)
  - Other: 2 files
- Fixed 845 total path references
- Updated `.gitignore` patterns
- Updated logs/metadata: 8 additional files

### Quality Verification ✅
- All 133 unit tests pass
- Type checking clean (mypy: 88 files)
- Linting clean (ruff)
- Pre-commit hooks pass
- Git history fully preserved (`git log --follow` verified)

### Documentation ✅
- Regenerated `CURRENT_STRUCTURE.txt` (50,695 lines)
- Updated `CLAUDE.md`, `REPOSITORY_CLEANUP_PLAN.md`
- Created this plan document for future reference

---

## Migration Details

### Pre-Migration State

```text
test_datasets/
├── harvey/
│   ├── raw/              (3 CSVs, git-ignored)
│   ├── processed/        (1 CSV: harvey.csv)
│   ├── canonical/        (EMPTY - intentional, see README)
│   └── fragments/        (7 files: VHH_only_harvey.csv + 6 CDR fragments)
├── jain/
│   ├── raw/              (5 Excel files, git-ignored)
│   ├── processed/        (6 intermediate CSVs)
│   ├── canonical/        (3 parity files for Novo benchmark)
│   └── fragments/        (17 files: VH_only_jain.csv + 16 fragments)
└── shehata/
    ├── raw/              (4 Excel files, git-ignored)
    ├── processed/        (1 CSV: shehata.csv)
    ├── canonical/        (EMPTY - intentional, see README)
    └── fragments/        (16 files: VH_only_shehata.csv + 15 fragments)
```

### Post-Migration State (Current)

```text
data/
└── test/
    ├── harvey/     (same 4 subdirs, moved via git mv)
    ├── jain/       (same 4 subdirs, moved via git mv)
    └── shehata/    (same 4 subdirs, moved via git mv)
```

### Preserved Architectural Decisions

**Harvey `canonical/`:**
- Decision (Nov 5, 2025): Keep empty
- Rationale: Full 141k dataset already balanced, no subsampling needed
- Migration: Moved empty directory as-is ✅

**Shehata `canonical/`:**
- Decision: Intentionally empty
- Rationale: External test set, use `fragments/` directly
- Migration: Moved empty directory as-is ✅

**Jain `canonical/`:**
- Current: Populated with 3 Novo parity files
- Migration: Moved all files as-is ✅

---

## Execution Strategy (Used)

### Migration Script

Created `/tmp/migrate_test_datasets.sh`:

```bash
#!/bin/bash
set -euo pipefail

# Find all files with test_datasets/ references
FILES=$(grep -rl "test_datasets/" \
  --include="*.py" \
  --include="*.md" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="*.sh" \
  --include="Dockerfile*" \
  . 2>/dev/null | grep -v "^./.git/")

# Update each file
for file in $FILES; do
  sed -i '' 's|test_datasets/|data/test/|g' "$file"
  echo "Updated: $file"
done

echo ""
echo "Migration complete. Updated $(echo "$FILES" | wc -l | tr -d ' ') files."
```

### Execution Steps (Completed)

1. **Filesystem migration** (2 min)
   ```bash
   mkdir -p data/test
   git mv test_datasets/harvey data/test/
   git mv test_datasets/jain data/test/
   git mv test_datasets/shehata data/test/
   rmdir test_datasets/
   ```

2. **Run migration script** (2 min)
   ```bash
   chmod +x /tmp/migrate_test_datasets.sh
   /tmp/migrate_test_datasets.sh
   ```

3. **Verify no references remain** (2 min)
   ```bash
   grep -r "test_datasets/" --include="*.py" --include="*.md" --include="*.yaml" . 2>/dev/null
   # Result: 0 references in code (verified post-execution)
   ```

4. **Update `.gitignore`** (1 min)
   ```bash
   sed -i '' 's|test_datasets/|data/test/|g' .gitignore
   ```

5. **Full test suite** (15 min)
   ```bash
   uv run pytest -v
   make typecheck
   make lint
   ```

6. **Manual spot checks** (5 min)
   - Verified: `default_paths.py`, configs, test files
   - Checked: Docker/CI configs
   - Reviewed: 10-15 random updated files

7. **Regenerate `CURRENT_STRUCTURE.txt`** (1 min)

8. **Commit** (2 min)
   ```bash
   git add -A
   git commit -m "feat: Migrate test datasets to data/test/ directory (Phase 1)"
   ```

---

## Success Criteria (All Met ✅)

- ✅ `data/test/` exists with all datasets
- ✅ 135 files updated (0 `test_datasets/` references in code)
- ✅ Full test suite passes (133 tests, mypy clean, ruff clean)
- ✅ Git history preserved
- ✅ `.gitignore` works
- ✅ Zero functional changes

---

## Rollback Instructions

```bash
# If needed, rollback via:
git revert 288905c

# OR manually:
git mv data/test/{harvey,jain,shehata} test_datasets/
rmdir data/test/ data/
# Then revert code changes via git checkout
```

---

## Future Work (Phase 2)

**Next:** Migrate `train_datasets/` → `data/train/` (~85 references, 7-9 hours)

Apply same proven pattern:
1. Create `data/train/`
2. Move datasets via `git mv`
3. Update code references
4. Verify tests pass
5. Commit

---

## Alignment with SSOT

This plan implements **Phase 1 of Problem 1 (Data Directory Consolidation)** from `REPOSITORY_CLEANUP_PLAN.md`:
- Full target: `data/train/` + `data/test/`
- Full estimate: 13-15 hours
- **Phase 1:** Test datasets (actual: 90 minutes)
- **Phase 2:** Training datasets (pending)

---

**Status:** ✅ Phase 1 complete. Zero legacy `test_datasets/` references remain. Ready for Phase 2 when needed.
