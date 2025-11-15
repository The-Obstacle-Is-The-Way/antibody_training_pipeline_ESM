# Data Directory Consolidation - Phase 1: Test Datasets

**Created:** 2025-11-14
**Revised:** 2025-11-14 (v2 - verified from first principles)
**Status:** Ready for execution
**Estimated Time:** 12-16 hours (Phase 1 only)
**Risk Level:** High (135 files require updates)
**Parent Plan:** See `REPOSITORY_CLEANUP_PLAN.md` (Problem 1, lines 29-238)

---

## Executive Summary

**Problem:** Dataset paths hardcoded across 250+ files with inconsistent naming (`train_datasets/`, `data/test/`). Creates friction for contributors, complicates Docker/CI, makes dataset additions error-prone.

**Solution:** Migrate `data/test/` → `data/test/` (Phase 1), then `train_datasets/` → `data/train/` (Phase 2).

**Why Phase 1 first?**
- Smaller scope: 135 files vs 250+ for full migration
- Lower risk: Test data changes don't affect production training
- Proves pattern: Validates migration before tackling `train_datasets/`
- Immediate CI/CD value: Standardized `data/test/` paths

**What this DOES:**

- ✅ Move `data/test/{harvey,jain,shehata}` → `data/test/{harvey,jain,shehata}`
- ✅ Update 135 files with path references (30 Python, 84 Markdown, 19 YAML, 2 other)
- ✅ Preserve existing structure (raw/, processed/, canonical/, fragments/)
- ✅ Maintain git history via `git mv`
- ✅ Verify all tests pass

**What this does NOT do:**

- ❌ Move training data (`train_datasets/` in Phase 2)
- ❌ Create duplicate canonical files (structure already correct)
- ❌ Change canonical/ architectural decisions (Nov 5, 2025)

---

## Alignment with SSOT

### From REPOSITORY_CLEANUP_PLAN.md

This plan implements **Phase 1 of Problem 1 (Data Directory Consolidation)**:
- Full target: `data/train/` + `data/test/`
- Full estimate: 13-15 hours
- **This plan:** Test datasets only (6-8 hours)

### Respecting Nov 5, 2025 Architectural Decisions

**Harvey `canonical/`:**
- **Decision:** Keep empty (logged in data/test/harvey/canonical/README.md:86-102)
- **Rationale:** Full 141k dataset already balanced, no subsampling needed
- **This plan:** Moves empty directory as-is ✅

**Shehata `canonical/`:**
- **Decision:** Intentionally empty (logged in data/test/shehata/canonical/README.md:1-3)
- **Rationale:** External test set, use fragments/ directly
- **This plan:** Moves empty directory as-is ✅

**Jain `canonical/`:**
- **Current:** Populated with 3 Novo parity files
- **This plan:** Moves all files as-is ✅

**Zero changes to canonical/ contents or decisions.** Just moving directories.

---

## Current State

```
data/test/
├── harvey/
│   ├── raw/         (3 CSVs, git-ignored)
│   ├── processed/   (harvey.csv)
│   ├── canonical/   (EMPTY + README - intentional per Nov 5 decision)
│   └── fragments/   (7 files: VHH_only_harvey.csv + CDRs)
├── jain/
│   ├── raw/         (5 Excel, git-ignored)
│   ├── processed/   (6 intermediate CSVs)
│   ├── canonical/   (3 Novo parity files)
│   └── fragments/   (17 files: VH_only_jain.csv + fragments)
└── shehata/
    ├── raw/         (4 Excel, git-ignored)
    ├── processed/   (shehata.csv)
    ├── canonical/   (EMPTY + README - intentional)
    └── fragments/   (16 files: VH_only_shehata.csv + fragments)
```

**References:** 135 unique files mention `data/test/` (verified 2025-11-14)

**Breakdown:**

- Python files: 30 (src/, tests/, preprocessing/, scripts/)
- Markdown files: 84 (docs/, experiments/, root)
- YAML files: 19 (configs/, experiments/)
- Other: 2

---

## Target State

```text
data/
└── test/
    ├── harvey/     (same 4 subdirs, moved via git mv)
    ├── jain/       (same 4 subdirs, moved via git mv)
    └── shehata/    (same 4 subdirs, moved via git mv)
```

---

## Execution Strategy

**Approach:** Automated migration script + manual verification

**Why not manual?** 135 files × 845 total references = too error-prone for manual edits.

**Script-based migration:**

```bash
# Create migration script
cat > /tmp/migrate_test_datasets.sh <<'EOF'
#!/bin/bash
set -euo pipefail

# Find all files with data/test/ references
FILES=$(grep -rl "data/test/" \
  --include="*.py" \
  --include="*.md" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="*.sh" \
  --include="Dockerfile*" \
  . 2>/dev/null)

# Update each file
for file in $FILES; do
  # Skip .git directory
  [[ "$file" =~ \.git/ ]] && continue

  # Replace data/test/ with data/test/
  sed -i '' 's|data/test/|data/test/|g' "$file"
  echo "Updated: $file"
done

echo "Migration complete. Updated $(echo "$FILES" | wc -l) files."
EOF

chmod +x /tmp/migrate_test_datasets.sh
```

**Execution steps:**

1. **Filesystem migration** (2 min)
   ```bash
   mkdir -p data/test
   git mv data/test/harvey data/test/
   git mv data/test/jain data/test/
   git mv data/test/shehata data/test/
   rmdir data/test/
   ```

2. **Dry-run verification** (5 min)
   ```bash
   # Preview changes without modifying files
   grep -rl "data/test/" --include="*.py" --include="*.md" --include="*.yaml" . | head -20
   # Verify these are expected files
   ```

3. **Run migration script** (2 min)
   ```bash
   /tmp/migrate_test_datasets.sh
   ```

4. **Verify no references remain** (2 min)
   ```bash
   grep -r "data/test/" --include="*.py" --include="*.md" --include="*.yaml" . 2>/dev/null || echo "✓ All references updated"
   ```

5. **Update `.gitignore`** (1 min)
   ```bash
   sed -i '' 's|data/test/|data/test/|g' .gitignore
   ```

6. **Full test suite** (10-15 min)
   ```bash
   uv run pytest -v
   make typecheck
   make lint
   ```

7. **Manual spot checks** (30 min)
   - Verify critical files: `default_paths.py`, configs, test files
   - Check Docker/CI configs
   - Review 10-15 random updated files

8. **Regenerate CURRENT_STRUCTURE.txt** (1 min)
   ```bash
   tree -a -I '.git|.mypy_cache|.uv_cache|.benchmarks|__pycache__|*.pyc|embeddings_cache|external_datasets' > CURRENT_STRUCTURE.txt
   ```

9. **Commit** (5 min)
   ```bash
   git add -A
   git commit -m "feat: Migrate test datasets to data/test/ (Phase 1)"
   ```

**Total time:** ~60 min execution + 11-15 hours buffer for fixes/verification

---

## Success Criteria

✅ `data/test/` exists with all datasets
✅ 135 files updated (0 data/test/ references remain)
✅ Full test suite passes (pytest + mypy + ruff)
✅ Git history preserved
✅ `.gitignore` works
✅ Zero functional changes

---

## Rollback

```bash
git revert HEAD
# OR:
git mv data/test/{harvey,jain,shehata} data/test/
rmdir data/test/ data/
```

---

## Future Work (Phase 2)

After Phase 1:
- Migrate `train_datasets/` → `data/train/` (~85 references, 7-9 hours)
- Apply same pattern, proven by Phase 1

---

## Summary of Corrections (v2)

**What was WRONG in v1:**

- ❌ Claimed "~60 files" need updates (actual: 135 files)
- ❌ Estimated 6-8 hours (actual: 12-16 hours needed)
- ❌ Risk marked "Medium" (actual: High - 135 files is major surgery)
- ❌ Referenced non-existent `TEST_DATASETS_CONSOLIDATION_PLAN_DETAILED.md`

**What is CORRECT now (v2):**

- ✅ Verified 135 unique files from first principles (30 Python, 84 Markdown, 19 YAML, 2 other)
- ✅ Realistic time estimate: 12-16 hours (60 min execution + 11-15 hour buffer)
- ✅ Risk properly assessed: High (automated script required, extensive verification)
- ✅ Complete execution plan in this file (no external dependencies)
- ✅ Automated migration script to handle 845 total path references
- ✅ Preserves all Nov 5, 2025 canonical/ architectural decisions
- ✅ Aligns with REPOSITORY_CLEANUP_PLAN.md

**Blockers removed:**

- ✅ Accurate file count (verified via grep)
- ✅ No missing reference files
- ✅ Complete execution strategy in single document

---

**Ready for execution.** All blockers killed. Plan verified from first principles.
