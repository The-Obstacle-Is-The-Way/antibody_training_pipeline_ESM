#!/bin/bash
#
# Migration Script: train_datasets/ → data/train/
#
# Part of Phase 2 Data Consolidation (REPOSITORY_CLEANUP_PLAN.md)
# Executed: 2025-11-15 (commit TBD)
#
# This script was used to migrate all train_datasets/ path references to data/train/
# after the filesystem migration was complete. Preserved here for:
# 1. Documentation of exact migration process
# 2. Rollback support if needed
# 3. Reference for future migrations
#
# ⚠️  WARNING: DO NOT RE-RUN THIS SCRIPT ⚠️
#
# This script has already been executed and all train_datasets/ references
# have been migrated to data/train/. Re-running would corrupt plan documents
# that legitimately reference train_datasets/ for historical context.
#
# This file is preserved as DOCUMENTATION ONLY.
#

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           MIGRATION COMPLETE                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  This script was already executed on 2025-11-15."
echo ""
echo "   Phase 2 Status: ✅ COMPLETE"
echo "   - train_datasets/ → data/train/ (42 files moved, 19 code files updated)"
echo "   - Zero train_datasets/ references remain in production code"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This file is preserved for DOCUMENTATION purposes only."
echo "DO NOT re-run this script - it would corrupt plan documents."
echo ""
exit 0

# ============================================================================
# HISTORICAL CODE (NO LONGER EXECUTED)
# ============================================================================
#
# The code below documents the exact migration process used for Phase 2.
# It is NOT executed due to the exit 0 above.
#

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    PHASE 2: TRAIN DATASETS MIGRATION                       ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if migration is needed (CODE PATHS ONLY, EXCLUDE DOCS)
echo "📊 Scanning for train_datasets/ references in code..."
echo ""

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
  grep -v "TEST_DATASETS_CONSOLIDATION_PLAN.md" | \
  grep -v "scripts/migrate_train_datasets_to_data_train.sh" | \
  grep -v "scripts/migrate_test_datasets_to_data_test.sh" | \
  wc -l | tr -d ' ')

if [ "$REMAINING" -eq 0 ]; then
  echo "✅ Migration already complete!"
  echo ""
  echo "   No train_datasets/ references found in production code."
  echo "   Phase 2 migration successful."
  echo ""
  exit 0
fi

echo "Found $REMAINING files with train_datasets/ references"
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
  grep -v "TEST_DATASETS_CONSOLIDATION_PLAN.md" | \
  grep -v "scripts/migrate_train_datasets_to_data_train.sh" | \
  grep -v "scripts/migrate_test_datasets_to_data_test.sh")

# Update each file
echo "🔧 Updating code references..."
echo ""

for file in $FILES; do
  sed -i '' 's|train_datasets/|data/train/|g' "$file"
  echo "  ✓ Updated: $file"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Migration complete!"
echo ""
echo "Next steps:"
echo "  1. grep -r 'train_datasets/' --include='*.py' --include='*.yaml' src/ preprocessing/ tests/"
echo "  2. uv run pytest tests/unit/ -v                 # Verify tests pass"
echo "  3. docker build -f Dockerfile.dev .             # Verify Docker builds"
echo "  4. git add -A && git commit                     # Commit changes"
echo ""
