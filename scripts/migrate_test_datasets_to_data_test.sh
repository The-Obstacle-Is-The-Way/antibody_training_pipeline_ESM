#!/bin/bash
#
# Migration Script: test_datasets/ → data/test/
#
# Part of Phase 1 Data Consolidation (REPOSITORY_CLEANUP_PLAN.md)
# Executed: 2025-11-15 (commit 288905c)
#
# This script was used to migrate all test_datasets/ path references to data/test/
# after the filesystem migration was complete. Preserved here for:
# 1. Documentation of exact migration process
# 2. Reference for Phase 2 (train_datasets/ → data/train/)
# 3. Rollback support if needed
#
# Usage (historical):
#   chmod +x scripts/migrate_test_datasets_to_data_test.sh
#   ./scripts/migrate_test_datasets_to_data_test.sh
#
# WARNING: This script has already been run. Re-running will have no effect
# (all test_datasets/ references have been migrated to data/test/).

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ Test Datasets Migration Script                                            ║"
echo "║ Phase 1: test_datasets/ → data/test/                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if migration is needed
REMAINING=$(grep -rl "test_datasets/" \
  --include="*.py" \
  --include="*.md" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="*.sh" \
  --include="Dockerfile*" \
  . 2>/dev/null | grep -v "^./.git/" | wc -l | tr -d ' ')

if [ "$REMAINING" -eq 0 ]; then
  echo "✅ Migration already complete!"
  echo "   No test_datasets/ references found in code."
  echo ""
  echo "   Status: Phase 1 ✅ (commit 288905c)"
  echo "   Next: Phase 2 (train_datasets/ → data/train/)"
  exit 0
fi

echo "⚠️  Found $REMAINING files with test_datasets/ references"
echo ""
echo "Files to update:"
echo ""

# Find all files with test_datasets/ references
FILES=$(grep -rl "test_datasets/" \
  --include="*.py" \
  --include="*.md" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="*.sh" \
  --include="Dockerfile*" \
  . 2>/dev/null | grep -v "^./.git/")

# Show files (first 20)
echo "$FILES" | head -20
if [ "$(echo "$FILES" | wc -l | tr -d ' ')" -gt 20 ]; then
  echo "... and $(($(echo "$FILES" | wc -l | tr -d ' ') - 20)) more"
fi
echo ""

# Confirm
read -p "Proceed with migration? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "❌ Migration cancelled"
  exit 1
fi

echo ""
echo "🔄 Migrating files..."
echo ""

# Update each file
COUNT=0
for file in $FILES; do
  # Replace test_datasets/ with data/test/
  sed -i '' 's|test_datasets/|data/test/|g' "$file"
  echo "  ✓ Updated: $file"
  ((COUNT++))
done

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ Migration Complete!                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Files updated: $COUNT"
echo ""
echo "Next steps:"
echo "  1. Verify: grep -r 'test_datasets/' --include='*.py' --include='*.md' ."
echo "  2. Test: make test && make typecheck && make lint"
echo "  3. Commit: git add -A && git commit -m 'fix: Update test_datasets/ → data/test/ references'"
echo ""
