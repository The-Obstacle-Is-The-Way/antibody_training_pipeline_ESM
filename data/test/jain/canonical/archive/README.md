# Archived Jain Canonical Files

This directory contains deprecated Jain dataset files that have been superseded by newer versions.

## Files

### VH_only_jain_test_PARITY_86.csv

**Status:** ⚠️ DEPRECATED (2025-11-18)
**Replaced by:** `../VH_only_jain_86_p5e_s2.csv`

**Why deprecated:**
- Column name `sequence` is incompatible with `JainDataset.load_data()`
- New file uses correct column name: `vh_sequence`
- Data is identical, only column name differs

**Last used:** 2025-11-06 (in experiment logs)
**Superseded:** 2025-11-15 (when new file created)
**Archived:** 2025-11-18

**Migration path:**
- Code references updated to point to new file
- Error messages updated to recommend new file
- Deprecation header added before archiving

**If you need this file:**
Use `../VH_only_jain_86_p5e_s2.csv` instead. If you absolutely must use the old column name, you can restore from git history (commit < 2025-11-18) but this is not recommended.

---

**Archive Policy:**
Files are kept here for historical reference and backwards compatibility during transition periods. After 6 months with no usage, they may be permanently removed (with git history preservation).
