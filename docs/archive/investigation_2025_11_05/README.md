# Jain Dataset Investigation - Nov 5, 2025

This archive contains documentation from the investigation into Jain dataset methodologies and Novo parity verification.

## Conclusion

**Both methodologies achieve Novo parity:**
- OLD reverse-engineered (simple QC): [[40, 19], [10, 17]] ✅
- P5e-S2 canonical (PSR-based): [[40, 19], [10, 17]] ✅

**Recommendation:** Use OLD method for deterministic benchmarking, P5e-S2 for biophysical research.

See `JAIN_COMPLETE_GUIDE.md` in the repo root for the consolidated, accurate guide.

---

## Archived Files

### Investigation Documents (Nov 5, 2025)

- **BOUGHTER_JAIN_FINAL_RESOLUTION.md**
  - Combined Boughter verification + Jain analysis
  - Shows both datasets are correct

- **COMPLETE_JAIN_MODEL_RESOLUTION.md**
  - Model + dataset 2x2 matrix testing
  - Identified which combinations achieve parity

- **DATASETS_FINAL_SUMMARY.md**
  - Overview of all datasets
  - Comparison of methodologies

- **FINAL_RESOLUTION_AND_PATH_FORWARD.md**
  - Action plan after investigation
  - Recommendations for cleanup

- **JAIN_DATASETS_AUDIT_REPORT.md**
  - Detailed audit of all Jain CSV files
  - File counts and locations

- **JAIN_DATASET_COMPLETE_HISTORY.md**
  - Full historical evolution of Jain datasets
  - Timeline of changes

- **JAIN_EXPERIMENTS_DISCREPANCY.md** ⚠️ **OBSOLETE**
  - Claimed P5e-S2 was off by 1
  - This was due to file corruption (fixed at 10:49 AM Nov 5)
  - P5e-S2 now gives correct result

- **JAIN_MODELS_DATASETS_COMPLETE_ANALYSIS.md**
  - Comprehensive model/dataset testing
  - Confusion matrix comparisons

---

## Why Archived (Not Deleted)

These documents show the investigation process and were accurate at the time of writing. They're kept for:
- Historical record
- Understanding the evolution of understanding
- Reference if questions arise later

However, **they should NOT be used** as current documentation. Use `JAIN_COMPLETE_GUIDE.md` instead.

---

## Key Findings from Investigation

1. ✅ Boughter pipeline is 100% correct (1,171 → 914)
2. ✅ OLD model trained correctly on 914 sequences
3. ✅ OLD Jain dataset achieves exact Novo parity
4. ✅ P5e-S2 also achieves parity (file was fixed)
5. ⚠️ P5e-S2 has one borderline antibody (nimotuzumab ~0.5 prob)
6. 📊 72 Jain CSV files total (needs cleanup!)

---

**Date archived:** 2025-11-05
**Status:** Historical reference only
**Current docs:** See repo root

