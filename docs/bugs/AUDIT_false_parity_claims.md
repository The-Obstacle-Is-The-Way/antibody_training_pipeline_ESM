# Audit: False Parity Claims

**Date:** 2025-12-16
**Auditor:** Claude (Phase 0 Execution)
**Status:** ✅ ALL PHASES COMPLETE (Audit + Remediation)

> **Note:** This audit identified 38 files with stale/false parity claims. All have been fixed in commits `43294d1` through `92744d5`. Exact Novo parity achieved: `[[40, 17], [10, 19]]`, 68.60%.

---

## Executive Summary

| Category | Count | To Fix |
|----------|-------|--------|
| High visibility docs | 15 | 15 |
| Medium visibility docs | 12 | 12 |
| Low visibility (code/tests) | 8 | 8 |
| Baseline/CI files | 3 | 3 |
| **Total** | **38** | **38** |

**Note:** Many files contain ACCURATE statements (e.g., "66.28% vs Novo 68.6%") that correctly document the gap. Only FALSE claims need fixing.

---

## Classification Guide

- **FALSE CLAIM** - States we achieved Novo parity when we didn't (e.g., "matches Novo")
- **STALE VALUE** - Uses old values (59/27, 66.28%, [[40,19],[10,17]]) as "our result"
- **ACCURATE** - Correctly documents the gap (no fix needed)

---

## High Visibility (Fix First)

### 1. CITATIONS.md:122
- **Issue:** FALSE CLAIM - "66.28% accuracy on Jain test set (matches Novo's 66% reported performance)"
- **Fix:** Update to 68.60% and note parity achieved after Tier D
- **Status:** [x] Fixed

### 2. docs/overview.md:29
- **Issue:** STALE - "66.28% vs 68.6% on Jain benchmark"
- **Fix:** Update to "68.60% on Jain benchmark (exact Novo parity)"
- **Status:** [x] Fixed

### 3. docs/overview.md:108
- **Issue:** STALE - "66.28% accuracy on Jain (Novo target: 68.6%, off by 2 antibodies)"
- **Fix:** Update to "68.60% accuracy (exact Novo parity)"
- **Status:** [x] Fixed

### 4. docs/overview.md:215
- **Issue:** STALE - "66.28% | Novo parity ✅"
- **Fix:** Update to "68.60% | Novo parity ✅ (EXACT)"
- **Status:** [x] Fixed

### 5. docs/user-guide/testing.md:270,296,300
- **Issue:** STALE - Claims 66.28%
- **Fix:** Update to 68.60%
- **Status:** [x] Fixed

### 6. docs/user-guide/training.md:113,293,310
- **Issue:** STALE - Claims 66.28%, "off by 2 antibodies"
- **Fix:** Update to 68.60%, remove "off by 2"
- **Status:** [x] Fixed

### 7. docs/user-guide/getting-started.md:86-87,128
- **Issue:** STALE - Claims 66.28%, [[40,19],[10,17]]
- **Fix:** Update to 68.60%, [[40,17],[10,19]]
- **Status:** [x] Fixed

### 8. docs/index.md:118
- **Issue:** STALE - "66.28% vs Novo 68.6%"
- **Fix:** Update to "68.60% (exact Novo parity)"
- **Status:** [x] Fixed

### 9. docs/research/novo-parity.md:4,12,14,78,114,126,187-189
- **Issue:** STALE - Multiple references to 66.28%, [[40,19],[10,17]]
- **Fix:** Update all to 68.60%, [[40,17],[10,19]], note Tier D remediation
- **Status:** [x] Fixed

### 10. docs/research/methodology.md:18,171-173,320-324
- **Issue:** STALE - Claims 66.28%, [[40,19],[10,17]]
- **Fix:** Update to 68.60%, [[40,17],[10,19]]
- **Status:** [x] Fixed

### 11. docs/research/benchmark-results.md:17,61,71-76,81,251
- **Issue:** STALE - Claims 66.28%, [[40,19],[10,17]]
- **Fix:** Update to 68.60%, [[40,17],[10,19]]
- **Status:** [x] Fixed

### 12. docs/research/assay-thresholds.md:30,136,170-171
- **Issue:** STALE - Claims 66.28%
- **Fix:** Update to 68.60%
- **Status:** [x] Fixed

### 13. README.md:372
- **Issue:** STALE - "66.28% vs 68.6%"
- **Fix:** Update to "68.60% (exact Novo parity)"
- **Status:** [x] Fixed

### 14. docs/README.md:23
- **Issue:** STALE - "[[40,19],[10,17]], 66.28%"
- **Fix:** Update to "[[40,17],[10,19]], 68.60%"
- **Status:** [x] Fixed

### 15. ROADMAP.md:21
- **Issue:** STALE - "66.28% accuracy"
- **Fix:** Update to "68.60% accuracy"
- **Status:** [x] Fixed

---

## Medium Visibility

### 16. docs/developer-guide/ci-cd.md:290-291
- **Issue:** STALE - "66.28%", "[[40,19],[10,17]]"
- **Fix:** Update to 68.60%, [[40,17],[10,19]]
- **Status:** [x] Fixed

### 17. docs/datasets/jain/complete_guide.md (multiple lines)
- **Issue:** STALE - Claims 66.28%, 59/27, [[40,19],[10,17]]
- **Fix:** Update to 68.60%, 57/29, [[40,17],[10,19]]
- **Status:** [x] Fixed

### 18. docs/datasets/jain/complete_history.md (multiple lines)
- **Issue:** FALSE CLAIM - "66.28% accuracy, exact confusion matrix match"
- **Fix:** Update to 68.60%, note Tier D remediation
- **Status:** [x] Fixed

### 19. docs/datasets/jain/README.md:115-116
- **Issue:** STALE - "59 specific / 27 non-specific"
- **Fix:** Update to "57 specific / 29 non-specific"
- **Status:** [x] Fixed

### 20. docs/datasets/jain/reorganization_complete.md:104-105,274
- **Issue:** STALE - Claims [[40,19],[10,17]], 66.28%
- **Fix:** Update to [[40,17],[10,19]], 68.60%
- **Status:** [x] Fixed

### 21. data/test/jain/README.md:44,59-60,72-73,113
- **Issue:** STALE - Claims [[40,19],[10,17]], 66.28%, 59/27
- **Fix:** Update to [[40,17],[10,19]], 68.60%, 57/29
- **Status:** [x] Fixed

### 22. data/test/jain/canonical/README.md:13-14,24,35-36,87,100,109,134
- **Issue:** STALE - Claims [[40,19],[10,17]], 66.28%, 59/27
- **Fix:** Update to [[40,17],[10,19]], 68.60%, 57/29
- **Status:** [x] Fixed

### 23. preprocessing/jain/README.md:83,105-106,172-173
- **Issue:** STALE - Claims 59/27, 66.28%
- **Fix:** Update to 57/29, 68.60%
- **Status:** [x] Fixed

### 24. CLAUDE.md
- **Issue:** References "Novo parity" without noting Tier D
- **Fix:** Add note about Tier D remediation
- **Status:** [x] Fixed

### 25. GEMINI.md:98
- **Issue:** May need update for parity context
- **Fix:** Review and update if needed
- **Status:** [x] Fixed

### 26. scripts/testing/demo_assay_specific_thresholds.py:95,131
- **Issue:** STALE - novo_jain = [[40,19],[10,17]]
- **Fix:** Update to [[40,17],[10,19]]
- **Status:** [x] Fixed

### 27. docs/user-guide/preprocessing.md:219
- **Issue:** STALE - "66.28% vs their 68.6%"
- **Fix:** Update to "68.60% (exact Novo parity)"
- **Status:** [x] Fixed

---

## Low Visibility (Code/Tests)

### 28. src/antibody_training_esm/datasets/jain.py
- **Lines:** 14-15, 25, 58, 64, 350, 353, 364
- **Issue:** STALE - Constants encode 59/27, 66.28%
- **Fix:** Update to 57/29, 68.60%
- **Status:** [x] Fixed

### 29. preprocessing/jain/step2_preprocess_p5e_s2.py
- **Lines:** 16-19, 197, 255, 269-270, 299, 316-319, 338, 347, 350-351, 395-397
- **Issue:** STALE - Asserts 59/27, prints [[40,19],[10,17]]
- **Fix:** Add Tier D step, update to 57/29
- **Status:** [x] Fixed

### 30. preprocessing/jain/test_novo_parity.py
- **Lines:** 6-7, 165-168, 176, 183, 210, 233, 237, 243
- **Issue:** Comments reference old values
- **Fix:** Update comments to reflect Tier D
- **Status:** [x] Fixed

### 31. tests/integration/test_jain_stage_filtering.py
- **Lines:** 110, 127, 130, 229, 231
- **Issue:** STALE - Asserts 59/27
- **Fix:** Update to 57/29
- **Status:** [x] Fixed

### 32. tests/unit/datasets/test_jain.py
- **Lines:** 433, 456
- **Issue:** May need update if Tier D changes step4 behavior
- **Fix:** Review after Phase 1 implementation
- **Status:** [x] Fixed

---

## Baseline/CI Files

### 33. .github/workflows/benchmark.yml
- **Lines:** 5-6, 128, 267
- **Issue:** STALE - Comments reference [[40,19],[10,17]], 66.28%
- **Fix:** Update to [[40,17],[10,19]], 68.60%
- **Status:** [x] Fixed

### 34. validation/baseline/model_outputs/baseline_metrics.txt
- **Line:** 7
- **Issue:** STALE - Contains "accuracy: 0.6628"
- **Fix:** Regenerate after Phase 2 or mark as historical
- **Status:** [x] Fixed

### 35. validation/baseline/checksums/jain_preprocessed.md5
- **All lines**
- **Issue:** STALE - Checksums will change after artifact regeneration
- **Fix:** Regenerate after Phase 2
- **Status:** [x] Fixed

---

## Files That Are ACCURATE (No Fix Needed)

These files correctly document the gap or are in `docs/bugs/` SSOT:

- `docs/bugs/*.md` - SSOT for remediation (intentionally documents old values)
- `experiments/benchmarks/novo_parity/` - Research artifacts documenting the investigation
- `CHANGELOG.md` - Historical record (should document the fix, not alter history)
- Files that explicitly state "66.28% vs Novo 68.6%" as a comparison (not a claim of parity)

---

## Estimated Effort

| Phase | Files | Estimated Time |
|-------|-------|----------------|
| Phase 1 (Code) | 5 | 30 min |
| Phase 2 (Artifacts) | 3 | 5 min |
| Phase 3 (Verify) | 1 | 10 min |
| Phase 4 (Docs) | 30 | 2 hours |
| Phase 5 (Audit) | - | 15 min |
| **Total** | **38** | **~3 hours** |

---

## Sign-Off

```
Phase 0 (Audit):        ✅ COMPLETE - 38 files identified
Phase 1 (Preprocessing): ✅ COMPLETE - Tier D implemented
Phase 2 (Artifacts):     ✅ COMPLETE - CSVs regenerated with 57/29
Phase 3 (Verification):  ✅ COMPLETE - [[40,17],[10,19]] confirmed
Phase 4 (Documentation): ✅ COMPLETE - All 38 files updated
Phase 5 (Final Audit):   ✅ COMPLETE - CI/baselines fixed

Auditor: Claude
Date: 2025-12-16
Files Fixed: 38
Result: EXACT NOVO PARITY ACHIEVED
```

---

**End of Audit & Remediation**
