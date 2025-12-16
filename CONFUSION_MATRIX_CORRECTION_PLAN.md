# Confusion Matrix Documentation Correction Plan

**Created:** 2025-12-15
**Issue:** #33 - Jain parity documentation claims EXACT MATCH but is wrong
**Branch:** `investigate/jain-parity-verification`
**Senior Review:** COMPLETED - Plan verified and updated

---

## THE PROBLEM

Documentation throughout the codebase claims "EXACT MATCH" to Novo Nordisk's confusion matrix, but:
1. The **Novo target numbers are wrong** in most docs
2. The **"EXACT MATCH" claims are false** - we're off by 2 antibodies

### Source of Truth (Novo Nordisk Figure S14A)

| Metric | Novo (CORRECT) | Our Docs (WRONG) | Our Actual Result |
|--------|----------------|------------------|-------------------|
| Confusion Matrix | `[[40, 17], [10, 19]]` | Claims Novo is `[[40, 19], [10, 17]]` | `[[40, 19], [10, 17]]` |
| TN | 40 | 40 | 40 |
| FP | **17** | Claims Novo is **19** | **19** |
| FN | 10 | 10 | 10 |
| TP | **19** | Claims Novo is **17** | **17** |
| Specific | **57** | Claims Novo is **59** | **59** |
| Non-specific | **29** | Claims Novo is **27** | **27** |
| Accuracy | **68.6%** (59/86) | Claims Novo is **66.28%** | **66.28%** (57/86) |

**Key distinction:**
- Our result `[[40, 19], [10, 17]]` with 66.28% accuracy is CORRECT (that's what we actually get)
- The ERROR is claiming this matches Novo, when Novo's actual is `[[40, 17], [10, 19]]` with 68.6%

---

## WHAT NEEDS TO CHANGE

### CHANGE (Claims about Novo's target):
- `[[40, 19], [10, 17]]` as "Novo's matrix" → `[[40, 17], [10, 19]]`
- `66.28%` as "Novo's accuracy" → `68.6%`
- `59/27` as "Novo's split" → `57/29`
- `57/86 = 66.28%` as "Novo's calculation" → `59/86 = 68.6%`

### CHANGE (False claims):
- "EXACT MATCH" → "close match (off by 2 antibodies)"
- "Cell-for-cell identical" → REMOVE
- "exact parity" → "approximate parity"

### DO NOT CHANGE (Our actual results):
- Files reporting OUR result as `[[40, 19], [10, 17]]` (that's accurate)
- Files reporting OUR accuracy as 66.28% (that's what we achieved)
- `validation/baseline/model_outputs/baseline_metrics.txt` (actual measurements)
- Prediction CSV files in `experiments/benchmarks/` (actual data)

---

## FILES THAT ALREADY HAVE CORRECT NOVO MATRIX

These files already correctly state Novo's matrix - DO NOT CLOBBER:
- `docs/research/assay-thresholds.md:173` - "Novo benchmark: [[40, 17], [10, 19]] - 68.6%"
- `docs/research/assay-thresholds.md:224` - "Jain optimal: 0.467 (to match Novo [[40, 17], [10, 19]])"
- `docs/research/assay-thresholds.md:314` - "Novo benchmark: [[40, 17], [10, 19]] (68.6%)"
- `docs/research/benchmark-results.md:95` - "[[40, 17], [10, 19]]"

---

## NAMING ISSUE (Note for future fix)

**`data/test/jain/canonical/jain_86_novo_parity.csv`** is named "novo_parity" but contains:
- 59 specific / 27 non-specific (OUR split)
- NOT Novo's actual 57/29 split

This is a data artifact issue, not a documentation issue. Note for future: this file name is misleading.

---

## CLUSTERED FILE LIST

### CLUSTER 1: Source Code (CRITICAL - affects runtime behavior)

| File | Lines | What's Wrong | Fix |
|------|-------|--------------|-----|
| `src/antibody_training_esm/datasets/jain.py` | 14 | Claims Novo is `[[40, 19], [10, 17]]` | Change to `[[40, 17], [10, 19]]` + note it's target |
| `src/antibody_training_esm/datasets/jain.py` | 56 | Comment claims Novo benchmark | Update comment |
| `src/antibody_training_esm/datasets/jain.py` | 57-61 | Constants claim Novo's values | These are OUR values - rename/clarify |
| `src/antibody_training_esm/models/artifact.py` | 285, 290 | Example uses our values as "expected" | Clarify these are OUR results |

**Priority: HIGH**

---

### CLUSTER 2: Preprocessing Scripts (CRITICAL)

| File | Lines | What's Wrong | Fix |
|------|-------|--------------|-----|
| `preprocessing/jain/step2_preprocess_p5e_s2.py` | 18 | "EXACT MATCH" claim | Remove false claim |
| `preprocessing/jain/step2_preprocess_p5e_s2.py` | 347, 393 | Prints wrong Novo matrix | Update to correct Novo target |
| `preprocessing/jain/test_novo_parity.py` | 6-7 | Claims exact match | Update to "approximate" |
| `preprocessing/jain/test_novo_parity.py` | 168 | `novo_cm = [[40, 19], [10, 17]]` | Change to correct Novo: `[[40, 17], [10, 19]]` |
| `preprocessing/jain/test_novo_parity.py` | 206, 209 | Compares to wrong target | Update comparison |
| `preprocessing/jain/README.md` | 105, 172 | "EXACT MATCH" claims | Update |

**Priority: HIGH**

---

### CLUSTER 3: CI/CD (CRITICAL)

| File | Lines | What's Wrong |
|------|-------|--------------|
| `.github/workflows/benchmark.yml` | 126 | Claims match to `[[40, 19], [10, 17]]` |
| `.github/workflows/benchmark.yml` | 265 | Reports wrong Novo accuracy |

**Priority: HIGH**

---

### CLUSTER 4: Core Research Docs

| File | Lines | Notes |
|------|-------|-------|
| `docs/README.md` | 23 | "EXACT MATCH" claim |
| `docs/overview.md` | 29, 108, 201, 202, 215 | Multiple wrong claims |
| `docs/research/novo-parity.md` | 4, 12, 14, 78, 114, 126, 187, 188, 302 | **PRIMARY PARITY DOC** - many fixes needed |
| `docs/research/methodology.md` | 18, 171, 173, 320, 324 | Wrong Novo numbers |
| `docs/research/assay-thresholds.md` | 30, 136, 170, 171, 230 | Mixed - some correct, some wrong |
| `docs/research/benchmark-results.md` | 17, 71, 81, 251 | Mixed - line 95 is correct |

**Priority: HIGH**

---

### CLUSTER 5: Jain Dataset Docs

| File | Key Lines | Notes |
|------|-----------|-------|
| `docs/datasets/jain/README.md` | 116 | "EXACT MATCH" claim |
| `docs/datasets/jain/complete_guide.md` | 36, 96-99, 158, 193, 208, 234, 259, 277, 290-291, 359, 367 | **MAIN GUIDE** - many fixes |
| `docs/datasets/jain/complete_history.md` | 48, 307-308, 324, 355, 362-363, 414, 422, 441, 454, 487, 528 | Historical doc |
| `docs/datasets/jain/reorganization_complete.md` | 104-105, 274 | Wrong claims |
| `docs/datasets/jain/vh_benchmark_implementation_plan.md` | 256-257, 284-285, 331-332 | Wrong expected values |

**Priority: HIGH**

---

### CLUSTER 6: User/Developer Guides

| File | Lines |
|------|-------|
| `docs/user-guide/getting-started.md` | 86-87, 128 |
| `docs/user-guide/training.md` | 113, 293-294, 310 |
| `docs/user-guide/testing.md` | 270, 296, 300 |
| `docs/developer-guide/ci-cd.md` | 290-291 |

**Priority: MEDIUM**

---

### CLUSTER 7: Data READMEs

| File | Lines |
|------|-------|
| `data/test/jain/README.md` | 44, 59, 72, 113 |
| `data/test/jain/canonical/README.md` | 13, 35, 87, 100, 109, 134 |
| `data/train/boughter/README.md` | 33, 130 |

**Priority: MEDIUM**

---

### CLUSTER 8: Root Documentation

| File | Lines | Notes |
|------|-------|-------|
| `CHANGELOG.md` | 812 | Historical - just note wrong |
| `CITATIONS.md` | ~18, ~122 | References |
| `ROADMAP.md` | ~21 | Achievement claim |

**Priority: MEDIUM**

---

### CLUSTER 9: Implementation Docs

| File | Lines |
|------|-------|
| `docs/implementation/CODE_QUALITY_FIX_PLAN_2025-11-22.md` | 155, 160, 185 |
| `docs/implementation/PYDANTIC_PHASE_4_ARTIFACTS_METRICS.md` | 283, 288 |
| `docs/implementation/VALIDATION_PLAN.md` | 226, 242, 320, 410, 435-437, 477, 507 |

**Priority: LOW**

---

### CLUSTER 10: Scripts

| File | Lines |
|------|-------|
| `scripts/testing/demo_assay_specific_thresholds.py` | 95, 131 |

**Priority: MEDIUM**

---

### CLUSTER 11: Archive Docs (Low Priority)

| File | Key Lines |
|------|-----------|
| `docs/datasets/jain/archive/README.md` | 29 |
| `docs/datasets/jain/archive/JAIN_CANONICAL_INVESTIGATION.md` | 35, 112, 152 |
| `docs/datasets/jain/archive/JAIN_PIPELINE_EXPLAINED.md` | 50, 112, 299 |
| `docs/datasets/jain/archive/JAIN_QC_REMOVALS_COMPLETE.md` | 5-6, 156-157 |
| `docs/archive/audits/2025-11-05-scripts-audit.md` | 80, 149 |
| `docs/archive/plans/*` | Various |
| `docs/archive/investigations/*` | Various |

**Priority: LOW** - Archive files

---

### CLUSTER 12: Other Dataset Docs

| File | Lines |
|------|-------|
| `docs/datasets/boughter/README.md` | 186 |
| `docs/datasets/boughter/complete_history.md` | 14, 282, 425, 471 |
| `docs/datasets/boughter/cdr_boundary_first_principles_audit.md` | 528 |
| `docs/datasets/boughter/novo_methodology_clarification.md` | 263 |
| `docs/datasets/harvey/harvey_test_results.md` | 163 |
| `experiments/README.md` | 36 |

**Priority: LOW**

---

### CLUSTER 13: Misc Docs

| File | Lines |
|------|-------|
| `docs/ESM1V_ENSEMBLING_INVESTIGATION.md` | 116 |
| `docs/research/model-zoo-roadmap.md` | 202 |

**Priority: LOW**

---

## FILES TO LEAVE ALONE (Actual Measurements)

These files contain actual measured values, NOT documentation claims:
- `validation/baseline/model_outputs/baseline_metrics.txt` - Our actual metrics
- `experiments/benchmarks/**/*.csv` - Prediction outputs (0.6628 is probability, not accuracy reference)

**CRITICAL RULE:** Never apply numeric replacements inside:
- `data/**/*.csv` - Raw/processed data files
- `experiments/benchmarks/**/*.csv` - Prediction outputs
- `experiments/**/*.yaml` - Benchmark results (actual measurements)
- Any file containing actual model outputs

Only modify **documentation** (*.md), **source code** (*.py), and **CI config** (*.yml in .github/).

---

## REPLACEMENT PATTERNS

### Pattern 1: Confusion Matrix (when claiming it's Novo's)
```
WRONG:  [[40, 19], [10, 17]] (as Novo's target)
RIGHT:  [[40, 17], [10, 19]] (Novo's actual)
```

### Pattern 2: Accuracy (when claiming it's Novo's)
```
WRONG:  66.28% (as Novo's accuracy)
RIGHT:  68.6% (Novo's actual accuracy)
NOTE:   66.28% is OUR result, which is correct when reporting our performance
```

### Pattern 3: Label Distribution (when claiming it's Novo's)
```
WRONG:  59 specific / 27 non-specific (as Novo's)
RIGHT:  57 specific / 29 non-specific (Novo's actual)
NOTE:   59/27 is OUR data's distribution
```

### Pattern 4: Correct Predictions (when claiming it's Novo's)
```
WRONG:  57/86 = 66.28% (claiming this is Novo's)
RIGHT:  59/86 = 68.6% (Novo's actual)
NOTE:   57/86 is OUR correct prediction count
```

### Pattern 5: Claims to Remove/Modify
```
REMOVE: "EXACT MATCH" / "exact match"
REMOVE: "Cell-for-cell identical"
REMOVE: "EXACT Novo parity" / "exact parity"
REPLACE WITH: "close to Novo parity (off by 2 antibodies in label distribution)"
             or "approximate parity"
```

---

## EXECUTION PLAN

### Phase 1: Critical Source Code
- [ ] `src/antibody_training_esm/datasets/jain.py`
- [ ] `src/antibody_training_esm/models/artifact.py`

### Phase 2: Critical Preprocessing
- [ ] `preprocessing/jain/step2_preprocess_p5e_s2.py`
- [ ] `preprocessing/jain/test_novo_parity.py`
- [ ] `preprocessing/jain/README.md`

### Phase 3: CI/CD
- [ ] `.github/workflows/benchmark.yml`

### Phase 4: Core Research Docs
- [ ] `docs/research/novo-parity.md` (PRIMARY)
- [ ] `docs/research/methodology.md`
- [ ] `docs/research/benchmark-results.md`
- [ ] `docs/research/assay-thresholds.md` (careful - some lines correct)
- [ ] `docs/overview.md`
- [ ] `docs/README.md`

### Phase 5: Jain Dataset Docs
- [ ] `docs/datasets/jain/complete_guide.md`
- [ ] `docs/datasets/jain/README.md`
- [ ] `docs/datasets/jain/complete_history.md`
- [ ] `docs/datasets/jain/reorganization_complete.md`
- [ ] `docs/datasets/jain/vh_benchmark_implementation_plan.md`

### Phase 6: User/Developer Guides
- [ ] `docs/user-guide/*`
- [ ] `docs/developer-guide/ci-cd.md`

### Phase 7: Data READMEs
- [ ] `data/test/jain/README.md`
- [ ] `data/test/jain/canonical/README.md`
- [ ] `data/train/boughter/README.md`

### Phase 8: Root Documentation
- [ ] `CHANGELOG.md`
- [ ] `CITATIONS.md`
- [ ] `ROADMAP.md`

### Phase 9: Scripts
- [ ] `scripts/testing/demo_assay_specific_thresholds.py`

### Phase 10: Implementation Docs
- [ ] `docs/implementation/*`

### Phase 11: Other Dataset Docs
- [ ] `docs/datasets/boughter/*`
- [ ] `docs/datasets/harvey/*`
- [ ] `experiments/README.md`

### Phase 12: Archive Docs
- [ ] `docs/datasets/jain/archive/*`
- [ ] `docs/archive/*`

### Phase 13: Misc
- [ ] `docs/ESM1V_ENSEMBLING_INVESTIGATION.md`
- [ ] `docs/research/model-zoo-roadmap.md`

---

## VERIFICATION CHECKLIST

After all fixes, run these commands from repo root:

### 1. No misattribution of our matrix as Novo's
```bash
# Should return ZERO matches (our matrix claimed as Novo's)
grep -rE "\[\[40,?\s*19\],?\s*\[10,?\s*17\]\].*[Nn]ovo|[Nn]ovo.*\[\[40,?\s*19\],?\s*\[10,?\s*17\]\]" \
  --include="*.md" --include="*.py" --include="*.yml" --include="*.yaml" \
  . 2>/dev/null | grep -v "^Binary"
```

### 2. No "EXACT MATCH" claims remain
```bash
# Should return ZERO matches (except possibly CHANGELOG.md historical entry)
grep -rE "EXACT MATCH|exact match|Cell-for-cell identical" \
  --include="*.md" --include="*.py" --include="*.yml" \
  . 2>/dev/null | grep -v CHANGELOG.md | grep -v CONFUSION_MATRIX_CORRECTION_PLAN
```

### 3. No 66.28% claimed as Novo's accuracy
```bash
# Should return ZERO matches
grep -rE "66\.28.*[Nn]ovo|[Nn]ovo.*66\.28" \
  --include="*.md" --include="*.py" --include="*.yml" \
  . 2>/dev/null
```

### 4. CI/CD files updated (include .github/)
```bash
# Verify benchmark.yml has correct Novo target (68.6%, [[40,17],[10,19]])
grep -E "\[\[40,?\s*17\],?\s*\[10,?\s*19\]\]|68\.6" .github/workflows/benchmark.yml
```

### 5. Correct Novo matrix NOT clobbered
```bash
# These files should STILL have [[40, 17], [10, 19]] (Novo's actual)
grep -l "\[\[40,\s*17\],\s*\[10,\s*19\]\]" docs/research/assay-thresholds.md docs/research/benchmark-results.md
```

### 6. Data files untouched
```bash
# Verify no unintended changes to data CSVs
git diff --stat data/**/*.csv experiments/**/*.csv experiments/**/*.yaml 2>/dev/null
# Should show no changes (or only intentional ones)
```

---

## SENIOR REVIEW NOTES

### Review 1 (2025-12-15):
1. ✅ Novo S14A confirmed: `[[40, 17], [10, 19]]`, 68.6% accuracy
2. ✅ Our data confirmed: 59/27 split in `jain_86_novo_parity.csv`
3. ✅ Plan patterns verified correct
4. ✅ Additional patterns added: `57/86`, `0.6628`
5. ✅ Files with correct Novo matrix identified (avoid clobbering)
6. ⚠️ Note: CSV file named "novo_parity" has OUR split, not Novo's (naming issue for future)

### Review 2 (2025-12-15) - Spec Hardening:
7. ✅ Verification checklist expanded to include `*.yml/*.yaml` (catches CI files)
8. ✅ Added CRITICAL RULE: Never modify data/**/*.csv or experiments/**/*.yaml
9. ✅ Added misattribution check: `[[40,19].*Novo` and `Novo.*[[40,19]]` patterns
10. ✅ Verification commands now include `.github/` hidden directory
11. ✅ Added git diff check to verify data files untouched

**SPEC STATUS: IRONCLAD** - Ready for execution
