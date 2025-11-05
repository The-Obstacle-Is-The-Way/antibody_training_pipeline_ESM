# Jain Experiments Discrepancy Report

**Date:** 2025-11-05 09:30:00
**Status:** 🚨 **CRITICAL - EXPERIMENTS JSON DOES NOT MATCH ACTUAL TEST**

---

## TL;DR

**The experiments folder CLAIMS P5e-S2 achieves [[40, 19], [10, 17]], but when tested TODAY with the SAME model (OLD Nov 2), it gives [[39, 20], [10, 17]] (off by 1).**

**The OLD reverse-engineered dataset achieves exact match, NOT P5e-S2.**

---

## What the Experiments Claim

**File:** `experiments/novo_parity/results/permutations/P5e_S2_final_audit.json`
```json
{
  "permutation_id": "P5e-S2",
  "confusion_matrix": [[40, 19], [10, 17]],
  "accuracy": 0.6627906976744186,
  "exact_match_to_novo": true,
  "timestamp": "2025-11-03T23:33:17.127730"
}
```

**Documentation:** `experiments/novo_parity/EXACT_MATCH_FOUND.md`
> Winner #1: P5e-S2 (eldelumab + PSR/AC-SINS)
> **Result**: [[40, 19], [10, 17]] ✅ **EXACT MATCH**

---

## What Actually Happens When Tested

**Test Date:** 2025-11-05 08:29
**Model:** `models/boughter_vh_esm1v_logreg.pkl` (Nov 2, 914 training)
**Dataset:** `experiments/novo_parity/datasets/jain_86_p5e_s2.csv`

**Result:**
```
Confusion Matrix: [[39, 20], [10, 17]]
TN=39, FP=20, FN=10, TP=17
Accuracy: 65.12% (56/86)
```

**❌ OFF BY 1:** TN=39 vs 40, FP=20 vs 19

---

## Confirmed Test Results (Today)

| Model | Dataset | CM | Accuracy | Match Novo? |
|-------|---------|---|----------|-------------|
| OLD (914) | OLD reverse-eng | [[40,19],[10,17]] | 66.28% | ✅ YES |
| OLD (914) | P5e-S2 | [[39,20],[10,17]] | 65.12% | ❌ No (off by 1) |
| OLD (914) | P5e (no tiebreak) | [[39,20],[10,17]] | 65.12% | ❌ No (off by 2 in JSON) |

---

## Investigation: What's Different?

### Antibody Composition Differs

**P5e vs P5e_S2:** 4 antibodies different
- In P5e, NOT in P5e_S2: drozitumab, ocrelizumab, seribantumab, urelumab
- In P5e_S2, NOT in P5e: bapineuzumab, bevacizumab, efalizumab, ramucirumab

**P5e vs P5e_S4:** 3 antibodies different
**P5e_S2 vs P5e_S4:** 3 antibodies different

**The tie-breaking algorithm changed which antibodies got removed!**

---

## Possible Explanations

### Hypothesis 1: JSON is Wrong (Manual Edit)
- Someone manually edited the JSON to say [[40, 19], [10, 17]]
- The actual test never gave that result
- JSON timestamp: Nov 3, 23:33
- Files created: Nov 4, 19:23 (13 hours later)

### Hypothesis 2: Dataset Changed
- P5e-S2 was tested on Nov 3 and gave correct result
- Dataset file was regenerated on Nov 4 and changed slightly
- New version gives different result

### Hypothesis 3: Different Model Used
- Experiments used a different model (not Nov 2 OLD model)
- Model file doesn't exist in experiments folder
- Scripts reference `models/boughter_vh_esm1v_logreg.pkl` but maybe it was different on Nov 3?

### Hypothesis 4: Code Change
- The experiment scripts computed confusion matrix differently
- Or used different preprocessing
- But this seems unlikely (scripts look straightforward)

---

## Evidence Analysis

### For Hypothesis 1 (Manual Edit):
- ❌ JSON has detailed provenance (removed_ids, reclassified_ids)
- ❌ Multiple JSON files (P5e, P5e_S2, P5e_S4) all look auto-generated
- ✅ But easy to manually edit after the fact

### For Hypothesis 2 (Dataset Changed):
- ❌ Git shows files created Nov 4, never modified since
- ❌ No evidence of regeneration
- ✅ But possible if regenerated before commit

### For Hypothesis 3 (Different Model):
- ✅ No model files in experiments folder
- ✅ Model could have been different on Nov 3
- ❌ But scripts clearly reference `models/boughter_vh_esm1v_logreg.pkl`
- ❌ Git log shows that model was created Nov 2, before experiments

### For Hypothesis 4 (Code Change):
- ❌ Scripts are straightforward, use sklearn.metrics
- ❌ No evidence of different preprocessing

---

## Most Likely Explanation

**Hypothesis 1 + 2 Combined:**

1. Initial experiments (Nov 3) tested various permutations
2. P5e gave [[39, 20], [10, 17]] (documented in `targeted/P5e_result.json`)
3. They realized tie-breaking was needed
4. Created P5e_S2 (AC-SINS) and P5e_S4 (Tm) with DIFFERENT antibody sets
5. **ONE of them actually gave [[40, 19], [10, 17]]** but it wasn't P5e-S2
6. Documentation/JSON got mixed up or was aspirational

**OR:** The OLD reverse-engineered method was ALWAYS the answer, and the P5e experiments were just attempts that never actually succeeded.

---

## What We Know For Sure

1. ✅ **OLD reverse-engineered dataset gives EXACT match** [[40, 19], [10, 17]]
2. ✅ **P5e-S2 does NOT give exact match** (tested today: [[39, 20], [10, 17]])
3. ✅ **Experiments documentation CLAIMS P5e-S2 works** (but test says otherwise)
4. ✅ **OLD method is simpler** (no PSR tie-breaking, just length + borderline removals)
5. ✅ **Occam's Razor suggests OLD method is correct**

---

## User Was Right

**User's intuition:**
> "I don't think Novo did the whack-ass PSR experimentation shit"
> "The correct one should be in here... we should have clearly documented how we achieved the correct one"

**Verdict:** ✅ **CORRECT**

The OLD reverse-engineered method (simple QC) achieves parity, NOT the complex PSR-based P5e-S2.

---

## Recommendation

### For Benchmarking:
```
Model: models/boughter_vh_esm1v_logreg.pkl
Dataset: test_datasets/jain/VH_only_jain_test_PARITY_86.csv (OLD)
Expected: [[40, 19], [10, 17]], 66.28%
```

### For Documentation:
1. Add disclaimer to `experiments/novo_parity/EXACT_MATCH_FOUND.md`:
   ```
   ⚠️ WARNING: P5e-S2 claim requires verification.
   When tested on 2025-11-05 with Nov 2 model, P5e-S2 gave [[39, 20], [10, 17]] (off by 1).
   The OLD reverse-engineered method achieves exact parity.
   ```

2. Update `JAIN_DATASET_COMPLETE_HISTORY.md` to mark:
   - OLD reverse-engineered: ✅ **ACHIEVES NOVO PARITY**
   - P5e-S2: 📊 **BIOLOGICALLY PRINCIPLED (off by 1 in testing)**

---

## Next Steps

1. Test P5e_S4 to see if IT gives [[40, 19], [10, 17]]
2. Check if any OTHER P5 variant (P5f, P5g, P5h) gives exact match
3. Accept that OLD method is the answer (Occam's Razor)
4. Update docs to reflect reality

---

**Generated:** 2025-11-05 09:30:00
**Status:** 🚨 DISCREPANCY DOCUMENTED
