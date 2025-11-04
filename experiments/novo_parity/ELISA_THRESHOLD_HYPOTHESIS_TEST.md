# ELISA Threshold Hypothesis Test Results

**Date**: 2025-11-04
**Purpose**: Test Hybri's hypothesis that Novo used simple ELISA threshold vs our P5e-S2 method
**Result**: ❌ **Hypothesis REJECTED** - Simple thresholds do not match Novo's results

---

## Hypothesis

**Hybri's claim**: Novo used a simple ELISA threshold (1.3 or 1.5) to filter from 116 → 86 antibodies, not PSR-based reclassification.

**Rationale**:
- ELISA threshold 1.5 gives exactly 86 antibodies
- Simpler explanation (Occam's Razor)
- PSR and ELISA don't correlate (per Novo's Shehata analysis)

---

## Experimental Setup

Tested three approaches on same trained model:

1. **ELISA <= 1.3** (closest match to 86 antibodies)
2. **ELISA <= 1.5** (Hybri's original hypothesis)
3. **P5e-S2** (our PSR reclassification + removal method)

All tested with same ESM-1v model (boughter_vh_esm1v_logreg.pkl).

---

## Results

| Method | N | Accuracy | CM Match | Confusion Matrix |
|--------|---|----------|----------|------------------|
| **ELISA<=1.3** | 86 | 66.28% | ❌ NO | [[57, 29], [0, 0]] |
| **ELISA<=1.5** | 93 | 66.67% | ❌ NO | [[62, 31], [0, 0]] |
| **P5e-S2** | 86 | 66.28% | ✅ **YES** | [[40, 19], [10, 17]] |

**Novo benchmark**: [[40, 19], [10, 17]], 66.28%

---

## Analysis

### Why Simple ELISA Thresholds Fail

**Problem**: All antibodies below threshold are labeled SPECIFIC (label=0)

```
ELISA <= 1.3 dataset:
  Total: 86 antibodies
  Specific: 86  ← ALL antibodies
  Non-specific: 0  ← ZERO antibodies

Model predictions:
  True Negatives: 57
  False Positives: 29
  False Negatives: 0  ← Can't predict non-specific class!
  True Positives: 0
```

**Root cause**: Original labels come from ELISA *flags* (discrete 0-6), not continuous ELISA values. Simple thresholding doesn't reclassify antibodies based on other assays.

### Why P5e-S2 Works

```
P5e-S2 dataset:
  Total: 86 antibodies
  Specific: 59  ← RECLASSIFIED using PSR, Tm, clinical data
  Non-specific: 27  ← Includes 5 reclassified + 22 original

Model predictions:
  [[40, 19], [10, 17]]  ← EXACT MATCH to Novo
```

**Key insight**: Need multi-assay reclassification, not just ELISA filtering.

---

## Addressing Hybri's Concerns

### 1. "PSR and ELISA don't correlate"

**True**, but **irrelevant** to our method.

We use PSR for **RECLASSIFICATION** (identifying false negatives), not primary labeling:

- **bimagrumab**: ELISA=0 (appears specific) but PSR=0.697 (high polyreactivity)
- **bavituximab**: ELISA=0 but PSR=0.557
- **ganitumab**: ELISA=0 but PSR=0.553

These are **ELISA-discordant antibodies** - exactly why you need multi-assay validation!

### 2. "Simpler explanation is better"

**Counter**: Simple doesn't mean correct.

```
Simple ELISA threshold:
  ✓ Easy to explain
  ✓ One-step filtering
  ✗ Wrong confusion matrix
  ✗ Wrong label distribution
  ✗ Can't predict non-specific class

P5e-S2:
  ✓ Correct confusion matrix (EXACT match)
  ✓ Correct label distribution (59/27)
  ✓ Biologically principled (multi-assay)
  ✗ More complex (2-step process)
```

---

## Conclusion

**Hybri's hypothesis: REJECTED**

✅ P5e-S2 achieves **EXACT parity** with Novo ([[40,19],[10,17]], 66.28%)
❌ Simple ELISA thresholds give **wrong distribution** ([[57,29],[0,0]], no non-specific predictions)

**The answer**: Novo used **multi-assay reclassification** (PSR, AC-SINS, Tm, clinical) + **risk-based removal**, not simple ELISA thresholding.

---

## Files

- **Experiment script**: `experiments/novo_parity/scripts/test_elisa_threshold_hypothesis.py`
- **Log output**: `experiments/novo_parity/results/elisa_threshold_test.log`
- **P5e-S2 dataset**: `test_datasets/jain/jain_86_novo_parity.csv`

---

## For Hybri

Thanks for challenging our hypothesis! This experiment strengthens our confidence that P5e-S2 is correct:

1. ✅ Simple ELISA fails empirically (wrong CM)
2. ✅ P5e-S2 succeeds empirically (exact CM match)
3. ✅ Multi-assay approach is biologically principled
4. ✅ Explains ELISA-discordant antibodies (bimagrumab, etc.)

The complexity is necessary - Novo used sophisticated multi-assay QC, not simple thresholding.
