# Jain Parity Reverse Engineering — Research Spec

**Status:** OPEN (Research Required)
**Priority:** P1 (High) — Core benchmark produces incorrect results
**GitHub Issue:** [#33](https://github.com/Clarity-Digital-Twin/antibody_training_pipeline_ESM/issues/33)
**Created:** 2025-12-15
**Last Updated:** 2025-12-15

---

## Executive Summary

Our reverse-engineered P5e-S2 preprocessing pipeline produces **59 specific / 27 non-specific** antibodies, but Novo Nordisk's Figure S14A shows **57 specific / 29 non-specific**. We are **off by 2 antibodies** in the label distribution, resulting in:

| Metric | Ours (P5e-S2) | Novo (Figure S14A) | Delta |
|--------|---------------|---------------------|-------|
| Confusion Matrix | `[[40, 19], [10, 17]]` | `[[40, 17], [10, 19]]` | FP/TP differ by 2 |
| Accuracy | 66.28% | 68.6% | -2.32pp |
| Specific (label=0) | 59 | 57 | +2 |
| Non-specific (label=1) | 27 | 29 | -2 |
| TN | 40 | 40 | ✅ Match |
| FN | 10 | 10 | ✅ Match |

**Key Insight:** TN=40 and FN=10 match exactly. The discrepancy is entirely in FP/TP, meaning we have 2 antibodies that we classify as specific that Novo classifies as non-specific.

---

## Source of Truth: Novo Nordisk Paper

### What Novo Documents (Sakhnini et al. 2025)

From the paper's Methods section (Table 2):
- **Jain Dataset Size:** 137 clinical-stage IgG1-formatted antibodies
- **Label Policy:**
  - 0 flags = specific (label 0)
  - 1-3 flags = mildly non-specific (excluded)
  - >3 flags = non-specific (label 1)
- **Assay:** ELISA with panel of 6 ligands (ssDNA, dsDNA, insulin, LPS, cardiolipin, KLH)

From Figure S13:
- X-axis shows "ELISA flag" (singular) with range 0-6
- This confirms ELISA-only flags, NOT the total 10-flag system

From Figure S14A (ESM-1v VH-based LogisticReg on Jain):
- **Confusion Matrix:** `[[40, 17], [10, 19]]`
- **Total:** 86 antibodies (derived: 40+17 = 57 specific, 10+19 = 29 non-specific)
- **Accuracy:** 59/86 = 68.6%

### What Novo Does NOT Document

**THE GAP:** Novo does not document how they go from ~116 antibodies (after ELISA 1-3 filtering) to 86 antibodies. This is the step we are reverse-engineering.

---

## Our Current Pipeline (P5e-S2 Method)

```
137 antibodies (jain_with_private_elisa_FULL.csv)
    ↓ Step 1: Remove ELISA 1-3 (mild aggregators)
116 antibodies (94 specific + 22 non-specific)
    ↓ Step 2: Reclassify 5 specific → non-specific
    │   • Tier A (PSR >0.4): bimagrumab, bavituximab, ganitumab
    │   • Tier B (Tm <60°C): eldelumab
    │   • Tier C (Clinical): infliximab (61% ADA)
89 specific + 27 non-specific = 116 total
    ↓ Step 3: Remove 30 specific by PSR/AC-SINS ranking
59 specific + 27 non-specific = 86 total ← OUR RESULT

EXPECTED: 57 specific + 29 non-specific = 86 total ← NOVO TARGET
```

**Files:**
- Implementation: `preprocessing/jain/step2_preprocess_p5e_s2.py`
- Output: `data/test/jain/canonical/jain_86_novo_parity.csv`
- VH-only: `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv`

---

## The Research Question

**How do we close the 2-antibody gap?**

The discrepancy could come from:
1. **Missing reclassification:** We need to reclassify 2 more specific → non-specific
2. **Wrong removal strategy:** Different criteria for which 30 to remove
3. **Different threshold:** PSR threshold might be different from 0.4
4. **Alternative QC step:** Novo uses a QC filter we haven't identified

**Critical Constraint:** Any solution must be **biologically principled** — something Novo would do **blindly** (without knowing the confusion matrix outcome). Cherry-picking based on results would be scientifically inappropriate.

---

## Candidate Hypotheses

### Hypothesis 1: Additional PSR-Based Reclassification

**Rationale:** If Novo used a slightly different PSR threshold, more antibodies would be reclassified.

**Candidates to investigate:**
- Antibodies with PSR between 0.35-0.45 (near our 0.4 threshold)
- Antibodies with moderate PSR (0.2-0.4) combined with other risk factors

**Experiment:**
1. List all specific antibodies with PSR between 0.2-0.5
2. Try reclassifying combinations of 2 additional antibodies
3. Check which combinations produce [[40, 17], [10, 19]]

### Hypothesis 2: Different Removal Strategy

**Rationale:** Instead of PSR/AC-SINS ranking, Novo might use a different metric.

**Alternative strategies to test:**
- HIC retention time ranking
- Fab Tm ranking
- Combined score (PSR + AC-SINS + HIC + Tm)
- Biophysical descriptor-based ranking (pI, bulkiness, etc.)

**Experiment:**
1. For each strategy, rank specific antibodies
2. Remove top 30 (or top N for various N)
3. Check which produces 57/29 split

### Hypothesis 3: Additional Biological QC Filter

**Rationale:** Novo may exclude antibodies based on biological criteria we haven't identified.

**Candidate filters:**
- **Sequence-based:** Unusual CDR3 length, unusual germline usage
- **Structural:** VH length outliers (z-score >2), framework anomalies
- **Clinical:** Development status, immunogenicity reports, withdrawal status
- **Expression:** Known expression issues, aggregation tendency

**Experiment:**
1. Identify outliers in each category
2. Test if removing/reclassifying outliers produces correct split

### Hypothesis 4: Different ELISA Threshold

**Rationale:** The ">3 flags" threshold might be interpreted differently.

**Variations to test:**
- "≥3 flags" instead of ">3 flags" (include 3-flag antibodies as non-specific)
- "≥4 flags" with different mild range

**Experiment:**
1. Recalculate labels with different thresholds
2. Check if any produces 57/29 after similar filtering

### Hypothesis 5: Label Swap (Not Filtering)

**Rationale:** Instead of removing antibodies, Novo might have swapped labels for 2 specific.

**Why this is concerning:** This would only make sense if there's biological justification.

**Experiment:**
1. Identify the 2 specific antibodies that, when flipped to non-specific, produce [[40, 17], [10, 19]]
2. Check if these antibodies have any documented issues (aggregation, immunogenicity, etc.)

---

## Experimental Protocol

### Phase 1: Data Preparation

1. **Export all 116 antibodies** with full metadata:
   - ELISA flags (0-6)
   - PSR score (0-1)
   - AC-SINS (nm)
   - HIC retention time (min)
   - Fab Tm (°C)
   - VH length
   - VH sequence
   - Current label (our classification)
   - Original label (from Jain paper)

2. **Identify boundary antibodies:**
   - Specific antibodies with high PSR (>0.2)
   - Specific antibodies with model prediction probability >0.4
   - Specific antibodies with any biophysical risk factor

### Phase 2: Systematic Permutation Testing

**Goal:** Find which 2 antibodies, when reclassified specific → non-specific, produce [[40, 17], [10, 19]]

**Approach:**
```python
from itertools import combinations

# Get list of specific antibodies in our 86-antibody set
specific_antibodies = df[df['label'] == 0]['id'].tolist()  # 59 antibodies

# Try all pairs
for ab1, ab2 in combinations(specific_antibodies, 2):
    # Create modified dataset
    df_modified = df.copy()
    df_modified.loc[df_modified['id'].isin([ab1, ab2]), 'label'] = 1

    # Run inference
    y_pred = model.predict(X_modified)
    cm = confusion_matrix(df_modified['label'], y_pred)

    # Check if matches Novo target
    if np.array_equal(cm, [[40, 17], [10, 19]]):
        print(f"MATCH: {ab1}, {ab2}")
```

**Combinatorics:** C(59, 2) = 1,711 combinations — computationally feasible.

### Phase 3: Biological Validation

For each candidate pair found in Phase 2:
1. **Literature review:** Any documented issues with these antibodies?
2. **Biophysical analysis:** Do they have unusual PSR, Tm, HIC, or AC-SINS?
3. **Developmental status:** Withdrawn, failed trials, or known problems?
4. **Sequence analysis:** CDR3 anomalies, germline issues?

**Key question:** Would Novo have identified these 2 as problematic **without** knowing the confusion matrix outcome?

### Phase 4: Alternative Strategy Testing

If Phase 2 doesn't yield biologically plausible results:

1. **Test different removal strategies:**
   - For each of 5+ ranking strategies (PSR, HIC, Tm, combined, etc.)
   - Remove top 28, 29, 30, 31, 32 specific antibodies
   - Check which produces 57/29 split

2. **Test reclassification threshold variations:**
   - PSR > 0.35, 0.38, 0.42, 0.45
   - Combined score thresholds

3. **Test biological QC filters:**
   - VH length z-score >1.5, >2.0, >2.5
   - Fab Tm <60°C, <62°C, <65°C
   - Clinical status filters

---

## Data Files Required

| File | Location | Description |
|------|----------|-------------|
| jain_with_private_elisa_FULL.csv | data/test/jain/processed/ | 137 antibodies with ELISA data |
| jain_ELISA_ONLY_116.csv | data/test/jain/processed/ | 116 antibodies (SSOT) |
| jain_sd03.csv | data/test/jain/processed/ | Biophysical data (PSR, AC-SINS, HIC, Tm) |
| jain_86_novo_parity.csv | data/test/jain/canonical/ | Our 86-antibody output |
| VH_only_jain_86_p5e_s2.csv | data/test/jain/canonical/ | VH sequences for inference |

---

## Success Criteria

### Primary Goal
Find a preprocessing methodology that:
1. Produces exactly **57 specific / 29 non-specific = 86 total**
2. Results in confusion matrix **[[40, 17], [10, 19]]** with ESM-1v VH LogReg model
3. Is **biologically principled** (something Novo would do blindly)

### Secondary Goals
1. Document the exact QC step(s) we were missing
2. Update `step2_preprocess_p5e_s2.py` with the correct methodology
3. Update all documentation to reflect actual vs target parity
4. Create regression tests to prevent future drift

---

## Known Constraints

### Model Reproducibility
- ESM-1v embeddings have slight nondeterminism (~0.1% variance)
- One antibody (nimotuzumab) is borderline (~0.5 probability)
- Results may vary by ±1 TN/FP due to embedding variance

### Biological Plausibility
- Solutions must be "blinded" — no cherry-picking based on results
- Any reclassification must have biological justification
- Clinical evidence (ADA, withdrawal, trial failure) is acceptable
- Biophysical outliers (extreme PSR, Tm, etc.) are acceptable

---

## Experimental Code Location

Experiments should be implemented in:
- `experiments/benchmarks/novo_parity/scripts/` (new Python files)
- Results in `experiments/benchmarks/novo_parity/results/`
- Datasets in `experiments/benchmarks/novo_parity/datasets/`

**Suggested files:**
- `phase2_permutation_search.py` — Systematic pair testing
- `phase3_biological_validation.py` — Metadata analysis
- `phase4_alternative_strategies.py` — Alternative ranking tests

---

## References

- **Novo Paper:** Sakhnini et al. 2025, bioRxiv, DOI: 10.1101/2025.04.28.650927
- **Figure S14A:** `literature/markdown/novo_2025_supplementary/novo-media-1/_page_17_Figure_0.jpeg`
- **Jain 2017 Paper:** PNAS 114(5), 944-949
- **Current implementation:** `preprocessing/jain/step2_preprocess_p5e_s2.py`
- **GitHub Issue:** #33

---

## Appendix A: The 5 Currently Reclassified Antibodies

| Antibody | Tier | Reason | PSR | Tm | Notes |
|----------|------|--------|-----|-----|-------|
| bimagrumab | A | PSR >0.4 | 0.697 | - | Highest PSR in specific |
| bavituximab | A | PSR >0.4 | 0.557 | - | High PSR |
| ganitumab | A | PSR >0.4 | 0.553 | - | High PSR |
| eldelumab | B | Tm <60°C | - | 59.50°C | Extreme thermal instability |
| infliximab | C | Clinical | - | - | 61% ADA (NEJM) + chimeric |

---

## Appendix B: Candidate Antibodies for Additional Reclassification

*To be populated during Phase 1 data preparation*

Candidates should have:
- PSR between 0.2-0.4 (below our threshold but elevated)
- Model prediction probability between 0.4-0.5 (borderline)
- Any documented clinical/developmental issues
- Biophysical outliers (extreme HIC, AC-SINS, etc.)

---

## Appendix C: Historical Context

### Retired Methodologies

**94→86 VH-length + Clinical QC (RETIRED):**
- Removed 3 VH length outliers (crenezumab, fletikumab, secukinumab)
- Removed 5 clinical/borderline (muromonab, cetuximab, girentuximab, tabalumab, abituzumab)
- Did NOT match Novo parity
- Files archived in `preprocessing/jain/archive/`

**Why P5e-S2 was adopted:**
- Biologically principled (PSR measures polyreactivity directly)
- Uses published biophysical data from Jain SD03
- Achieves closest match (66.28% vs 68.6%)

---

## Appendix D: Confusion Matrix Interpretation

```
                Predicted Label
                0 (Specific)    1 (Non-specific)
True Label 0:   TN=40           FP=17 (Novo) / 19 (Ours)
True Label 1:   FN=10           TP=19 (Novo) / 17 (Ours)
```

**Observation:**
- TN and FN are identical — the model predicts the same number of True Negatives and False Negatives
- FP/TP differ by 2 — we have 2 more False Positives and 2 fewer True Positives
- This means 2 antibodies that are labeled as specific in our dataset are labeled as non-specific in Novo's

**Implication:**
The discrepancy is in the **label distribution**, not in the model's behavior. If we reclassify 2 specific → non-specific, the confusion matrix would shift:
- FP: 19 → 17 (2 fewer specific to misclassify)
- TP: 17 → 19 (2 more non-specific to correctly classify)

---

**End of Research Spec**
