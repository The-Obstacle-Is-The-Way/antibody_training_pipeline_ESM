# Jain Parity Decision — Triple Agent Convergence

**Status:** CONSENSUS REACHED (2025-12-16)
**Decision:** lebrikizumab + galiximab
**Confidence:** High (3/3 agents agree)
**GitHub Issue:** [#33](https://github.com/Clarity-Digital-Twin/antibody_training_pipeline_ESM/issues/33)

---

## Executive Summary

Three independent AI agents (Google DeepThink, ChatGPT, Claude) were asked to analyze which of the 3 matching antibody pairs Novo Nordisk most likely used for their Jain dataset preprocessing. All three converged on the same recommendation.

**Consensus:** Reclassify **lebrikizumab + galiximab** from specific to non-specific.

---

## The 3 Matching Pairs (All Produce Exact Novo Parity)

| Pair | Flag Types | Confusion Matrix | Accuracy |
|------|-----------|------------------|----------|
| **lebrikizumab + galiximab** | chromatography + chromatography | `[[40, 17], [10, 19]]` | 68.60% |
| lebrikizumab + otelixizumab | chromatography + stability | `[[40, 17], [10, 19]]` | 68.60% |
| galiximab + otelixizumab | chromatography + stability | `[[40, 17], [10, 19]]` | 68.60% |

---

## Agent Recommendations

### Google DeepThink (Gemini)

| Field | Value |
|-------|-------|
| **Recommendation** | lebrikizumab + galiximab |
| **Confidence** | High |
| **Rationale** | HIC/SMAC = "stickiness" directly measures non-specificity. Both share same flag type (chromatography), enabling a single methodologically consistent rule. Novo paper explicitly uses chromatography-based features. |
| **What would increase confidence** | Novo's exact antibody-level label list, preprocessing code, or explicit mention of chromatography-based reclassification rule. |

### ChatGPT (GPT-4)

| Field | Value |
|-------|-------|
| **Recommendation** | lebrikizumab + galiximab |
| **Confidence** | Medium |
| **Rationale** | Single mechanism (chromatography/HIC) is most methodologically consistent. Jain 2017 treats SMAC/HIC as a coherent cluster capturing self-association. Pulling in otelixizumab would mix in stability (distinct axis). |
| **What would increase confidence** | Evidence Novo applied explicit "chromatography red flags" rule, or per-ligand ELISA panel data to rule out "true ELISA flags differ" hypothesis. |

### Claude (Anthropic)

| Field | Value |
|-------|-------|
| **Recommendation** | lebrikizumab + galiximab |
| **Confidence** | High |
| **Rationale** | (1) Mechanistic consistency: HIC measures hydrophobicity → directly related to non-specific binding. (2) Same flag type = single defensible rule. (3) Occam's Razor: simplest explanation. (4) Blind criterion: would be flagged by standard QC regardless of confusion matrix. |
| **What would increase confidence** | Direct confirmation from Novo's preprocessing code or data release. |

---

## Consensus Reasoning

All three agents independently arrived at the same conclusion for the same core reasons:

### 1. Mechanistic Consistency

HIC (Hydrophobic Interaction Chromatography) directly measures **surface hydrophobicity** — the molecular property that causes non-specific binding ("stickiness"). This is the actual mechanism:

```
High surface hydrophobicity → binds to hydrophobic patches on off-target proteins → polyreactivity
```

Stability (aggregation slope) measures a **different** biophysical property — propensity to self-associate. While aggregation is a developability concern, it's not mechanistically equivalent to polyreactivity.

### 2. Same Flag Type = Single Defensible Rule

Both lebrikizumab and galiximab have **chromatography flags** (elevated HIC retention). This allows a consistent rule:

> "Antibodies with chromatography flags (HIC > threshold) are reclassified as non-specific regardless of ELISA results."

Mixing flag types (chromatography + stability) would require a less parsimonious rule:

> "Antibodies with chromatography OR stability flags are reclassified..."

### 3. Jain 2017 Paper Context

Jain et al. (2017) explicitly treats SMAC/HIC as a coherent "stickiness" cluster, distinct from thermal stability. Reclassifying based on chromatography alone is consistent with this published methodology.

### 4. Novo Paper Context

Sakhnini et al. (2025) uses biophysical descriptors including HIC in their Track B model. If they're training a model that uses HIC to predict non-specificity, it's logical they would trust HIC > ELISA for ground truth labeling when there's disagreement.

---

## The 3 Matching Antibodies

| Antibody | Model P(non-spec) | Flag Type | HIC | Additional Data |
|----------|-------------------|-----------|-----|-----------------|
| lebrikizumab | 0.5845 | chromatography | 12.38 | SMAC=15.71, FDA-approved (Ebglyss) |
| galiximab | 0.7963 | chromatography | 12.20 | SMAC=14.77, Phase 3 failure |
| otelixizumab | 0.6815 | stability | 9.08 | Slope=0.088, Phase 3 halted |

**Key observation:** All three are predicted as non-specific by the model (P > 0.5), but only lebrikizumab and galiximab share the same flag type.

---

## Why Not the Other Pairs?

### Pair 2: lebrikizumab + otelixizumab

- Mixes chromatography (HIC) with stability (aggregation)
- No single biophysical criterion explains both
- Less defensible methodology
- Jain 2017 treats these as distinct axes

### Pair 3: galiximab + otelixizumab

- Same problem — mixed flag types
- Would require post-hoc justification for mixing
- Less parsimonious than single-mechanism explanation

---

## Remaining Uncertainty

### What We Cannot Confirm

1. **Novo's exact methodology is not published** — they document the ELISA-flag parsing but not the 116 → 86 step
2. **This is reverse-engineering** — we're inferring their methodology from the confusion matrix
3. **Any of the 3 pairs produces exact parity** — mathematically, all work

### What Would Provide Definitive Confirmation

1. Novo's preprocessing code or antibody-level label list
2. Explicit mention of chromatography-based reclassification in methods/supplement
3. Per-ligand ELISA panel data to rule out "ELISA flags differ" hypothesis

---

## Decision for This Repository

**We adopt lebrikizumab + galiximab as our "preferred parity variant"** with the following caveats:

1. **Documented uncertainty:** We acknowledge this is reverse-engineering, not paper-stated methodology
2. **Alternative pairs preserved:** The other two pairs remain documented in `jain_parity_reverse_engineering.md`
3. **Scientific rationale:** Our choice is based on mechanistic consistency (chromatography/HIC measures stickiness), not outcome optimization
4. **Reproducible:** All data comes from public sources (Jain SD03)

---

## Implementation Plan (PENDING SENIOR REVIEW)

### Phase 1: Documentation Updates

1. Update `docs/bugs/index.md` with decision
2. Update `docs/bugs/jain_parity_reverse_engineering.md` with consensus section
3. Update GitHub Issue #33 with decision summary

### Phase 2: Preprocessing Script Updates

1. Update `preprocessing/jain/step2_preprocess_p5e_s2.py`:
   - Add Tier D: "Chromatography-flagged antibodies predicted as non-specific"
   - Reclassify lebrikizumab + galiximab
2. Add code comments explaining the rationale

### Phase 3: Artifact Regeneration

1. Run updated preprocessing script
2. Regenerate `data/test/jain/canonical/jain_86_novo_parity.csv`
3. Regenerate `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv`
4. Verify confusion matrix matches `[[40, 17], [10, 19]]`

### Phase 4: Verification

1. Run inference with ESM-1v VH LogReg model
2. Confirm accuracy = 68.60%
3. Document verification in results

---

## References

### Agent Queries

- **Google DeepThink:** Asked to analyze which pair Novo most likely used, given biophysical data, clinical history, and Jain/Novo paper context
- **ChatGPT:** Independent analysis with same inputs, emphasizing methodological consistency
- **Claude:** First-principles analysis with focus on mechanistic relevance and Occam's Razor

### Data Sources

- **Chromatography flags:** `jain-pnas.1616408114.sd03.xlsx` (PUBLIC)
- **HIC thresholds:** Jain 2017 Methods (HIC > 11.7 min)
- **Model predictions:** `experiments/benchmarks/novo_parity/results/phase2b_results.json`

### Papers

- Sakhnini et al. (2025) — Novo paper, bioRxiv DOI: 10.1101/2025.04.28.650927
- Jain et al. (2017) — PNAS 114(5), 944-949

---

**End of Decision Document**
