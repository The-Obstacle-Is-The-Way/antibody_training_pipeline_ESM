# Jain Parity Investigation — Complete Data Inventory

**Status:** STABLE REFERENCE DOCUMENT
**Created:** 2025-12-15
**Purpose:** Document all data artifacts and filtering stages for permutation experiments
**Branch:** `investigate/jain-parity-verification` (STABLE - do not modify data here)

---

## Quick Reference

| Stage | Count | Specific | Non-Specific | File |
|-------|-------|----------|--------------|------|
| FULL | 137 | 94 | 22 + 21 mild | `processed/jain_with_private_elisa_FULL.csv` |
| After ELISA 1-3 removal | 116 | 94 | 22 | `processed/jain_ELISA_ONLY_116.csv` |
| After reclassification | 116 | 89 | 27 | (computed, not saved) |
| After removal (OUR RESULT) | 86 | 59 | 27 | `canonical/jain_86_novo_parity.csv` |
| **NOVO TARGET** | 86 | **57** | **29** | Figure S14A |

**THE GAP:** We have 59/27, Novo has 57/29. Off by 2 antibodies.

---

## Data Directory Structure

```
data/test/jain/
├── README.md
├── raw/                              # Original Excel files (DO NOT MODIFY)
│   ├── jain-pnas.1616408114.sd01.xlsx
│   ├── jain-pnas.1616408114.sd02.xlsx
│   ├── jain-pnas.1616408114.sd03.xlsx
│   ├── Private_Jain2017_ELISA_indiv.xlsx
│   └── README.md
├── processed/                        # Intermediate files
│   ├── jain_with_private_elisa_FULL.csv    # 137 antibodies (SOURCE)
│   ├── jain_ELISA_ONLY_116.csv             # 116 antibodies (SSOT)
│   ├── jain_sd01.csv                       # Biophysical - sequences
│   ├── jain_sd02.csv                       # Biophysical - assays
│   ├── jain_sd03.csv                       # Biophysical - comprehensive (PSR, AC-SINS, etc.)
│   └── README.md
├── canonical/                        # Final outputs
│   ├── jain_86_novo_parity.csv             # 86 antibodies (full metadata)
│   ├── VH_only_jain_86_p5e_s2.csv          # 86 antibodies (VH only)
│   ├── archive/
│   │   └── VH_only_jain_test_PARITY_86.csv # DEPRECATED
│   └── README.md
└── fragments/                        # All 137 antibodies, various regions
    ├── VH_only_jain.csv              # 137 antibodies (NOT parity subset)
    └── ... (other fragment types)
```

---

## Filtering Pipeline (P5e-S2 Method)

### Stage 0: Starting Point
**File:** `processed/jain_with_private_elisa_FULL.csv`
**Count:** 137 antibodies
**Labels:** 94 specific (0) + 22 non-specific (1) + 21 mild (NaN, excluded)

### Stage 1: ELISA 1-3 Removal
**File:** `processed/jain_ELISA_ONLY_116.csv`
**Count:** 116 antibodies
**Labels:** 94 specific (0) + 22 non-specific (1)
**Removed:** 21 antibodies with ELISA flags 1, 2, or 3

### Stage 2: Reclassification (5 antibodies)
**File:** NOT SAVED (computed in step2_preprocess_p5e_s2.py)
**Count:** 116 antibodies
**Labels:** 89 specific (0) + 27 non-specific (1)

**Reclassified antibodies (specific → non-specific):**

| Antibody | Tier | Reason | PSR | Fab Tm |
|----------|------|--------|-----|--------|
| bimagrumab | A | PSR >0.4 | 0.697 | - |
| bavituximab | A | PSR >0.4 | 0.557 | - |
| ganitumab | A | PSR >0.4 | 0.553 | - |
| eldelumab | B | Extreme Tm | - | 59.50°C |
| infliximab | C | Clinical (61% ADA) | - | - |

### Stage 3: Removal (30 antibodies)
**File:** `canonical/jain_86_novo_parity.csv`
**Count:** 86 antibodies
**Labels:** 59 specific (0) + 27 non-specific (1)

**Removal criteria:** Top 30 specific by PSR (descending), AC-SINS tiebreaker

---

## The 89 Specific Antibodies (After Reclassification, Before Removal)

This is the **EXPERIMENT STARTING POINT**. These 89 antibodies are sorted by PSR descending.

### REMOVED (Top 30 by PSR/AC-SINS) — These went from 89 → 59

| # | Antibody | PSR | AC-SINS | Fab Tm | HIC | Status |
|---|----------|-----|---------|--------|-----|--------|
| 1 | olaratumab | 0.483 | 0.31 | 62.5 | 10.61 | REMOVED |
| 2 | basiliximab | 0.397 | 28.76 | 60.5 | 9.58 | REMOVED |
| 3 | rituximab | 0.384 | 2.13 | 69.0 | 10.80 | REMOVED |
| 4 | benralizumab | 0.354 | 5.98 | 76.0 | 9.47 | REMOVED |
| 5 | dinutuximab | 0.303 | 3.64 | 69.0 | 9.83 | REMOVED |
| 6 | pembrolizumab | 0.300 | 5.62 | 66.0 | 11.07 | REMOVED |
| 7 | fezakinumab | 0.265 | 2.50 | 69.0 | 11.80 | REMOVED |
| 8 | reslizumab | 0.230 | 1.74 | 75.5 | 9.82 | REMOVED |
| 9 | ipilimumab | 0.230 | 10.41 | 73.0 | 11.57 | REMOVED |
| 10 | lirilumab | 0.183 | 21.00 | 70.0 | 25.00 | REMOVED |
| 11 | muromonab | 0.176 | 2.31 | 74.5 | 8.90 | REMOVED |
| 12 | abituzumab | 0.167 | 1.46 | 75.5 | 9.23 | REMOVED |
| 13 | glembatumumab | 0.166 | 28.88 | 70.5 | 13.68 | REMOVED |
| 14 | rilotumumab | 0.160 | 2.13 | 79.0 | 12.63 | REMOVED |
| 15 | tralokinumab | 0.157 | 4.81 | 63.0 | 10.26 | REMOVED |
| 16 | tremelimumab | 0.145 | 29.65 | 75.0 | 11.56 | REMOVED |
| 17 | lumiliximab | 0.145 | 1.44 | 64.5 | 9.54 | REMOVED |
| 18 | nivolumab | 0.135 | 2.42 | 66.0 | 9.02 | REMOVED |
| 19 | radretumab | 0.134 | 3.38 | 77.0 | 9.51 | REMOVED |
| 20 | tigatuzumab | 0.129 | 5.49 | 64.5 | 10.02 | REMOVED |
| 21 | zanolimumab | 0.127 | 1.46 | 80.5 | 9.59 | REMOVED |
| 22 | enokizumab | 0.126 | 1.53 | 68.0 | 12.93 | REMOVED |
| 23 | obinutuzumab | 0.113 | 1.83 | 73.0 | 10.64 | REMOVED |
| 24 | crenezumab | 0.105 | 6.37 | 72.0 | 10.03 | REMOVED |
| 25 | atezolizumab | 0.066 | 14.97 | 73.5 | 13.35 | REMOVED |
| 26 | pinatuzumab | 0.011 | 0.59 | 79.0 | 9.22 | REMOVED |
| 27 | seribantumab | 0.000 | 21.21 | 77.5 | 10.42 | REMOVED |
| 28 | urelumab | 0.000 | 29.65 | 66.0 | 11.16 | REMOVED |
| 29 | drozitumab | 0.000 | 29.65 | 63.0 | 9.29 | REMOVED |
| 30 | ocrelizumab | 0.000 | 17.88 | 70.5 | 9.91 | REMOVED |

### KEPT (Bottom 59 by PSR/AC-SINS) — These are in the final 86

| # | Antibody | PSR | AC-SINS | Fab Tm | HIC |
|---|----------|-----|---------|--------|-----|
| 1 | efalizumab | 0.000 | 0.68 | 72.5 | 8.67 |
| 2 | bapineuzumab | 0.000 | -0.73 | 73.0 | 8.86 |
| 3 | ramucirumab | 0.000 | 0.02 | 66.0 | 9.43 |
| 4 | bevacizumab | 0.000 | 0.79 | 63.5 | 11.77 |
| 5 | polatuzumab | 0.000 | -1.00 | 74.0 | 8.76 |
| 6 | pertuzumab | 0.000 | -0.20 | 78.5 | 10.11 |
| 7 | romosozumab | 0.000 | -1.02 | 76.0 | 9.18 |
| 8 | canakinumab | 0.000 | 0.67 | 72.0 | 9.32 |
| 9 | panobacumab | 0.000 | -0.42 | 69.0 | 9.83 |
| 10 | panitumumab | 0.000 | -1.08 | 78.5 | 9.48 |
| 11 | palivizumab | 0.000 | -0.85 | 79.5 | 9.33 |
| 12 | otlertuzumab | 0.000 | 2.26 | 68.5 | 10.96 |
| 13 | anifrolumab | 0.000 | -0.56 | 62.5 | 8.80 |
| 14 | sarilumab | 0.000 | 1.11 | 64.0 | 8.99 |
| 15 | onartuzumab | 0.000 | -0.05 | 80.0 | 9.92 |
| 16 | secukinumab | 0.000 | -0.58 | 72.0 | 11.39 |
| 17 | siltuximab | 0.000 | 2.64 | 64.5 | 11.00 |
| 18 | tabalumab | 0.000 | 1.96 | 64.0 | 10.85 |
| 19 | alirocumab | 0.000 | 1.23 | 71.5 | 9.04 |
| 20 | tildrakizumab | 0.000 | 0.81 | 77.5 | 11.08 |
| 21 | tocilizumab | 0.000 | 1.32 | 91.5 | 9.09 |
| 22 | tovetumab | 0.000 | 2.25 | 63.5 | 8.67 |
| 23 | alemtuzumab | 0.000 | -0.79 | 74.5 | 8.77 |
| 24 | trastuzumab | 0.000 | 2.04 | 78.5 | 9.66 |
| 25 | adalimumab | 0.000 | 1.06 | 71.0 | 8.82 |
| 26 | vedolizumab | 0.000 | 0.39 | 80.5 | 10.94 |
| 27 | veltuzumab | 0.000 | 4.83 | 70.0 | 11.09 |
| 28 | otelixizumab | 0.000 | 4.44 | 75.5 | 9.08 |
| 29 | olokizumab | 0.000 | -0.50 | 69.0 | 9.91 |
| 30 | omalizumab | 0.000 | -0.44 | 77.5 | 9.52 |
| 31 | elotuzumab | 0.000 | -0.22 | 83.5 | 10.31 |
| 32 | farletuzumab | 0.000 | -0.50 | 75.5 | 9.49 |
| 33 | fasinumab | 0.000 | -0.67 | 71.0 | 10.03 |
| 34 | eculizumab | 0.000 | 0.04 | 66.0 | 10.41 |
| 35 | ficlatuzumab | 0.000 | -0.89 | 75.0 | 9.42 |
| 36 | fletikumab | 0.000 | -0.13 | 71.5 | 11.04 |
| 37 | fresolimumab | 0.000 | -0.52 | 74.0 | 10.88 |
| 38 | galiximab | 0.000 | 1.09 | 67.5 | 12.20 |
| 39 | gemtuzumab | 0.000 | 1.02 | 72.5 | 12.26 |
| 40 | gevokizumab | 0.000 | -0.51 | 71.5 | 8.83 |
| 41 | girentuximab | 0.000 | -0.75 | 63.0 | 9.08 |
| 42 | ibalizumab | 0.000 | -0.34 | 72.0 | 10.24 |
| 43 | daratumumab | 0.000 | 1.81 | 71.0 | 9.51 |
| 44 | lampalizumab | 0.000 | 0.49 | 67.0 | 9.25 |
| 45 | lebrikizumab | 0.000 | 0.26 | 66.0 | 12.38 |
| 46 | lintuzumab | 0.000 | 0.89 | 75.5 | 10.87 |
| 47 | dacetuzumab | 0.000 | -0.01 | 68.0 | 8.47 |
| 48 | matuzumab | 0.000 | -0.93 | 72.0 | 9.84 |
| 49 | mavrilimumab | 0.000 | -0.78 | 68.5 | 10.30 |
| 50 | abrilumab | 0.000 | -0.93 | 71.0 | 9.41 |
| 51 | mogamulizumab | 0.000 | -0.51 | 68.5 | 9.64 |
| 52 | clazakizumab | 0.000 | 0.93 | 69.5 | 9.57 |
| 53 | natalizumab | 0.000 | 0.82 | 79.5 | 9.70 |
| 54 | necitumumab | 0.000 | 1.30 | 76.5 | 10.81 |
| 55 | nimotuzumab | 0.000 | -0.59 | 65.5 | 25.00 |
| 56 | cetuximab | 0.000 | 1.30 | 68.5 | 10.11 |
| 57 | ofatumumab | 0.000 | 1.21 | 68.0 | 9.73 |
| 58 | certolizumab | 0.000 | 0.16 | 81.5 | 11.48 |
| 59 | mepolizumab | 0.000 | -0.96 | 78.5 | 9.24 |

---

## The 27 Non-Specific Antibodies (in final 86)

| Antibody | Original Label | Reclassified? | Reason |
|----------|----------------|---------------|--------|
| bavituximab | specific | YES | Tier A: PSR >0.4 |
| bimagrumab | specific | YES | Tier A: PSR >0.4 |
| ganitumab | specific | YES | Tier A: PSR >0.4 |
| eldelumab | specific | YES | Tier B: Extreme Tm (59.50°C) |
| infliximab | specific | YES | Tier C: Clinical (61% ADA) |
| belimumab | non-specific | NO | Original ELISA ≥4 |
| blosozumab | non-specific | NO | Original ELISA ≥4 |
| bococizumab | non-specific | NO | Original ELISA ≥4 |
| briakinumab | non-specific | NO | Original ELISA ≥4 |
| carlumab | non-specific | NO | Original ELISA ≥4 |
| cixutumumab | non-specific | NO | Original ELISA ≥4 |
| codrituzumab | non-specific | NO | Original ELISA ≥4 |
| dalotuzumab | non-specific | NO | Original ELISA ≥4 |
| denosumab | non-specific | NO | Original ELISA ≥4 |
| duligotuzumab | non-specific | NO | Original ELISA ≥4 |
| dupilumab | non-specific | NO | Original ELISA ≥4 |
| emibetuzumab | non-specific | NO | Original ELISA ≥4 |
| gantenerumab | non-specific | NO | Original ELISA ≥4 |
| imgatuzumab | non-specific | NO | Original ELISA ≥4 |
| ixekizumab | non-specific | NO | Original ELISA ≥4 |
| lenzilumab | non-specific | NO | Original ELISA ≥4 |
| parsatuzumab | non-specific | NO | Original ELISA ≥4 |
| patritumab | non-specific | NO | Original ELISA ≥4 |
| ponezumab | non-specific | NO | Original ELISA ≥4 |
| robatumumab | non-specific | NO | Original ELISA ≥4 |
| simtuzumab | non-specific | NO | Original ELISA ≥4 |
| sirukumab | non-specific | NO | Original ELISA ≥4 |

---

## Experiment Strategy

### Goal
Find the preprocessing methodology that produces **57 specific + 29 non-specific = 86 total**, matching Novo's Figure S14A confusion matrix `[[40, 17], [10, 19]]`.

### Key Insight
Since TN=40 and FN=10 match exactly, the issue is in the **label distribution**, not the model. We need to reclassify 2 more specific → non-specific.

### Experimental Approaches

#### Approach A: Additional Reclassification
1. Take the 59 specific antibodies in the final 86
2. Try all C(59,2) = 1,711 pairs
3. For each pair, flip labels to non-specific
4. Run inference, check if confusion matrix matches `[[40, 17], [10, 19]]`

#### Approach B: Different Removal Strategy
1. Take the 89 specific antibodies (after reclassification, before removal)
2. Try different ranking criteria (HIC, Tm, combined scores)
3. Remove top 30 by each criterion
4. Check which produces 57/29 split

#### Approach C: Different Reclassification Thresholds
1. Vary PSR threshold (0.35, 0.38, 0.40, 0.42, 0.45)
2. Vary Tm threshold (58°C, 60°C, 62°C)
3. Check which produces 57/29 split

#### Approach D: Alternative QC Filters
1. Try VH length outlier removal
2. Try clinical status filters (withdrawn, failed trials)
3. Try immunogenicity filters (chimeric antibodies)

---

## Files for Experiments

### Input Files (DO NOT MODIFY)
- `data/test/jain/processed/jain_ELISA_ONLY_116.csv` — 116 antibodies with original labels
- `data/test/jain/processed/jain_sd03.csv` — Biophysical data (PSR, AC-SINS, HIC, Tm)

### Reference Files
- `data/test/jain/canonical/jain_86_novo_parity.csv` — Our current 86-antibody output
- `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv` — VH sequences for inference

### Experiment Outputs (CREATE IN SANDBOX BRANCH)
- `experiments/benchmarks/novo_parity/datasets/` — Experiment datasets
- `experiments/benchmarks/novo_parity/results/` — Confusion matrices, metrics

---

## Branch Strategy

1. **Current branch:** `investigate/jain-parity-verification` — STABLE reference
2. **Create new branch:** `experiment/jain-parity-permutations` — For running experiments
3. **If experiments succeed:** Merge findings back to investigation branch
4. **If experiments fail:** Delete experiment branch, try different approach

---

## Validation Criteria

Any proposed solution MUST:
1. **Produce 57/29 split** — Exactly 57 specific + 29 non-specific
2. **Match confusion matrix** — `[[40, 17], [10, 19]]` with ESM-1v VH LogReg model
3. **Be biologically principled** — Not cherry-picking based on results
4. **Be reproducible** — Same result every time (accounting for embedding variance ±1)

---

## Next Steps

1. [ ] Commit this document and research spec to stable branch
2. [ ] Create `experiment/jain-parity-permutations` branch
3. [ ] Implement Phase 2: Systematic permutation search
4. [ ] Analyze results for biological plausibility
5. [ ] Update preprocessing pipeline if solution found

---

**End of Data Inventory**
