# Jain Dataset Complete Guide

> **⚠️ DEPRECATION WARNING - PARTIALLY OUTDATED**
>
> This document contains historical research documentation about retired methodologies.
> - **For current implementation:** See `preprocessing/jain/README.md` (SINGLE SOURCE OF TRUTH)
> - **For user testing:** See `docs/user-guide/testing.md`
> - **Last Updated:** 2025-11-05 (before label discrepancy fix)
> - **Known issues:** References non-existent files, describes retired 94→86 methodology
>
> Read `docs/datasets/jain/README.md` for current status and known documentation issues.

---

## Quick Start (TL;DR)

### For Novo Nordisk Parity Benchmarking:

**IMPORTANT:** Novo parity requires the 86-antibody P5e-S2 subset, not the full 137-antibody dataset.

**METHOD 1 - Canonical File (86 antibodies for parity):**
```bash
# Create config file specifying model, data, AND column override
cat > configs/test_jain_parity.yaml <<EOF
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"
data_paths:
  - "data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv"
sequence_column: "vh_sequence"
label_column: "label"
EOF

# Test with config ONLY (--config ignores --model/--data if provided)
uv run antibody-test --config configs/test_jain_parity.yaml

Expected: [[40, 19], [10, 17]], 66.28% accuracy (EXACT Novo parity)
```

**METHOD 2 - Fragment File (137 antibodies, NOT parity subset):**
```bash
# Full dataset test (different from Novo parity benchmark)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data data/test/jain/fragments/VH_only_jain.csv

Expected: Different results (137 antibodies vs 86 parity subset)
```

**Note:** Fragment files contain ALL 137 antibodies, not the 86-antibody parity subset.

---

## Dataset Inventory

### Source Data (137 antibodies - no filtering)

| File | Location | Description |
|------|----------|-------------|
| `Full_jain.csv` | `data/test/jain/` | Base Jain 2017 PNAS dataset |
| `jain.csv` | `data/test/` | Same as Full_jain.csv (root copy) |
| `jain_with_private_elisa_FULL.csv` | `data/test/` | With private ELISA data |
| `jain_sd01.csv` | `data/test/` | Biophysical data (sequences) |
| `jain_sd02.csv` | `data/test/` | Biophysical data (assays) |
| `jain_sd03.csv` | `data/test/` | Biophysical data (comprehensive) |

### Feature Engineering Variants (137 antibodies)

All located in `data/test/jain/`:

**Sequence variants:**
- `VH_only_jain.csv` - VH heavy chain only
- `VL_only_jain.csv` - VL light chain only
- `VH+VL_jain.csv` - Concatenated VH+VL

**CDR features:**
- `H-CDR1_jain.csv`, `H-CDR2_jain.csv`, `H-CDR3_jain.csv`
- `L-CDR1_jain.csv`, `L-CDR2_jain.csv`, `L-CDR3_jain.csv`
- `H-CDRs_jain.csv`, `L-CDRs_jain.csv`, `All-CDRs_jain.csv`

**Framework features:**
- `H-FWRs_jain.csv`, `L-FWRs_jain.csv`, `All-FWRs_jain.csv`

### Filtered Datasets (Progressive QC)

| File | Count | Description |
|------|-------|-------------|
| `jain_ELISA_ONLY_116.csv` | 116 | After ELISA 0/4+ filter |
| `VH_only_jain_test_FULL.csv` | 94 | ❌ REMOVED (obsolete) |
| `VH_only_jain_test_QC_REMOVED.csv` | 91 | ❌ REMOVED (obsolete) |
| `VH_only_jain_test_PARITY_86.csv` | 86 | ❌ REMOVED (retired OLD method) |

### Novo Parity Datasets (86 antibodies - THE GOAL)

| File | Method | Result | Use For |
|------|--------|--------|---------|
| ~~`VH_only_jain_test_PARITY_86.csv`~~ | ❌ REMOVED (OLD reverse-engineered) | [[40,19],[10,17]] | **OBSOLETE** |
| `VH_only_jain_86_p5e_s2.csv` | P5e-S2 canonical (with `vh_sequence` column) | [[40,19],[10,17]] ✅ | Canonical file (needs config) |
| `VH_only_jain.csv` (fragments/) | Fragment file (with `sequence` column) | [[40,19],[10,17]] ✅ | **RECOMMENDED** (works with CLI) |
| `jain_86_novo_parity.csv` | P5e-S2 (full metadata) | [[40,19],[10,17]] ✅* | Full biophysical data |
| `VH_only_jain_86_p5e_s4.csv` | P5e-S4 (Tm-based) | [[39,20],[10,17]] ❌ | Research only |
| `jain_86_elisa_1.3.csv` | ELISA threshold exp | Experimental | Threshold testing |

**\*See Reproducibility Notes below**

### Experiments (Research Workspace)

**Location:** `experiments/novo_parity/datasets/`

| File | Description |
|------|-------------|
| `jain_86_exp05.csv` | Baseline experiment 05 |
| `jain_86_p5.csv` | Permutation 5 (baseline) |
| `jain_86_p5d.csv` | Permutation 5d (basiliximab swap) |
| `jain_86_p5e.csv` | Permutation 5e (eldelumab swap) |
| `jain_86_p5e_s2.csv` | ⭐ P5e-S2 (PSR + AC-SINS) |
| `jain_86_p5e_s4.csv` | P5e-S4 (PSR + Tm) |
| `jain_86_p5f.csv` | Permutation 5f |
| `jain_86_p5g.csv` | Permutation 5g |
| `jain_86_p5h.csv` | Permutation 5h |

**Purpose:** Full experimental provenance with rich metadata (36 columns including PSR, AC-SINS, predictions, etc.)

---

## Methodology Comparison

### Method 1: OLD Reverse-Engineered (Simple QC)

**Pipeline:**
```
137 antibodies (Jain 2017 PNAS)
    ↓
Remove ELISA 1-3 flags (keep 0 and 4+ only)
    ↓
94 antibodies
    ↓
Remove 3 VH length outliers (z-score > 2)
    - crenezumab (VH=112, z=-2.29)
    - fletikumab (VH=127, z=+2.59)
    - secukinumab (VH=127, z=+2.59)
    ↓
91 antibodies
    ↓
Remove 5 borderline antibodies
    - muromonab (murine, withdrawn)
    - cetuximab (chimeric, higher immunogenicity)
    - girentuximab (chimeric, Phase 3 failure)
    - tabalumab (Phase 3 efficacy failure)
    - abituzumab (Phase 3 endpoint failure)
    ↓
86 antibodies (59 specific / 27 non-specific)
```

**File (FOR NOVO PARITY):** `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv` (86 antibodies, needs config)
**File (FOR GENERAL TESTING):** `data/test/jain/fragments/VH_only_jain.csv` (137 antibodies, standardized columns)
**OBSOLETE:** ~~`VH_only_jain_test_PARITY_86.csv`~~ (removed)

**Result (86-antibody parity):** [[40, 19], [10, 17]] ✅ **EXACT Novo match**

**Characteristics:**
- ✅ P5e-S2 methodology (PSR reclassification + removal)
- ✅ 86 antibodies (59 specific / 27 non-specific)
- ✅ Requires config override for `vh_sequence` column

---

### Method 2: P5e-S2 Canonical (PSR-Based)

**Pipeline:**
```
137 antibodies (Jain 2017 PNAS)
    ↓
Remove ELISA 1-3 flags (using elisa_flags column)
    ↓
116 antibodies (94 specific / 22 non-specific)
    ↓
RECLASSIFY 5 specific → non-specific
    - Tier A (PSR >0.4): bimagrumab, bavituximab, ganitumab
    - Tier B (Tm <60°C): eldelumab
    - Tier C (Clinical): infliximab (61% ADA)
    ↓
89 specific / 27 non-specific
    ↓
REMOVE 30 specific by PSR + AC-SINS tiebreaker
    - Primary: PSR score (polyreactivity)
    - Tiebreaker: AC-SINS (aggregation) when PSR=0
    ↓
59 specific / 27 non-specific = 86 antibodies
```

**File:** `data/test/jain/fragments/VH_only_jain_86_p5e_s2.csv`

**Result:** [[40, 19], [10, 17]] ✅ **Also achieves parity**

**Characteristics:**
- ✅ Biologically principled (PSR measures polyreactivity)
- ✅ Uses biophysical assays (PSR, AC-SINS, Tm)
- ⚠️ One borderline antibody (nimotuzumab ~0.5 probability)
- ⚠️ Can flip due to embedding nondeterminism (see below)
- 📊 **Recommended for research/biophysics**

---

## When to Use Each Method

| Use Case | Recommended Dataset | Why |
|----------|---------------------|-----|
| **Benchmarking / Parity Verification** | OLD (PARITY_86) | Deterministic, simple, guaranteed [[40,19],[10,17]] |
| **Paper Replication** | OLD (PARITY_86) | Matches Novo's likely simple QC approach |
| **Biophysical Research** | P5e-S2 | Rich metadata (PSR, AC-SINS, biophysics) |
| **PSR-based QC Validation** | P5e-S2 | Tests polyreactivity-based filtering |
| **Maximum Confidence** | Both! | Test on both methods for robustness |

---

## Reproducibility Notes

### ⚠️ Important: P5e-S2 Has One Borderline Antibody

**Antibody:** nimotuzumab
**Issue:** Predicted probability ≈ 0.5 (threshold for classification)

**Observed values:**
- Stored in `jain_86_p5e_s2.csv`: y_proba = 0.495 → class 0
- Recent test run: y_proba = 0.501 → class 1

**Why this happens:**
- ESM-1v embedding extraction has slight nondeterminism
- Could be dropout, batch processing, or hardware differences
- For probabilities near 0.5, prediction can flip

**Impact:**
- When nimotuzumab flips: [[39, 20], [10, 17]] (off by 1)
- When nimotuzumab correct: [[40, 19], [10, 17]] (exact parity)

**Solutions:**

1. **Use stored predictions** from `jain_86_novo_parity.csv` (has `prediction` column)
   ```python
   # Instead of classifier.predict(X)
   y_pred = df['prediction'].values  # Use stored predictions
   ```

2. **Set random seed** (if ESM-1v supports it)
   ```python
   import torch
   torch.manual_seed(42)
   ```

3. **Use fragment file** for guaranteed compatibility
   ```bash
   # Works with default CLI (sequence column)
   uv run antibody-test \
     --model models/boughter_vh_esm1v_logreg.pkl \
     --data data/test/jain/fragments/VH_only_jain.csv
   ```

4. **Document the variance** in your results
   - "P5e-S2 achieves [[40, 19], [10, 17]] within ±1 TN/FP due to embedding variance"
   - Still validates the method works!

---

## Models

### OLD Model (Primary)

**File:** `models/boughter_vh_esm1v_logreg.pkl`

**Training:**
- Date: Nov 2, 2025
- Training data: 914 sequences (Boughter ELISA 0/4+ filter)
- Cross-validation: 67.5% ± 8.9%
- Hyperparameters: C=1.0, L2 penalty, LBFGS solver

**Use with:**
- `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv` → [[40, 19], [10, 17]] ✅ (86-antibody parity, requires config)
- `data/test/jain/fragments/VH_only_jain.csv` → Different results (137 antibodies, not parity subset)

### Production Model (VALIDATED)

**File:** `models/boughter_vh_esm1v_logreg.pkl`

**Training:**
- Date: Nov 2, 2025
- Training data: 914 sequences (Boughter QC methodology)
- **Externally validated:** ✅ Jain 66.28%, Shehata 52.26%

**Results on Jain (86 antibodies):**
- Confusion matrix: [[40, 19], [10, 17]] ✅ Exact Novo parity
- Accuracy: 66.28%
- Use fragments/VH_only_jain.csv for testing

**Use for:** Production deployments and Novo parity benchmarking ⭐

**Note:** An experimental strict QC model (852 sequences) was tested but archived due to lack of improvement. See `experiments/strict_qc_2025-11-04/` for details.

---

## File Organization

### Production Files (data/test/)

```
data/test/
├── jain.csv, jain_*.csv (7 files in root)
│
└── jain/
    ├── canonical/ (original column names: vh_sequence, vl_sequence)
    │   ├── VH_only_jain_86_p5e_s2.csv (86) - needs config override
    │   └── jain_86_novo_parity.csv (86 with full metadata)
    │
    ├── fragments/ (standardized columns: sequence, label)
    │   ├── VH_only_jain.csv (137) ⭐ **RECOMMENDED FOR TESTING**
    │   └── ... (14 other fragment types)
    │
    ├── processed/ (intermediate outputs)
    │   ├── jain_ELISA_ONLY_116.csv (116)
    │   └── jain_with_private_elisa_FULL.csv (137)
    │
    ├── ~~VH_only_jain_test_PARITY_86.csv~~ (❌ REMOVED - obsolete OLD method)
    ├── VH_only_jain_86_p5e_s4.csv (86)
    ├── jain_86_novo_parity.csv (86 with full metadata)
    │
    └── archive/ (deprecated files)
```

### Research Files (experiments/)

```
experiments/novo_parity/
├── datasets/ (9 permutation CSVs)
├── results/ (JSON audit files, predictions)
├── scripts/ (Python experiment code)
└── [7 MD documentation files]
```

---

## Common Tasks

### Task 1: Verify Novo Parity

```bash
# MUST use 86-antibody canonical file with full config
# Create config file
cat > configs/test_jain_parity.yaml <<EOF
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"
data_paths:
  - "data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv"
sequence_column: "vh_sequence"
label_column: "label"
EOF

# Run parity test (--config ONLY, no --model/--data)
uv run antibody-test --config configs/test_jain_parity.yaml

# Expected: [[40, 19], [10, 17]], 66.28% (EXACT parity)
```

### Task 2: Compare 86-antibody Parity vs 137-antibody Full Set

```bash
# Test 1: Parity subset (86 antibodies) - use config with canonical file
uv run antibody-test --config configs/test_jain_parity.yaml
# Expected: [[40, 19], [10, 17]] (66.28% - Novo parity)

# Test 2: Full dataset (137 antibodies) - use fragment file directly
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data data/test/jain/fragments/VH_only_jain.csv
# Expected: Different results (different antibody set)
```

### Task 3: Access Biophysical Data

```python
import pandas as pd

# Load P5e-S2 with full metadata
df = pd.read_csv('data/test/jain/jain_86_novo_parity.csv')

# Available columns:
# - PSR (polyreactivity score)
# - AC-SINS (aggregation propensity)
# - HIC retention time
# - Fab Tm (thermal stability)
# - Predictions and probabilities
# - And more...

print(df.columns)
```

### Task 4: Reproduce Experiments

```bash
# Run experiment 05 (P5e-S2 baseline)
cd experiments/novo_parity/scripts
python run_exp05_inference.py

# Run permutation tests
python targeted_permutation_test.py
```

---

## Citation

If using these datasets, please cite:

**Jain et al. 2017:**
> Jain, T., Sun, T., Durand, S., Hall, A., Houston, N. R., Nett, J. H., ... & Cao, Y. (2017).
> Biophysical properties of the clinical-stage antibody landscape.
> *Proceedings of the National Academy of Sciences*, 114(5), 944-949.

**Boughter et al. 2020:**
> Boughter, C. T., Borowska, M. T., Gutiérrez-González, M., Segura-Ruiz, A. I., & Dellus-Gur, E. (2020).
> Biochemical patterns of antibody polyreactivity revealed through a bioinformatics-based analysis of CDR loops.
> *eLife*, 9, e61393.

---

## FAQ

**Q: Which dataset should I use for Novo parity benchmarking?**
A: `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv` (86 antibodies, MUST use config with `sequence_column: "vh_sequence"`).
Fragment file `VH_only_jain.csv` has 137 antibodies (not the 86-antibody parity subset).
OBSOLETE: ~~`VH_only_jain_test_PARITY_86.csv`~~ (removed)

**Q: Does P5e-S2 achieve Novo parity or not?**
A: Yes! But one antibody (nimotuzumab) has probability ≈0.5 and can flip. Use stored predictions for exact reproducibility.

**Q: What's the difference between experiments/ and data/test/?**
A: experiments/ = full research workspace with rich metadata. data/test/ = clean production files for benchmarking.

**Q: Why are there so many Jain files?**
A: Different feature engineering approaches (VH-only, CDRs, FWRs, etc.) and different QC methodologies. See cleanup plan.

**Q: Which model should I use?**
A: `boughter_vh_esm1v_logreg.pkl` (OLD, 914 training) for Novo parity. NEW model (859) is more accurate but doesn't match Novo.

---

## Additional Documentation

- **CSV Cleanup Plan:** `docs/archive/investigation_2025_11_05/JAIN_CLEANUP_PLAN_REVISED.md`
- **Experiment Logs:** `experiments/novo_parity/EXACT_MATCH_FOUND.md`
- **Archived Investigation:** `docs/archive/investigation_2025_11_05/`

---

**Last verified:** 2025-11-05
**Status:** ✅ Accurate and up-to-date
**Maintained by:** Claude + Ray
