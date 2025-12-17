# Jain Dataset Complete Guide

This guide documents the **current** Jain preprocessing pipeline and the **canonical 86-antibody benchmark artifacts** used to reproduce Novo Nordisk Figure S14A (ESM-1v VH-based Logistic Regression).

**Single sources of truth (SSOT):**
- **Implementation + scripts:** `preprocessing/jain/README.md`
- **Canonical benchmark artifacts:** `data/test/jain/canonical/README.md`
- **Research provenance:** `docs/bugs/jain_parity_reverse_engineering.md`, `docs/bugs/jain_parity_decision.md`

---

## Quick Start

### 1) Verify exact Novo parity

Runs inference on the canonical 86-antibody benchmark set using the production checkpoint. This may download the ESM model weights if not already cached.

```bash
PYTHONPATH=. uv run python preprocessing/jain/test_novo_parity.py
```

**Expected:**
- Confusion matrix: `[[40, 17], [10, 19]]`
- Accuracy: `0.6860` (68.60%)
- Label split: **57 specific / 29 non-specific** (86 total)

### 2) Load the canonical benchmark artifact

```python
import pandas as pd

df = pd.read_csv("data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv")
assert set(df.columns) == {"id", "vh_sequence", "label"}
```

---

## Dataset Overview

- **Source:** Jain et al. (2017), *PNAS* — 137 clinical-stage antibodies with biophysical measurements.
- **Primary label source:** Private per-antibody ELISA panel flags (`Private_Jain2017_ELISA_indiv.xlsx`) as used in the Sakhnini et al. (2025) benchmark.
  - 0 flags → specific (label 0)
  - 1–3 flags → mildly non-specific (excluded)
  - ≥4 flags → non-specific (label 1)
- **Biophysical descriptors:** Public SD03 (`jain-pnas.1616408114.sd03.xlsx`) including PSR, AC-SINS, HIC/SMAC, stability slope, Tm, and related fields used for selection/reclassification.

---

## File Layout (Current)

All files live under `data/test/jain/`:

- `raw/`
  - Original Excel files (do not modify).
- `processed/`
  - CSV exports from Step 1, plus SSOT intermediate outputs used by Step 2.
- `canonical/`
  - The canonical 86-antibody benchmark artifacts used for parity verification:
    - `jain_86_novo_parity.csv` (full metadata)
    - `VH_only_jain_86_p5e_s2.csv` (VH-only inference file)
- `fragments/`
  - Region-specific extracts (CDRs/FWRs/etc.) for the **full 137-antibody** dataset; not used for the 86-antibody parity benchmark.

---

## Preprocessing Pipeline (P5e-S2 + Tier D)

Implemented in `preprocessing/jain/step1_convert_excel_to_csv.py` and `preprocessing/jain/step2_preprocess_p5e_s2.py`.

### Step 1: Excel → CSV conversion

Inputs:
- `data/test/jain/raw/Private_Jain2017_ELISA_indiv.xlsx`
- `data/test/jain/raw/jain-pnas.1616408114.sd01.xlsx`
- `data/test/jain/raw/jain-pnas.1616408114.sd02.xlsx`
- `data/test/jain/raw/jain-pnas.1616408114.sd03.xlsx`

Outputs (key):
- `data/test/jain/processed/jain_with_private_elisa_FULL.csv` (137)
- `data/test/jain/processed/jain_sd03.csv` (biophysical descriptors)

### Step 2: Build the 86-antibody benchmark set

Pipeline (counts):
1. Start: **137**
2. Remove ELISA flags 1–3 (mild) → **116**
3. Reclassify 5 specific → non-specific (Tiers A–C) → **89 specific / 27 non-specific** (still 116 total)
4. Remove 30 specific by PSR (primary) + AC-SINS (tiebreak) → **86** (selection: 59 from the specific pool + 27 non-specific)
5. **Tier D (final label adjustment on the 86-set):** reclassify `lebrikizumab` and `galiximab` as non-specific → **57 specific / 29 non-specific**

Outputs:
- `data/test/jain/canonical/jain_86_novo_parity.csv`
- `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv`

### Tier D Rationale (why these 2)

Tier D uses **public Jain SD03 chromatography flags** (HIC threshold) to reclassify two antibodies as non-specific:
- `lebrikizumab` — HIC above threshold
- `galiximab` — HIC above threshold

This matches the Novo S14A target label split (57/29) and yields exact parity with the published confusion matrix.

Full rationale: `docs/bugs/jain_parity_decision.md`

---

## Using the Benchmark in Code

- For **benchmarking / inference**, prefer the VH-only file:
  - `data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv` (`id`, `vh_sequence`, `label`)
- For **analysis with biophysical metadata**, use:
  - `data/test/jain/canonical/jain_86_novo_parity.csv`

If you load `VH_only_jain_86_p5e_s2.csv` through `JainDataset.load_data(...)`, treat it as already-filtered data (do not re-run the parity filter stages on it).

---

## Historical Notes

Older Jain preprocessing approaches (including retired reverse-engineered QC removals and legacy file naming) are considered historical and are documented in:
- `docs/datasets/jain/complete_history.md`

---

## References

- Jain et al. (2017) *PNAS*: Biophysical properties of the clinical-stage antibody landscape. DOI: `10.1073/pnas.1616408114`
- Sakhnini et al. (2025) *bioRxiv*: Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. DOI: `10.1101/2025.04.28.650927`
- Dunbar & Deane (2016) ANARCI. DOI: `10.1093/bioinformatics/btv552`

