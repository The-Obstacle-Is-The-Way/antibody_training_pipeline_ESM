# Jain Test Dataset

**Source**: Jain et al. (2017) PNAS - Biophysical properties of the clinical-stage antibody landscape

---

## Canonical Novo Nordisk Parity Datasets

Two curated 86-antibody subsets reproduce Novo Nordisk’s confusion matrix:

### 1. `VH_only_jain_test_PARITY_86.csv` (OLD simple QC)
- **Method:** ELISA filter → remove VH length outliers → remove 5 borderline antibodies
- **Result:** [[40, 19], [10, 17]] (deterministic)
- **Use when:** You need a guaranteed parity benchmark

### 2. `jain_86_novo_parity.csv` / `VH_only_jain_86_p5e_s2.csv` (P5e-S2 canonical)
- **Method:** ELISA filter → PSR/Tm/clinical reclassification (5 antibodies) → PSR + AC-SINS removal (30 antibodies)
- **Result:** [[40, 19], [10, 17]] when nimotuzumab scores <0.5. Because its predicted probability sits ≈0.5, reruns may occasionally give [[39, 20], [10, 17]]. Use the stored `prediction` column or OLD dataset when you need strict determinism.
- **Full provenance:** `experiments/novo_parity/`

---

## Foundation Datasets (Root Level)

**Source Data**:
- `jain-pnas.1616408114.sd01.xlsx` - Original sequences
- `jain-pnas.1616408114.sd02.xlsx` - Original biophysical properties
- `jain-pnas.1616408114.sd03.xlsx` - Original thermal stability

**Converted**:
- `jain_sd01.csv` - Sequences (converted from Excel)
- `jain_sd02.csv` - Biophysical properties (converted from Excel)
- `jain_sd03.csv` - Thermal stability (converted from Excel)

**Intermediate Steps**:
- `jain_with_private_elisa_FULL.csv` - 137 antibodies with private ELISA data
- `jain_ELISA_ONLY_116.csv` - 116 antibodies after ELISA QC (removes ELISA 1-3)

---

## Archive

`archive/` contains historical or deprecated artifacts:
- `jain_116_qc_candidates.csv`, `jain_ELISA_ONLY_116_with_zscores.csv`
- `archive/legacy_reverse_engineered/` – duplicate copies of the OLD datasets (kept for history)
- `archive/legacy_total_flags_methodology/` – ❌ incorrect total_flags methodology (do not use)

---

## Usage

**For training/testing with Novo Nordisk parity**:
```python
import pandas as pd

# Load canonical 86-antibody dataset
df = pd.read_csv('test_datasets/jain/jain_86_novo_parity.csv')

# Expected confusion matrix: [[40, 19], [10, 17]]
# Expected accuracy: 66.28%
```

**For sensitivity analysis**:
- P5e-S4 (Tm tiebreaker): `experiments/novo_parity/datasets/jain_86_p5e_s4.csv`
  - Produces [[39, 20], [10, 17]] with the OLD model (off by 1 FP)

---

**Last Updated**: November 5, 2025
**Branch**: `clean-jain`
