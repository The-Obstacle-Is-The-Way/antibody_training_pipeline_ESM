# Jain Test Dataset

**Source**: Jain et al. (2017) PNAS - Biophysical properties of the clinical-stage antibody landscape

---

## Canonical Novo Nordisk Parity Dataset

**`jain_86_novo_parity.csv`** - EXACT MATCH to Novo Nordisk confusion matrix

- **Antibodies**: 86 total (59 specific / 27 non-specific)
- **Confusion Matrix**: [[40, 19], [10, 17]]
- **Accuracy**: 66.28%
- **Method**: P5e-S2 (PSR reclassification + PSR/AC-SINS removal)

**Pipeline**:
1. Start: 137 antibodies (jain_with_private_elisa_FULL.csv)
2. Remove ELISA 1-3 (mild aggregators) → 116 antibodies
3. Reclassify 5 antibodies specific→non-specific:
   - 3 by PSR>0.4: bimagrumab, bavituximab, ganitumab
   - eldelumab (extreme Tm outlier)
   - infliximab (clinical withdrawn)
4. Remove 30 by PSR primary, AC-SINS tiebreaker → 86 antibodies

**Full Provenance**: See `experiments/novo_parity/` for complete reverse engineering history

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

**`legacy_reverse_engineered/`** - Old failed reverse engineering attempts (pre-P5e)

**`legacy_total_flags_methodology/`** - Old total_flags approach (superseded)

**`archive/`** - Intermediate analysis files from reverse engineering process

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
  - Same exact match, different tiebreaker for PSR=0 antibodies

---

**Last Updated**: November 4, 2025
**Branch**: `ray/novo-parity-experiments`
