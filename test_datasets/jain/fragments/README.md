# Jain Dataset - Fragments

Region-specific extracts for ablation studies, CDR-focused training, and targeted analysis.

All fragments are derived from the **ELISA-based SSOT dataset** (`../processed/jain_with_private_elisa_FULL.csv`) and share the same labeling schema:

- 137 antibodies total
- 94 specific (label `0`)
- 22 non-specific (label `1`)
- 21 mild / held-out (label `NaN`)

> ℹ️ The previous 67/27/43 fragments that used the paper-style `flags_total` labeling have been removed from the working tree (see git commit `09d6121` if historical reference is ever required). Do not use those legacy artifacts for training or validation.

---

## Full-Length Sequences (137 antibodies)

- `Full_jain.csv` - Complete VH+VL sequences with all metadata
- `VH_only_jain.csv` - VH sequences only
- `VL_only_jain.csv` - VL sequences only
- `VH+VL_jain.csv` - Concatenated VH+VL

**Use when:** Training on full sequences or comparing VH vs VL vs VH+VL performance.

---

## Heavy Chain Regions (137 antibodies each)

- `H-CDR1_jain.csv` - Heavy chain CDR1 only
- `H-CDR2_jain.csv` - Heavy chain CDR2 only
- `H-CDR3_jain.csv` - Heavy chain CDR3 only
- `H-CDRs_jain.csv` - All heavy CDRs concatenated
- `H-FWRs_jain.csv` - All heavy framework regions concatenated

**Use when:** Studying CDR3 importance, or ablating specific regions.

---

## Light Chain Regions (137 antibodies each)

- `L-CDR1_jain.csv` - Light chain CDR1 only
- `L-CDR2_jain.csv` - Light chain CDR2 only
- `L-CDR3_jain.csv` - Light chain CDR3 only
- `L-CDRs_jain.csv` - All light CDRs concatenated
- `L-FWRs_jain.csv` - All light framework regions concatenated

**Use when:** Studying light chain contribution to polyreactivity.

---

## Combined Regions (137 antibodies each)

- `All-CDRs_jain.csv` - All CDRs (heavy + light)
- `All-FWRs_jain.csv` - All framework regions (heavy + light)

**Use when:** Testing CDR-only vs FWR-only models.

---

## 86-Antibody VH Fragments (Novo Parity Benchmarks)

- `VH_only_jain_86_p5e_s2.csv` - P5e-S2 canonical method, VH only
  - Same 86 antibodies as `../canonical/jain_86_novo_parity.csv`
  - VH fragment for faster embedding extraction

- `VH_only_jain_86_p5e_s4.csv` - P5e-S4 variant (Tm tiebreaker)
  - Alternative 86-antibody set using Tm instead of AC-SINS for tiebreaking
  - Results: [[39, 20], [10, 17]] (off by 1 FP with OLD model)

**Use when:** You need the benchmark datasets but only want VH embeddings.

---

## Fragment Extraction

Fragments are generated using ANARCI for numbering and region extraction:

```python
# Pseudocode for fragment extraction
from anarci import anarci

def extract_cdr3(sequence):
    results = anarci([('seq', sequence)], scheme='imgt')
    numbering = results[0][0][0]

    # IMGT CDR3: positions 105-117
    cdr3 = ''.join([aa for pos, aa in numbering if 105 <= pos <= 117])
    return cdr3
```

**Script:** `preprocessing/jain/step3_extract_fragments.py`

See `manifest.yml` for exact provenance (source file hash, script version, expected counts).

---

## Usage Example: CDR3-Only Training

```python
import pandas as pd

# Load CDR3-only dataset
df = pd.read_csv('test_datasets/jain/fragments/H-CDR3_jain.csv')

# Train model on CDR3 only
sequences = df['sequence'].tolist()  # CDR3 sequences
labels = df['label'].values

# Compare to full VH performance
df_vh = pd.read_csv('test_datasets/jain/fragments/VH_only_jain.csv')
```

---

## Ablation Study Example

```python
# Test importance of each region
regions = [
    'H-CDR1_jain.csv',
    'H-CDR2_jain.csv',
    'H-CDR3_jain.csv',
    'H-FWRs_jain.csv'
]

for region_file in regions:
    df = pd.read_csv(f'test_datasets/jain/fragments/{region_file}')
    # Train and evaluate...
    # Compare accuracies to understand which regions matter most
```

---

## Notes

- All fragments maintain the same antibody IDs and **ELISA-based labels** as the SSOT dataset
- Mild antibodies (`NaN`) are included in the 137-antibody fragments but excluded from the 116-antibody ELISA test set and the 86-antibody parity set
- Provenance, expected label counts, and script references are recorded in `manifest.yml`
- Legacy fragments (67/27/43 distribution) were removed; restore from git commit `09d6121` if needed for historical comparison only
- For 86-antibody benchmark fragments, use `VH_only_jain_86_p5e_s2.csv`

---

**See:** `JAIN_COMPLETE_GUIDE.md` (repo root) for complete documentation
