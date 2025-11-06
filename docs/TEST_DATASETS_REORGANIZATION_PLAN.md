# Test Datasets Reorganization Plan

**Date:** 2025-11-05
**Branch:** leroy-jenkins/full-send
**Goal:** Clear, traceable, reproducible dataset organization for Jain (and eventually Harvey/Shehata)

---

## Problem: Confusing Mixing of Raw, Intermediate, and Fragment Files

**Current state:**
```
test_datasets/
├── [ROOT LEVEL] Raw Excel files, base CSVs, intermediate CSVs (MESSY MIX)
├── jain/           [SUBDIRECTORY] Fragments + curated datasets + OLD datasets
├── harvey/         [SUBDIRECTORY] Fragments only
└── shehata/        [SUBDIRECTORY] Fragments only
```

**Issues:**
1. Root has mix of raw sources AND intermediate files (jain_86_elisa_1.3.csv, jain_ELISA_ONLY_116.csv)
2. jain/ subdirectory has fragments PLUS canonical datasets PLUS old datasets
3. Unclear what's "source" vs "derived" vs "fragment" vs "deprecated"
4. No clear data flow from raw → processed → fragments

---

## Guiding Principles

1. **Clear Data Flow**: Raw → Intermediate → Final → Fragments
2. **Separation of Concerns**: Each directory has ONE purpose
3. **Traceability**: README.md in each directory explains what's there and how it was created
4. **Reproducibility**: All derived files can be regenerated from scripts
5. **No Redundancy**: Each file exists in exactly ONE location

---

## Proposed Structure for Jain Dataset

```
test_datasets/jain/
├── README.md                              # Master guide (already exists)
│
├── raw/                                   # 📁 NEVER TOUCH - Original sources
│   ├── jain-pnas.1616408114.sd01.xlsx    # Original sequences
│   ├── jain-pnas.1616408114.sd02.xlsx    # Original biophysical
│   ├── jain-pnas.1616408114.sd03.xlsx    # Original thermal stability
│   ├── Private_Jain2017_ELISA_indiv.xlsx # Private ELISA data
│   └── README.md                          # Documents source provenance
│
├── processed/                             # 📁 Derived datasets (reproducible)
│   ├── jain_sd01.csv                     # Converted from Excel (script: convert_jain_excel_to_csv.py)
│   ├── jain_sd02.csv                     # Converted from Excel
│   ├── jain_sd03.csv                     # Converted from Excel
│   ├── jain.csv                          # Base merged dataset (137 antibodies)
│   ├── jain_with_private_elisa_FULL.csv  # 137 with ELISA flags
│   ├── jain_ELISA_ONLY_116.csv           # ELISA QC filtered (removes ELISA 1-3)
│   ├── jain_86_elisa_1.3.csv             # Deprecated OLD method intermediate
│   └── README.md                          # Documents processing steps + scripts
│
├── canonical/                             # 📁 Final curated datasets for benchmarking
│   ├── jain_86_novo_parity.csv           # P5e-S2 canonical (86 antibodies, full-length)
│   ├── VH_only_jain_test_PARITY_86.csv   # OLD reverse-engineered (86 antibodies, VH fragment)
│   ├── VH_only_jain_test_FULL.csv        # OLD method (94 antibodies, VH fragment)
│   ├── VH_only_jain_test_QC_REMOVED.csv  # OLD method (91 antibodies, VH fragment)
│   └── README.md                          # Documents which to use when + expected results
│
├── fragments/                             # 📁 Region-specific extracts for ablation studies
│   ├── Full_jain.csv                     # Full VH+VL
│   ├── VH_only_jain.csv                  # VH only (137 antibodies)
│   ├── VL_only_jain.csv                  # VL only (137 antibodies)
│   ├── VH+VL_jain.csv                    # Concatenated VH+VL
│   ├── H-CDR1_jain.csv                   # Heavy CDR1
│   ├── H-CDR2_jain.csv                   # Heavy CDR2
│   ├── H-CDR3_jain.csv                   # Heavy CDR3
│   ├── H-CDRs_jain.csv                   # All heavy CDRs
│   ├── H-FWRs_jain.csv                   # All heavy FWRs
│   ├── L-CDR1_jain.csv                   # Light CDR1
│   ├── L-CDR2_jain.csv                   # Light CDR2
│   ├── L-CDR3_jain.csv                   # Light CDR3
│   ├── L-CDRs_jain.csv                   # All light CDRs
│   ├── L-FWRs_jain.csv                   # All light FWRs
│   ├── All-CDRs_jain.csv                 # All CDRs (H+L)
│   ├── All-FWRs_jain.csv                 # All FWRs (H+L)
│   ├── VH_only_jain_86_p5e_s2.csv        # P5e-S2 VH fragment (86 antibodies)
│   ├── VH_only_jain_86_p5e_s4.csv        # P5e-S4 VH fragment (86 antibodies)
│   └── README.md                          # Documents fragment extraction method
│
└── archive/                               # 📁 Deprecated/historical (if needed)
    └── README.md
```

---

## Data Flow Diagram

```
RAW (Excel)
    ↓ [preprocessing/jain/step1_convert_excel_to_csv.py]
PROCESSED (CSV base files)
    ↓ preprocessing/jain/step2_preprocess_p5e_s2.py
CANONICAL (86-antibody benchmarks)
    ↓ [scripts/fragmentation/extract_jain_fragments.py]
FRAGMENTS (region-specific extracts)
```

---

## Reorganization Steps

### Step 1: Create New Directory Structure

```bash
# Create new subdirectories
mkdir -p test_datasets/jain/raw
mkdir -p test_datasets/jain/processed
mkdir -p test_datasets/jain/canonical
mkdir -p test_datasets/jain/fragments
```

### Step 2: Move Raw Files (Root → jain/raw/)

**Files to move:**
```bash
mv test_datasets/jain-pnas.1616408114.sd01.xlsx test_datasets/jain/raw/
mv test_datasets/jain-pnas.1616408114.sd02.xlsx test_datasets/jain/raw/
mv test_datasets/jain-pnas.1616408114.sd03.xlsx test_datasets/jain/raw/
mv test_datasets/Private_Jain2017_ELISA_indiv.xlsx test_datasets/jain/raw/
```

**Create raw/README.md:**
```markdown
# Jain Dataset - Raw Source Files

**DO NOT MODIFY THESE FILES**

## Files

### Public Data (Jain et al. 2017 PNAS)
- `jain-pnas.1616408114.sd01.xlsx` - Supplementary Data 1: Sequences (137 antibodies)
- `jain-pnas.1616408114.sd02.xlsx` - Supplementary Data 2: Biophysical properties
- `jain-pnas.1616408114.sd03.xlsx` - Supplementary Data 3: Thermal stability

### Private Data
- `Private_Jain2017_ELISA_indiv.xlsx` - Private ELISA flags (6 antigens)

## Conversion

Convert to CSV using:
```bash
python3 preprocessing/jain/step1_convert_excel_to_csv.py
```
```

### Step 3: Move Processed Files (Root → jain/processed/)

**Files to move:**
```bash
mv test_datasets/jain_sd01.csv test_datasets/jain/processed/
mv test_datasets/jain_sd02.csv test_datasets/jain/processed/
mv test_datasets/jain_sd03.csv test_datasets/jain/processed/
mv test_datasets/jain.csv test_datasets/jain/processed/
mv test_datasets/jain_with_private_elisa_FULL.csv test_datasets/jain/processed/
mv test_datasets/jain_ELISA_ONLY_116.csv test_datasets/jain/processed/
mv test_datasets/jain_86_elisa_1.3.csv test_datasets/jain/processed/
```

**Create processed/README.md:**
```markdown
# Jain Dataset - Processed Files

All files derived from `raw/` using scripts in `preprocessing/jain/`.

## Conversion (Raw Excel → CSV)

**Script:** `preprocessing/jain/step1_convert_excel_to_csv.py`

- `jain_sd01.csv` ← jain-pnas.1616408114.sd01.xlsx
- `jain_sd02.csv` ← jain-pnas.1616408114.sd02.xlsx
- `jain_sd03.csv` ← jain-pnas.1616408114.sd03.xlsx

## Merging + ELISA Integration

**Script:** `preprocessing/jain/step2_preprocess_p5e_s2.py`

- `jain.csv` (137 antibodies) - Base merged dataset
- `jain_with_private_elisa_FULL.csv` (137 antibodies) - With ELISA flags added
- `jain_ELISA_ONLY_116.csv` (116 antibodies) - After ELISA QC (removes ELISA 1-3)

## Deprecated

- `jain_86_elisa_1.3.csv` - OLD intermediate file (consider archiving)
```

### Step 4: Organize Canonical Files (Keep in jain/, create canonical/)

**Files to move into jain/canonical/:**
```bash
mv test_datasets/jain/jain_86_novo_parity.csv test_datasets/jain/canonical/
mv test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv test_datasets/jain/canonical/
mv test_datasets/jain/canonical/VH_only_jain_test_FULL.csv test_datasets/jain/canonical/
mv test_datasets/jain/canonical/VH_only_jain_test_QC_REMOVED.csv test_datasets/jain/canonical/
```

**Create canonical/README.md:**
```markdown
# Jain Dataset - Canonical Benchmarks

Final curated datasets for reproducible benchmarking.

## Novo Nordisk Parity Datasets (86 antibodies each)

### 1. `jain_86_novo_parity.csv` (P5e-S2 Canonical) ✅ RECOMMENDED

- **Method:** ELISA filter → PSR reclassification → AC-SINS removal
- **Script:** `preprocessing/jain/step2_preprocess_p5e_s2.py`
- **Result:** [[40, 19], [10, 17]], 66.28%
- **Columns:** Full-length sequences + biophysical properties
- **Reproducibility:** 1 borderline antibody (nimotuzumab ~0.5) may flip

### 2. `VH_only_jain_test_PARITY_86.csv` (OLD Reverse-Engineered)

- **Method:** ELISA filter → VH length outliers → borderline removals
- **Script:** Legacy `preprocessing/process_jain.py` (removed; see git history)
- **Result:** [[40, 19], [10, 17]], 66.28% (deterministic)
- **Columns:** VH fragment only

## Intermediate (OLD Method)

- `VH_only_jain_test_FULL.csv` (94 antibodies) - After ELISA filter
- `VH_only_jain_test_QC_REMOVED.csv` (91 antibodies) - After length outlier removal

## Usage

```python
# Recommended for new benchmarks
df = pd.read_csv('test_datasets/jain/canonical/jain_86_novo_parity.csv')

# For deterministic reproducibility
df = pd.read_csv('test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv')
```
```

### Step 5: Organize Fragments (Already in jain/, create fragments/)

**Files to move into jain/fragments/:**
```bash
mv test_datasets/jain/All-CDRs_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/All-FWRs_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/Full_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/H-CDR1_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/H-CDR2_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/H-CDR3_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/H-CDRs_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/H-FWRs_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/L-CDR1_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/L-CDR2_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/L-CDR3_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/L-CDRs_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/L-FWRs_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/VH+VL_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/fragments/VH_only_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/VL_only_jain.csv test_datasets/jain/fragments/
mv test_datasets/jain/fragments/VH_only_jain_86_p5e_s2.csv test_datasets/jain/fragments/
mv test_datasets/jain/fragments/VH_only_jain_86_p5e_s4.csv test_datasets/jain/fragments/
```

**Create fragments/README.md:**
```markdown
# Jain Dataset - Fragments

Region-specific extracts for ablation studies and targeted training.

## Full-Length Sequences

- `Full_jain.csv` - Complete VH+VL sequences
- `VH_only_jain.csv` - VH sequences only (137 antibodies)
- `VL_only_jain.csv` - VL sequences only (137 antibodies)
- `VH+VL_jain.csv` - Concatenated VH+VL

## Heavy Chain Regions

- `H-CDR1_jain.csv` - Heavy CDR1
- `H-CDR2_jain.csv` - Heavy CDR2
- `H-CDR3_jain.csv` - Heavy CDR3
- `H-CDRs_jain.csv` - All heavy CDRs
- `H-FWRs_jain.csv` - All heavy FWRs

## Light Chain Regions

- `L-CDR1_jain.csv` - Light CDR1
- `L-CDR2_jain.csv` - Light CDR2
- `L-CDR3_jain.csv` - Light CDR3
- `L-CDRs_jain.csv` - All light CDRs
- `L-FWRs_jain.csv` - All light FWRs

## Combined

- `All-CDRs_jain.csv` - All CDRs (H+L)
- `All-FWRs_jain.csv` - All FWRs (H+L)

## 86-Antibody VH Fragments (Novo Parity)

- `VH_only_jain_86_p5e_s2.csv` - P5e-S2 VH fragment
- `VH_only_jain_86_p5e_s4.csv` - P5e-S4 VH fragment

## Extraction

Fragments generated using:
```bash
python3 scripts/fragmentation/extract_jain_fragments.py
```
```

### Step 6: Update Master README

**Update test_datasets/jain/README.md** to reflect new structure:
```markdown
# Jain Test Dataset

**Source**: Jain et al. (2017) PNAS - Biophysical properties of the clinical-stage antibody landscape

---

## Directory Structure

```
jain/
├── raw/               📁 Original Excel files (NEVER MODIFY)
├── processed/         📁 Converted CSVs + intermediate datasets
├── canonical/         📁 Final benchmarks for Novo parity
├── fragments/         📁 Region-specific extracts (CDRs, FWRs, etc.)
└── README.md          📄 This file
```

## Quick Start

**For benchmarking with Novo Nordisk parity:**
```python
import pandas as pd

# Recommended: P5e-S2 canonical
df = pd.read_csv('test_datasets/jain/canonical/jain_86_novo_parity.csv')

# Expected: [[40, 19], [10, 17]], 66.28%
```

See `canonical/README.md` for details on methodology.

---

## Data Flow

```
raw/ (Excel)
  ↓ [convert_jain_excel_to_csv.py]
processed/ (CSV base files)
  ↓ [preprocess_jain_p5e_s2.py]
canonical/ (86-antibody benchmarks)
  ↓ [extract_jain_fragments.py]
fragments/ (region-specific)
```

---

**For full documentation, see:** `JAIN_COMPLETE_GUIDE.md` in repo root.
```

### Step 7: Verify Scripts Still Work

**After reorganization, verify:**

1. **Conversion script** can find raw files:
   ```bash
   # Check if preprocessing/jain/step1_convert_excel_to_csv.py references correct paths
   # Update paths from test_datasets/*.xlsx → test_datasets/jain/raw/*.xlsx
   ```

2. **Preprocessing scripts** can find input files:
   ```bash
   # Check preprocessing/jain/step2_preprocess_p5e_s2.py
   # Update paths to point to processed/ directory
   ```

3. **Test scripts** can find canonical files:
   ```bash
   # Check scripts/testing/test_jain_novo_parity.py
   # Update paths to test_datasets/jain/canonical/
   ```

### Step 8: Run Tests to Confirm

```bash
# Test P5e-S2 parity
python3 test.py --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/canonical/jain_86_novo_parity.csv

# Test OLD parity
python3 test.py --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv
```

Both should give [[40, 19], [10, 17]].

---

## Final Directory Tree (After Reorganization)

```
test_datasets/
├── harvey/
│   └── [fragments...]
├── jain/
│   ├── README.md
│   ├── raw/
│   │   ├── README.md
│   │   ├── jain-pnas.1616408114.sd01.xlsx
│   │   ├── jain-pnas.1616408114.sd02.xlsx
│   │   ├── jain-pnas.1616408114.sd03.xlsx
│   │   └── Private_Jain2017_ELISA_indiv.xlsx
│   ├── processed/
│   │   ├── README.md
│   │   ├── jain_sd01.csv
│   │   ├── jain_sd02.csv
│   │   ├── jain_sd03.csv
│   │   ├── jain.csv
│   │   ├── jain_with_private_elisa_FULL.csv
│   │   ├── jain_ELISA_ONLY_116.csv
│   │   └── jain_86_elisa_1.3.csv (deprecated)
│   ├── canonical/
│   │   ├── README.md
│   │   ├── jain_86_novo_parity.csv
│   │   ├── VH_only_jain_test_PARITY_86.csv
│   │   ├── VH_only_jain_test_FULL.csv
│   │   └── VH_only_jain_test_QC_REMOVED.csv
│   └── fragments/
│       ├── README.md
│       ├── [18 fragment files...]
│       ├── VH_only_jain_86_p5e_s2.csv
│       └── VH_only_jain_86_p5e_s4.csv
├── shehata/
│   └── [fragments...]
├── harvey.csv
├── harvey_high.csv
├── harvey_low.csv
├── shehata.csv
├── shehata-mmc2.xlsx
├── shehata-mmc3.xlsx
├── shehata-mmc4.xlsx
└── shehata-mmc5.xlsx
```

---

## Scripts to Update

### 1. `preprocessing/jain/step1_convert_excel_to_csv.py`
- **Change:** Update paths to read from `test_datasets/jain/raw/`
- **Change:** Update paths to write to `test_datasets/jain/processed/`

### 2. `preprocessing/jain/step2_preprocess_p5e_s2.py`
- **Change:** Read from `test_datasets/jain/processed/`
- **Change:** Write to `test_datasets/jain/canonical/`

### 3. Legacy `preprocessing/process_jain.py` (old method)
- **Change:** Read from `test_datasets/jain/processed/`
- **Change:** Write to `test_datasets/jain/canonical/`

### 4. `scripts/testing/test_jain_novo_parity.py`
- **Change:** Read from `test_datasets/jain/canonical/`

### 5. `scripts/fragmentation/extract_jain_fragments.py` (if exists)
- **Change:** Read from `test_datasets/jain/canonical/`
- **Change:** Write to `test_datasets/jain/fragments/`

---

## Benefits of This Organization

✅ **Clear data lineage:** raw → processed → canonical → fragments
✅ **No confusion:** Each directory has ONE purpose
✅ **Reproducible:** All files can be regenerated from scripts
✅ **Traceable:** README in each directory documents provenance
✅ **Clean root:** test_datasets/ root only has top-level merged files
✅ **Scalable:** Same structure can be applied to Harvey and Shehata

---

## Ready to Execute?

1. Review this plan
2. Execute steps 1-6 to reorganize files
3. Update scripts in step 7
4. Test in step 8
5. Commit with message: "Reorganize test_datasets/jain/ for clarity and reproducibility"

**Estimated time:** 30 minutes
**Risk:** Low (moving files, not deleting)
**Benefit:** Massive improvement in clarity and maintainability
