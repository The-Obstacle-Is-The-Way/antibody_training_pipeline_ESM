# Jain Files Cleanup Plan

**Date**: November 4, 2025
**Branch**: `novo-parity-exp-cleaned`
**Goal**: Establish clean SSOT for Novo parity while preserving experiments/

---

## Current State Analysis

### ✅ KEEP AS-IS (Foundational Files)

**test_datasets/ (root level)**
- ✅ `jain-pnas.1616408114.sd01.xlsx` - Original Jain 2017 data (sequences)
- ✅ `jain-pnas.1616408114.sd02.xlsx` - Original Jain 2017 data (biophysical)
- ✅ `jain-pnas.1616408114.sd03.xlsx` - Original Jain 2017 data (thermal stability)
- ✅ `jain_sd01.csv` - Converted sequences
- ✅ `jain_sd02.csv` - Converted biophysical
- ✅ `jain_sd03.csv` - Converted thermal stability
- ✅ `jain_with_private_elisa_FULL.csv` - 137 antibodies with private ELISA (foundation)
- ✅ `jain_ELISA_ONLY_116.csv` - 116 antibodies after ELISA QC (foundation)

**experiments/novo_parity/**
- ✅ KEEP ENTIRE FOLDER - This is the permutation testing archive
- Contains: all P5* variants, permutation results, audit trails, scripts

---

## ❌ PROBLEMS TO FIX

### Problem 1: No Clean P5e-S2 Dataset in test_datasets/jain/

**Issue**: The canonical Novo parity dataset (P5e-S2) only exists in `experiments/novo_parity/datasets/`

**Fix**: Copy final datasets to `test_datasets/jain/` with clean names:
```
experiments/novo_parity/datasets/jain_86_p5e_s2.csv
  → test_datasets/jain/jain_86_novo_parity.csv (or similar)

experiments/novo_parity/datasets/jain_86_p5e_s4.csv
  → test_datasets/jain/jain_86_novo_parity_sensitivity.csv (optional)
```

### Problem 2: preprocessing/process_jain.py Uses OLD Methodology

**Current**:
```python
# preprocessing/process_jain.py (OUTDATED)
# Input: 94 antibodies
# QC Removals: 8 antibodies (VH length outliers + biology/clinical)
# Output: 86 antibodies (WRONG 86!)
```

**Should be**:
```python
# preprocessing/process_jain.py (CORRECT P5e-S2)
# Input: 137 antibodies (jain_with_private_elisa_FULL.csv)
# Step 1: Remove ELISA 1-3 (mild) → 116 antibodies
# Step 2: Reclassify 5 (3 PSR>0.4 + eldelumab + infliximab) → 89/27
# Step 3: Remove 30 by PSR+AC-SINS → 86 antibodies (59/27)
# Output: jain_86_novo_parity.csv (EXACT MATCH)
```

**Action**: Rewrite `preprocessing/process_jain.py` with P5e-S2 logic

### Problem 3: scripts/testing/test_jain_novo_parity.py References Wrong Path

**Current**:
```python
# Line 37
df = pd.read_csv('test_datasets/jain/VH_only_jain_test_PARITY_86.csv')
```

**Should be**:
```python
df = pd.read_csv('test_datasets/jain/jain_86_novo_parity.csv')
```

**Action**: Update to use new canonical dataset

### Problem 4: Old Backup Scripts in scripts/conversion/

**Files to archive**:
- `scripts/conversion/convert_jain_excel_to_csv_OLD_BACKUP.py`
- `scripts/conversion/convert_jain_excel_to_csv_TOTAL_FLAGS_WRONG.py`

**Action**: Move to `scripts/archive/conversion/` or delete (they're in git history)

### Problem 5: Legacy Datasets in test_datasets/jain/

**Already organized (good!)**:
- ✅ `test_datasets/jain/legacy_reverse_engineered/` - Old failed attempts
- ✅ `test_datasets/jain/legacy_total_flags_methodology/` - Old total_flags approach

**Additional files to archive**:
- `test_datasets/jain/jain_116_qc_candidates.csv` - intermediate analysis
- `test_datasets/jain/jain_ELISA_ONLY_116_with_zscores.csv` - intermediate analysis

---

## RECOMMENDED STRUCTURE (After Cleanup)

```
test_datasets/
├── jain/
│   ├── jain_86_novo_parity.csv ✅ NEW - P5e-S2 canonical dataset
│   ├── jain_86_novo_parity_sensitivity.csv (OPTIONAL - P5e-S4)
│   ├── legacy_reverse_engineered/ (archive - old failed attempts)
│   ├── legacy_total_flags_methodology/ (archive - old total_flags approach)
│   └── archive/ ✅ NEW
│       ├── jain_116_qc_candidates.csv
│       └── jain_ELISA_ONLY_116_with_zscores.csv
├── jain_ELISA_ONLY_116.csv ✅ KEEP (foundation - 116 antibody starting point)
├── jain_with_private_elisa_FULL.csv ✅ KEEP (foundation - full 137)
├── jain_sd01.csv ✅ KEEP (converted from Excel)
├── jain_sd02.csv ✅ KEEP (converted from Excel)
├── jain_sd03.csv ✅ KEEP (converted from Excel)
└── jain-*.xlsx ✅ KEEP (original data)

preprocessing/
├── process_jain.py ✅ REWRITE with P5e-S2 methodology
├── process_boughter.py ✅ KEEP
├── process_harvey.py ✅ KEEP
└── process_shehata.py ✅ KEEP

scripts/
├── conversion/
│   ├── convert_jain_excel_to_csv.py ✅ KEEP (current correct version)
│   └── archive/ ✅ NEW
│       ├── convert_jain_excel_to_csv_OLD_BACKUP.py
│       └── convert_jain_excel_to_csv_TOTAL_FLAGS_WRONG.py
├── testing/
│   └── test_jain_novo_parity.py ✅ UPDATE (use new dataset path)
└── ...

experiments/
└── novo_parity/ ✅ KEEP ENTIRE FOLDER (permutation archive)
    ├── datasets/ (all P5* variants - keep for provenance)
    ├── scripts/ (permutation testing scripts)
    ├── results/ (audit trails)
    └── *.md (documentation)
```

---

## EXECUTION PLAN

### Step 1: Copy Final Dataset to test_datasets/jain/

```bash
# Copy P5e-S2 as canonical Novo parity dataset
cp experiments/novo_parity/datasets/jain_86_p5e_s2.csv \
   test_datasets/jain/jain_86_novo_parity.csv

# Optional: Copy P5e-S4 as sensitivity
cp experiments/novo_parity/datasets/jain_86_p5e_s4.csv \
   test_datasets/jain/jain_86_novo_parity_sensitivity.csv
```

### Step 2: Archive Intermediate Analysis Files

```bash
# Create archive folder
mkdir -p test_datasets/jain/archive

# Move intermediate files
git mv test_datasets/jain/jain_116_qc_candidates.csv test_datasets/jain/archive/
git mv test_datasets/jain/jain_ELISA_ONLY_116_with_zscores.csv test_datasets/jain/archive/
```

### Step 3: Archive Old Backup Scripts

```bash
# Create archive folder
mkdir -p scripts/archive/conversion

# Move old backups
git mv scripts/conversion/convert_jain_excel_to_csv_OLD_BACKUP.py \
       scripts/archive/conversion/
git mv scripts/conversion/convert_jain_excel_to_csv_TOTAL_FLAGS_WRONG.py \
       scripts/archive/conversion/
```

### Step 4: Rewrite preprocessing/process_jain.py

**Option A**: Rewrite completely with P5e-S2 logic
**Option B**: Create new `preprocessing/process_jain_novo_parity.py` and keep old one archived

**Recommended**: Rewrite with P5e-S2, move old one to backup

```bash
# Backup old version first
cp preprocessing/process_jain.py preprocessing/process_jain_OLD_94to86.py.bak

# Then rewrite process_jain.py with correct P5e-S2 logic
```

### Step 5: Update scripts/testing/test_jain_novo_parity.py

Change dataset path from:
```python
df = pd.read_csv('test_datasets/jain/VH_only_jain_test_PARITY_86.csv')
```

To:
```python
df = pd.read_csv('test_datasets/jain/jain_86_novo_parity.csv')
```

### Step 6: Create README in test_datasets/jain/

```bash
# Create README explaining structure
cat > test_datasets/jain/README.md << 'EOF'
# Jain Test Dataset

## Canonical Novo Parity Dataset

- **jain_86_novo_parity.csv** - P5e-S2 exact match (recommended)
  - 86 antibodies (59 specific / 27 non-specific)
  - Confusion matrix: [[40, 19], [10, 17]]
  - Accuracy: 66.28%
  - Method: PSR reclassification + PSR/AC-SINS removal

- **jain_86_novo_parity_sensitivity.csv** - P5e-S4 exact match (sensitivity)
  - Same result, different tiebreaker (Tm instead of AC-SINS)

## Archive

- **legacy_reverse_engineered/** - Old failed reverse engineering attempts
- **legacy_total_flags_methodology/** - Old total_flags approach
- **archive/** - Intermediate analysis files

## Provenance

Full permutation testing history: `experiments/novo_parity/`
EOF
```

---

## QUESTIONS FOR USER

1. **Dataset naming**: Do you prefer:
   - `jain_86_novo_parity.csv` (descriptive)
   - `jain_86.csv` (short, since it's THE canonical one)
   - Something else?

2. **preprocessing/process_jain.py**: Should we:
   - Rewrite completely (move old to backup)?
   - Create new file `process_jain_novo_parity.py`?

3. **P5e-S4 sensitivity**: Should we include it in test_datasets/jain/?
   - Pro: Shows robustness (2 methods, same result)
   - Con: May confuse users ("which one to use?")

4. **Old backup scripts**: Delete or archive?
   - They're in git history, so safe to delete
   - Or move to `scripts/archive/`

---

## READY TO EXECUTE?

Once you confirm:
1. Naming preferences
2. Whether to include P5e-S4
3. How to handle old preprocessing script

I'll execute the full cleanup with git history preserved.

---

**Status**: PLAN READY
**Next**: User review and confirmation
