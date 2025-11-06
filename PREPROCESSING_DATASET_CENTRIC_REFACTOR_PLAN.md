# Preprocessing Dataset-Centric Refactor Plan (Option 2)

**Date:** 2025-11-05
**Branch:** leroy-jenkins/full-send (main synced as backup at a868338)
**Status:** 🟡 AWAITING SENIOR APPROVAL
**Author:** Claude Code (Ray's request)

---

## Executive Summary

**Goal:** Reorganize preprocessing to follow industry-standard **dataset-centric pattern** where ALL preprocessing for each dataset lives in ONE subdirectory.

**Current State:** INCONSISTENT
- Boughter: Dataset-centric (preprocessing/boughter/ - 7 scripts)
- Harvey/Jain/Shehata: SPLIT between scripts/conversion/ and preprocessing/

**Target State:** CONSISTENT - All datasets follow Boughter pattern

**Impact:**
- **Code:** Move 6 files (3 conversion scripts + 3 preprocessing scripts)
- **Structure:** Create 3 new dataset subdirectories + 4 __init__.py files
- **Documentation:** Create 4 new READMEs + update 30+ existing docs/READMEs
- **Imports:** Fix 2 validation scripts (clean package imports)
- **Messages:** Update 3 script output messages
- **Total files affected:** ~45 files (6 moves, 8 new, 30+ updates)
- **NO changes to data files or test datasets**

**Risk Level:** LOW
- ✅ Backup on main branch (a868338)
- ✅ Git history preserved (all moves via git mv)
- ✅ No code logic changes, only file locations

---

## Rationale: Why Dataset-Centric?

### Industry Standard Pattern

**Professional ML repositories organize by DATASET, not by PHASE:**

1. **HuggingFace datasets:** Each dataset owns its preprocessing
   ```
   datasets/mnist/mnist.py          # All MNIST preprocessing
   datasets/cifar10/cifar10.py      # All CIFAR-10 preprocessing
   ```

2. **TensorFlow datasets:** Dataset-centric organization
   ```
   tensorflow_datasets/image/mnist.py
   tensorflow_datasets/text/imdb_reviews.py
   ```

3. **PyTorch torchvision:** Each dataset = one file with all preprocessing
   ```
   torchvision/datasets/mnist.py
   torchvision/datasets/cifar.py
   ```

4. **scikit-learn:** Dataset loaders own their preprocessing
   ```
   sklearn/datasets/olivetti_faces.py
   sklearn/datasets/lfw.py
   ```

### Core Principle: Dataset Ownership

**"Everything for Dataset X lives in preprocessing/X/"**

**Benefits:**
1. **Discoverability:** New contributor asks "How do I preprocess Harvey?" → ONE directory
2. **Maintainability:** Bug in Jain preprocessing → ONE directory to investigate
3. **Documentation:** Each dataset has README with COMPLETE pipeline
4. **Consistency:** All datasets follow same pattern (no special cases)
5. **Isolation:** Changes to Harvey don't touch Jain/Shehata directories

**Current Problem:**
- Harvey preprocessing is SPLIT: scripts/conversion/convert_harvey_csvs.py + preprocessing/process_harvey.py
- New contributor must search TWO directories to understand Harvey pipeline
- Inconsistent with Boughter pattern (which works well!)

---

## Current State Analysis

### ✅ What Works (Keep This Pattern)

**Boughter: Dataset-Centric Structure**
```
preprocessing/boughter/
├── README.md                       # Complete pipeline documentation
├── stage1_dna_translation.py       # FASTA → CSV
├── stage2_stage3_annotation_qc.py  # Annotation + fragments
├── stage4_additional_qc.py         # Additional QC
├── validate_stage1.py              # Validation
├── validate_stage4.py
└── validate_stages2_3.py
```

**Why this works:**
- All Boughter preprocessing in ONE place
- README documents full pipeline
- Self-contained and discoverable

---

### ❌ What's Inconsistent (Fix This)

**Harvey: Split Between Two Directories**
```
scripts/conversion/
└── convert_harvey_csvs.py          # Phase 1: raw → processed

preprocessing/
└── process_harvey.py               # Phase 2: processed → fragments
```

**Jain: Split Between Two Directories**
```
scripts/conversion/
└── convert_jain_excel_to_csv.py    # Phase 1: Excel → CSV

preprocessing/
└── preprocess_jain_p5e_s2.py       # Phase 2: CSV → canonical
```

**Shehata: Split Between Two Directories**
```
scripts/conversion/
└── convert_shehata_excel_to_csv.py # Phase 1: Excel → CSV

preprocessing/
└── process_shehata.py              # Phase 2: CSV → fragments
```

**Problems:**
- Inconsistent with Boughter pattern
- Dataset preprocessing split across directories
- Harder to discover full pipeline
- Conversion scripts grouped by "type" not by "dataset"

---

## Target State: Consistent Dataset-Centric Structure

### Final Directory Structure

```
preprocessing/
├── README.md                       # NEW: Overview of all datasets
├── boughter/                       # ✅ Already correct
│   ├── README.md
│   ├── stage1_dna_translation.py
│   ├── stage2_stage3_annotation_qc.py
│   ├── stage4_additional_qc.py
│   ├── validate_stage1.py
│   ├── validate_stage4.py
│   └── validate_stages2_3.py
├── harvey/                         # 🆕 NEW DIRECTORY
│   ├── README.md                   # NEW: Harvey pipeline docs
│   ├── step1_convert_raw_csvs.py  # MOVED from scripts/conversion/convert_harvey_csvs.py
│   └── step2_extract_fragments.py # MOVED from preprocessing/process_harvey.py
├── jain/                           # 🆕 NEW DIRECTORY
│   ├── README.md                   # NEW: Jain pipeline docs
│   ├── step1_convert_excel_to_csv.py # MOVED from scripts/conversion/convert_jain_excel_to_csv.py
│   └── step2_preprocess_p5e_s2.py    # MOVED from preprocessing/preprocess_jain_p5e_s2.py
└── shehata/                        # 🆕 NEW DIRECTORY
    ├── README.md                   # NEW: Shehata pipeline docs
    ├── step1_convert_excel_to_csv.py # MOVED from scripts/conversion/convert_shehata_excel_to_csv.py
    └── step2_extract_fragments.py    # MOVED from preprocessing/process_shehata.py
```

### scripts/ Directory (After Refactor)

```
scripts/
├── analysis/                       # ✅ Keep as-is
│   ├── README.md
│   ├── analyze_threshold_optimization.py
│   ├── compare_jain_methodologies.py
│   ├── rethreshold_harvey.py
│   └── zscore_jain_116_outliers.py
├── conversion/                     # ⚠️  EMPTY (delete or repurpose)
│   ├── README.md                   # UPDATE: Point to preprocessing/
│   └── legacy/                     # ✅ Keep legacy scripts
│       ├── README.md
│       ├── convert_jain_excel_to_csv_OLD_BACKUP.py
│       └── convert_jain_excel_to_csv_TOTAL_FLAGS_WRONG.py
├── testing/                        # ✅ Keep as-is
│   ├── README.md
│   ├── demo_assay_specific_thresholds.py
│   ├── test_harvey_psr_threshold.py
│   └── test_jain_novo_parity.py
├── training/                       # ✅ Keep as-is
│   └── train_hyperparameter_sweep.py
└── validation/                     # ⚠️  UPDATE: Fix imports
    ├── README.md
    ├── audit_boughter_training_qc.py
    ├── validate_fragments.py
    ├── validate_jain_conversion.py    # UPDATE: Fix import path
    ├── validate_shehata_conversion.py # UPDATE: Fix import path
    └── verify_novo_parity.py
```

---

## Detailed Execution Plan

### Phase 1: Create New Dataset Directories (3 directories)

**Commands:**
```bash
mkdir -p preprocessing/harvey
mkdir -p preprocessing/jain
mkdir -p preprocessing/shehata
```

**Verification:**
```bash
ls -la preprocessing/harvey preprocessing/jain preprocessing/shehata
# Expected: Each directory exists
```

---

### Phase 2: Move Conversion Scripts (3 files)

**2.1 Move Harvey Conversion Script**
```bash
git mv scripts/conversion/convert_harvey_csvs.py preprocessing/harvey/step1_convert_raw_csvs.py
```

**Rationale for rename:**
- `convert_harvey_csvs.py` → `step1_convert_raw_csvs.py`
- Makes pipeline sequence explicit (step1, step2)
- Consistent with Boughter pattern (stage1, stage2, stage3)

---

**2.2 Move Jain Conversion Script**
```bash
git mv scripts/conversion/convert_jain_excel_to_csv.py preprocessing/jain/step1_convert_excel_to_csv.py
```

**Rationale for rename:**
- `convert_jain_excel_to_csv.py` → `step1_convert_excel_to_csv.py`
- Shorter, clearer (dataset context obvious from directory)
- Explicit step numbering

---

**2.3 Move Shehata Conversion Script**
```bash
git mv scripts/conversion/convert_shehata_excel_to_csv.py preprocessing/shehata/step1_convert_excel_to_csv.py
```

**Rationale for rename:**
- `convert_shehata_excel_to_csv.py` → `step1_convert_excel_to_csv.py`
- Consistent with Jain pattern

---

### Phase 3: Move Preprocessing Scripts (3 files)

**3.1 Move Harvey Processing Script**
```bash
git mv preprocessing/process_harvey.py preprocessing/harvey/step2_extract_fragments.py
```

**Rationale for rename:**
- `process_harvey.py` → `step2_extract_fragments.py`
- Describes what it does (extract fragments)
- Sequential with step1

---

**3.2 Move Jain Processing Script**
```bash
git mv preprocessing/preprocess_jain_p5e_s2.py preprocessing/jain/step2_preprocess_p5e_s2.py
```

**Rationale for rename:**
- `preprocess_jain_p5e_s2.py` → `step2_preprocess_p5e_s2.py`
- Keep P5e-S2 methodology name (important!)
- Sequential with step1

---

**3.3 Move Shehata Processing Script**
```bash
git mv preprocessing/process_shehata.py preprocessing/shehata/step2_extract_fragments.py
```

**Rationale for rename:**
- `process_shehata.py` → `step2_extract_fragments.py`
- Consistent with Harvey pattern

---

### Phase 4: Create __init__.py Files for Package Imports

**Decision:** Use proper Python package structure (cleaner than sys.path manipulation)

**4.1 Create preprocessing/__init__.py**
```bash
touch preprocessing/__init__.py
```

**Content:**
```python
"""Antibody dataset preprocessing pipelines."""
```

---

**4.2 Create dataset-specific __init__.py files**
```bash
touch preprocessing/harvey/__init__.py
touch preprocessing/jain/__init__.py
touch preprocessing/shehata/__init__.py
```

**Content (example for harvey/__init__.py):**
```python
"""Harvey dataset preprocessing pipeline."""
```

**Why this matters:**
- Enables clean imports: `from preprocessing.harvey.step1_convert_raw_csvs import ...`
- Makes preprocessing a proper Python package
- Follows Python best practices
- No need for sys.path manipulation

---

### Phase 5: Fix Import Paths in Validation Scripts (2 files)

**Files that import conversion scripts:**
- `scripts/validation/validate_jain_conversion.py`
- `scripts/validation/validate_shehata_conversion.py`

**5.1 Fix validate_jain_conversion.py**

**Current import (line ~26):**
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "conversion"))
from convert_jain_excel_to_csv import (
```

**New import:**
```python
# Clean package import (works because we added __init__.py files)
from preprocessing.jain.step1_convert_excel_to_csv import (
```

**Remove the old sys.path line** - no longer needed!

---

**5.2 Fix validate_shehata_conversion.py**

**Current import (line ~24):**
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "conversion"))
from convert_shehata_excel_to_csv import (
```

**New import:**
```python
# Clean package import
from preprocessing.shehata.step1_convert_excel_to_csv import (
```

**Remove the old sys.path line** - no longer needed!

---

### Phase 5: Update scripts/conversion/README.md

**Update README to point users to new locations:**

```markdown
# Data Conversion Scripts

**⚠️  MOVED:** Conversion scripts have been reorganized.

## New Locations (Dataset-Centric)

Conversion scripts now live with their respective datasets in `preprocessing/`:

- **Boughter:** `preprocessing/boughter/stage1_dna_translation.py`
- **Harvey:** `preprocessing/harvey/step1_convert_raw_csvs.py`
- **Jain:** `preprocessing/jain/step1_convert_excel_to_csv.py`
- **Shehata:** `preprocessing/shehata/step1_convert_excel_to_csv.py`

## Rationale

Following industry-standard **dataset-centric organization**:
- All preprocessing for a dataset lives in ONE directory
- Consistent with HuggingFace, TensorFlow datasets, PyTorch patterns
- Easier to discover and maintain

## Legacy Scripts

See `legacy/` for historical incorrect implementations (archived for reference).
```

---

### Phase 6: Update Script Output Messages (3 files)

**Problem:** Conversion scripts print old paths in their output messages

**6.1 Update convert_harvey_csvs.py output messages**

**File:** `preprocessing/harvey/step1_convert_raw_csvs.py` (after move)

**Find and replace these print statements:**
```python
# OLD:
print("  Run preprocessing/process_harvey.py to generate fragments")

# NEW:
print("  Run preprocessing/harvey/step2_extract_fragments.py to generate fragments")
```

---

**6.2 Update convert_jain_excel_to_csv.py output messages**

**File:** `preprocessing/jain/step1_convert_excel_to_csv.py` (after move)

**Find and replace:**
```python
# OLD:
print("  Run preprocessing/preprocess_jain_p5e_s2.py to generate canonical files")

# NEW:
print("  Run preprocessing/jain/step2_preprocess_p5e_s2.py to generate canonical files")
```

---

**6.3 Update convert_shehata_excel_to_csv.py output messages**

**File:** `preprocessing/shehata/step1_convert_excel_to_csv.py` (after move)

**Find and replace:**
```python
# OLD:
print("  Run preprocessing/process_shehata.py to generate fragments")

# NEW:
print("  Run preprocessing/shehata/step2_extract_fragments.py to generate fragments")
```

---

### Phase 7: Update Documentation References (20+ files)

**Problem:** Documentation and README files reference old script paths

**7.1 Update Harvey Documentation (7 files)**

Files to update:
1. `docs/harvey/HARVEY_P0_FIX_REPORT.md` (line ~109)
2. `docs/harvey/harvey_preprocessing_implementation_plan.md`
3. `docs/harvey/harvey_data_cleaning_log.md`
4. `docs/harvey/harvey_script_status.md`
5. `docs/harvey/harvey_script_audit_request.md`
6. `docs/harvey/HARVEY_CLEANUP_INVESTIGATION.md`
7. `docs/harvey_data_sources.md`

**Find and replace:**
```bash
# Replace in all docs
find docs/ -name "*.md" -type f -exec sed -i '' \
  's|scripts/conversion/convert_harvey_csvs.py|preprocessing/harvey/step1_convert_raw_csvs.py|g' \
  's|preprocessing/process_harvey.py|preprocessing/harvey/step2_extract_fragments.py|g' {} +
```

---

**7.2 Update Jain Documentation (8+ files)**

Files to update:
1. `docs/jain/JAIN_REPLICATION_PLAN.md`
2. `docs/jain/jain_conversion_verification_report.md`
3. `docs/jain/jain_data_sources.md`
4. `docs/jain/JAIN_REORGANIZATION_COMPLETE.md`
5. `docs/archive/investigation_2025_11_05/JAIN_DATASETS_AUDIT_REPORT.md`
6. `docs/archive/p5_close_attempt/JAIN_NOVO_PARITY_VALIDATION_REPORT.md`
7. `scripts/conversion/legacy/README.md`
8. `scripts/conversion/legacy/convert_jain_excel_to_csv_TOTAL_FLAGS_WRONG.py`

**Find and replace:**
```bash
# Replace in all docs
find docs/ scripts/conversion/legacy/ -name "*.md" -o -name "*.py" -type f -exec sed -i '' \
  's|scripts/conversion/convert_jain_excel_to_csv.py|preprocessing/jain/step1_convert_excel_to_csv.py|g' \
  's|preprocessing/preprocess_jain_p5e_s2.py|preprocessing/jain/step2_preprocess_p5e_s2.py|g' {} +
```

---

**7.3 Update Shehata Documentation (7 files)**

Files to update:
1. `docs/shehata/shehata_preprocessing_implementation_plan.md`
2. `docs/shehata/shehata_phase2_completion_report.md`
3. `docs/shehata/shehata_data_sources.md`
4. `docs/shehata/shehata_conversion_verification_report.md`
5. `docs/shehata/SHEHATA_CLEANUP_PLAN.md`
6. `docs/shehata/SHEHATA_BLOCKER_ANALYSIS.md`
7. `docs/shehata/P0_BLOCKER_FIRST_PRINCIPLES_VALIDATION.md`

**Find and replace:**
```bash
# Replace in all docs
find docs/shehata/ -name "*.md" -type f -exec sed -i '' \
  's|scripts/conversion/convert_shehata_excel_to_csv.py|preprocessing/shehata/step1_convert_excel_to_csv.py|g' \
  's|preprocessing/process_shehata.py|preprocessing/shehata/step2_extract_fragments.py|g' {} +
```

---

**7.4 Update Test Dataset READMEs (10 files)**

Files to update:
1. `test_datasets/harvey/raw/README.md` (line ~46)
2. `test_datasets/harvey/processed/README.md`
3. `test_datasets/harvey/fragments/README.md`
4. `test_datasets/harvey/README.md`
5. `test_datasets/jain/raw/README.md`
6. `test_datasets/jain/processed/README.md`
7. `test_datasets/jain/canonical/README.md`
8. `test_datasets/jain/README.md`
9. `test_datasets/shehata/raw/README.md`
10. `test_datasets/shehata/processed/README.md`
11. `test_datasets/shehata/fragments/README.md`
12. `test_datasets/shehata/README.md`

**Find and replace:**
```bash
# Update all test dataset READMEs
find test_datasets/ -name "README.md" -type f -exec sed -i '' \
  's|scripts/conversion/convert_harvey_csvs.py|preprocessing/harvey/step1_convert_raw_csvs.py|g' \
  's|preprocessing/process_harvey.py|preprocessing/harvey/step2_extract_fragments.py|g' \
  's|scripts/conversion/convert_jain_excel_to_csv.py|preprocessing/jain/step1_convert_excel_to_csv.py|g' \
  's|preprocessing/preprocess_jain_p5e_s2.py|preprocessing/jain/step2_preprocess_p5e_s2.py|g' \
  's|scripts/conversion/convert_shehata_excel_to_csv.py|preprocessing/shehata/step1_convert_excel_to_csv.py|g' \
  's|preprocessing/process_shehata.py|preprocessing/shehata/step2_extract_fragments.py|g' {} +
```

---

**7.5 Update Cross-Dataset Documentation (5+ files)**

Files to update:
1. `docs/TEST_DATASETS_REORGANIZATION_PLAN.md`
2. `docs/excel_to_csv_conversion_methods.md`
3. `docs/METHODOLOGY_AND_DIVERGENCES.md`
4. `docs/boughter/boughter_processing_implementation.md`
5. `README.md` (root)

**Find and replace:**
```bash
# Update cross-dataset docs
find docs/ -maxdepth 1 -name "*.md" -type f -exec sed -i '' \
  's|scripts/conversion/convert_harvey_csvs.py|preprocessing/harvey/step1_convert_raw_csvs.py|g' \
  's|preprocessing/process_harvey.py|preprocessing/harvey/step2_extract_fragments.py|g' \
  's|scripts/conversion/convert_jain_excel_to_csv.py|preprocessing/jain/step1_convert_excel_to_csv.py|g' \
  's|preprocessing/preprocess_jain_p5e_s2.py|preprocessing/jain/step2_preprocess_p5e_s2.py|g' \
  's|scripts/conversion/convert_shehata_excel_to_csv.py|preprocessing/shehata/step1_convert_excel_to_csv.py|g' \
  's|preprocessing/process_shehata.py|preprocessing/shehata/step2_extract_fragments.py|g' {} +
```

---

**7.6 Update Test Files (1 file)**

File to update:
- `tests/test_harvey_embedding_compatibility.py` (line ~22)

**Find and replace:**
```python
# OLD:
# from scripts.conversion.convert_harvey_csvs import ...

# NEW:
# from preprocessing.harvey.step1_convert_raw_csvs import ...
```

**Command:**
```bash
sed -i '' 's|scripts/conversion/convert_harvey_csvs|preprocessing/harvey/step1_convert_raw_csvs|g' tests/test_harvey_embedding_compatibility.py
```

---

### Phase 8: Create Dataset READMEs (3 new files)

**8.1 Create preprocessing/harvey/README.md**

```markdown
# Harvey Dataset Preprocessing Pipeline

**Source:** Harvey et al. (2022) - Nanobody polyreactivity dataset
**Test Set:** 141,474 nanobody sequences (VHH only)

---

## Pipeline Overview

```
raw/*.csv → processed/harvey.csv → fragments/*.csv
  (Step 1)         (Step 2)
```

---

## Step 1: Convert Raw CSVs

**Script:** `step1_convert_raw_csvs.py`

**Purpose:** Combines high/low polyreactivity CSVs into single processed file.

**Input:**
- `test_datasets/harvey/raw/high_polyreactivity_high_throughput.csv` (71,772 sequences)
- `test_datasets/harvey/raw/low_polyreactivity_high_throughput.csv` (69,702 sequences)

**Output:**
- `test_datasets/harvey/processed/harvey.csv` (141,474 sequences)

**Run:**
```bash
python3 preprocessing/harvey/step1_convert_raw_csvs.py
```

**What it does:**
1. Extracts full sequences from IMGT position columns (1-128)
2. Extracts pre-annotated CDRs (CDR1_nogaps, CDR2_nogaps, CDR3_nogaps)
3. Assigns binary labels (0=low polyreactivity, 1=high polyreactivity)
4. Combines into single CSV

---

## Step 2: Extract Fragments

**Script:** `step2_extract_fragments.py`

**Purpose:** Annotate with ANARCI and extract VHH fragments (nanobody-specific).

**Input:**
- `test_datasets/harvey/processed/harvey.csv` (141,474 sequences)

**Output:**
- `test_datasets/harvey/fragments/*.csv` (6 fragment files)
  - VHH_only_harvey.csv
  - H-CDR1_harvey.csv
  - H-CDR2_harvey.csv
  - H-CDR3_harvey.csv
  - H-CDRs_harvey.csv (concatenated CDR1+2+3)
  - H-FWRs_harvey.csv (concatenated FWR1+2+3+4)

**Run:**
```bash
python3 preprocessing/harvey/step2_extract_fragments.py
```

**What it does:**
1. Annotates sequences with ANARCI (IMGT numbering scheme)
2. Extracts CDR regions (CDR1, CDR2, CDR3) using IMGT boundaries
3. Extracts framework regions (FWR1, FWR2, FWR3, FWR4)
4. Creates fragment-specific CSV files

---

## Full Pipeline Execution

**Run both steps sequentially:**
```bash
# Step 1: Convert raw CSVs
python3 preprocessing/harvey/step1_convert_raw_csvs.py

# Step 2: Extract fragments
python3 preprocessing/harvey/step2_extract_fragments.py
```

---

## Dataset Statistics

- **Total sequences:** 141,474 nanobodies
- **High polyreactivity:** 71,772 (label=1)
- **Low polyreactivity:** 69,702 (label=0)
- **Sequence type:** VHH only (nanobodies, no light chain)
- **Fragment files:** 6 (VHH, 3 CDRs, concatenated CDRs, concatenated FWRs)

---

## Dependencies

- pandas
- numpy
- riot_na (ANARCI for annotation)
- tqdm

---

## References

- **Harvey et al. (2022):** [Citation needed - add when available]
- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models
- **ANARCI:** IMGT numbering scheme for antibody annotation

---

**Last Updated:** 2025-11-05
**Status:** ✅ Production Ready
```

---

**8.2 Create preprocessing/jain/README.md**

```markdown
# Jain Dataset Preprocessing Pipeline

**Source:** Jain et al. (2017) PNAS - Biophysical properties of clinical-stage antibodies
**Test Set:** 86 antibodies (Novo Nordisk parity benchmark)

---

## Pipeline Overview

```
raw/*.xlsx → processed/*.csv → canonical/*.csv
  (Step 1)         (Step 2)
```

---

## Step 1: Convert Excel to CSV

**Script:** `step1_convert_excel_to_csv.py`

**Purpose:** Convert Jain Excel files to standardized CSV format using ELISA-only methodology.

**Input:**
- `test_datasets/jain/raw/Private_Jain2017_ELISA_indiv.xlsx`
- `test_datasets/jain/raw/jain-pnas.1616408114.sd01.xlsx`
- `test_datasets/jain/raw/jain-pnas.1616408114.sd02.xlsx`
- `test_datasets/jain/raw/jain-pnas.1616408114.sd03.xlsx`

**Output:**
- `test_datasets/jain/processed/jain_with_private_elisa_FULL.csv` (137 antibodies)
- `test_datasets/jain/processed/jain_sd01.csv`
- `test_datasets/jain/processed/jain_sd02.csv`
- `test_datasets/jain/processed/jain_sd03.csv`

**Run:**
```bash
python3 preprocessing/jain/step1_convert_excel_to_csv.py
```

**What it does:**
1. Loads private ELISA data (137 antibodies)
2. Loads public supplement data (SD01, SD02, SD03)
3. Applies ELISA-only flag calculation (0-6 range, NOT total flags 0-10)
4. Exports processed CSVs for downstream use

**Key Methodology:**
- **ELISA-only flags:** Uses ONLY 6 ELISA antigens (NOT all 10 assays)
- **Threshold:** ≥4 ELISA flags = non-specific
- **Corrected approach:** Fixes previous "total_flags" bug

---

## Step 2: Preprocess P5e-S2 (Novo Parity)

**Script:** `step2_preprocess_p5e_s2.py`

**Purpose:** Apply P5e-S2 methodology to achieve EXACT Novo Nordisk parity.

**Input:**
- `test_datasets/jain/processed/jain_with_private_elisa_FULL.csv` (137 antibodies)
- `test_datasets/jain/processed/jain_sd03.csv` (PSR/AC-SINS data)

**Output:**
- `test_datasets/jain/processed/jain_ELISA_ONLY_116.csv` (116 antibodies)
- `test_datasets/jain/canonical/jain_86_novo_parity.csv` (86 antibodies)

**Run:**
```bash
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

**What it does:**

**Pipeline:**
```
137 antibodies (FULL)
  ↓ Remove ELISA 1-3 (mild aggregators)
116 antibodies (ELISA_ONLY_116.csv) ✅ OUTPUT 1
  ↓ Reclassify 5 spec→nonspec (3 PSR>0.4 + eldelumab + infliximab)
89 spec / 27 nonspec
  ↓ Remove 30 by PSR primary, AC-SINS tiebreaker
86 antibodies (59 spec / 27 nonspec) ✅ OUTPUT 2
```

**Result:** Confusion matrix [[40, 19], [10, 17]] - **EXACT MATCH** (66.28% accuracy)

**Method:** P5e-S2 (PSR reclassification + PSR/AC-SINS removal)

---

## Full Pipeline Execution

**Run both steps sequentially:**
```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/jain/step1_convert_excel_to_csv.py

# Step 2: Preprocess to Novo parity
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

---

## Dataset Statistics

- **Source:** 137 antibodies with private ELISA data
- **After ELISA filtering:** 116 antibodies
- **Final benchmark:** 86 antibodies (59 specific / 27 non-specific)
- **Novo parity:** 66.28% accuracy (EXACT match)

---

## Methodology Notes

**CRITICAL:** This preprocessing uses **ELISA-only flags** (0-6 range), NOT total flags (0-10).

**Evidence:**
- Figure S13: x-axis shows "ELISA flag" (singular) with range 0-6
- Table 2: "ELISA with a panel of 6 ligands"
- Paper text: "non-specificity ELISA flags"

**Retired Approach:**
- Previous 94→86 methodology (VH length outliers + biology removals) did NOT match Novo
- total_flags approach was INCORRECT (used all 10 assays instead of 6 ELISA)

---

## Dependencies

- pandas
- numpy
- openpyxl (for Excel reading)

---

## References

- **Jain et al. (2017) PNAS:** Biophysical properties of the clinical-stage antibody landscape
- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models

---

**Last Updated:** 2025-11-05
**Status:** ✅ Production Ready (Novo Parity Achieved)
```

---

**8.3 Create preprocessing/shehata/README.md**

```markdown
# Shehata Dataset Preprocessing Pipeline

**Source:** Shehata et al. (2019) - PSR assay dataset
**Test Set:** 398 human antibodies with polyspecific reagent (PSR) measurements

---

## Pipeline Overview

```
raw/*.xlsx → processed/shehata.csv → fragments/*.csv
  (Step 1)         (Step 2)
```

---

## Step 1: Convert Excel to CSV

**Script:** `step1_convert_excel_to_csv.py`

**Purpose:** Convert Shehata Excel file to standardized CSV format.

**Input:**
- `test_datasets/shehata/raw/shehata-mmc2.xlsx`

**Output:**
- `test_datasets/shehata/processed/shehata.csv` (398 antibodies)

**Run:**
```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
```

**What it does:**
1. Loads Shehata Excel supplementary file (mmc2.xlsx)
2. Extracts VH and VL sequences
3. Extracts PSR assay measurements
4. Assigns binary labels based on PSR threshold
5. Exports standardized CSV

---

## Step 2: Extract Fragments

**Script:** `step2_extract_fragments.py`

**Purpose:** Annotate with ANARCI and extract paired antibody fragments.

**Input:**
- `test_datasets/shehata/processed/shehata.csv` (398 antibodies)

**Output:**
- `test_datasets/shehata/fragments/*.csv` (16 fragment files)

**Fragment types:**
1. VH_only, VL_only (full variable domains)
2. H-CDR1, H-CDR2, H-CDR3 (heavy chain CDRs)
3. L-CDR1, L-CDR2, L-CDR3 (light chain CDRs)
4. H-CDRs, L-CDRs (concatenated CDRs per chain)
5. H-FWRs, L-FWRs (concatenated frameworks per chain)
6. VH+VL (paired variable domains)
7. All-CDRs, All-FWRs (all concatenated)
8. Full (alias for VH+VL)

**Run:**
```bash
python3 preprocessing/shehata/step2_extract_fragments.py
```

**What it does:**
1. Annotates VH and VL sequences with ANARCI (IMGT numbering)
2. Extracts CDR and FWR regions using IMGT boundaries
3. Creates 16 fragment-specific CSV files
4. Preserves PSR measurements and labels

---

## Full Pipeline Execution

**Run both steps sequentially:**
```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/shehata/step1_convert_excel_to_csv.py

# Step 2: Extract fragments
python3 preprocessing/shehata/step2_extract_fragments.py
```

---

## Dataset Statistics

- **Total sequences:** 398 human antibodies
- **Sequence type:** Paired VH+VL (full antibodies)
- **Assay:** PSR (polyspecific reagent)
- **Fragment files:** 16 (all combinations of CDRs, FWRs, paired/unpaired)

---

## Assay-Specific Threshold

**PSR Threshold:** 0.549 (optimized for PSR assay)

**Note:** Different assays require different classification thresholds:
- ELISA (Jain): 0.5 (default)
- PSR (Shehata): 0.549

See `scripts/analysis/analyze_threshold_optimization.py` for details.

---

## Dependencies

- pandas
- numpy
- openpyxl (for Excel reading)
- riot_na (ANARCI for annotation)
- tqdm

---

## References

- **Shehata et al. (2019):** [Citation needed - add when available]
- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models
- **ANARCI:** IMGT numbering scheme for antibody annotation

---

**Last Updated:** 2025-11-05
**Status:** ✅ Production Ready
```

---

**8.4 Create preprocessing/README.md (Overview)**

```markdown
# Antibody Dataset Preprocessing

**Overview:** This directory contains all preprocessing pipelines for the four core datasets used in antibody non-specificity prediction.

---

## Datasets

### 1. Boughter (Training Set)

**Directory:** `preprocessing/boughter/`
**Purpose:** Training data for antibody polyreactivity classification
**Size:** 914 training sequences (from 1,171 raw)
**Pipeline:** 3-stage (DNA translation → Annotation → QC)

**Quick Start:**
```bash
python3 preprocessing/boughter/stage1_dna_translation.py
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py
```

**Details:** See [boughter/README.md](boughter/README.md)

---

### 2. Harvey (Test Set - Nanobodies)

**Directory:** `preprocessing/harvey/`
**Purpose:** Test set for nanobody polyreactivity (VHH only)
**Size:** 141,474 nanobody sequences
**Pipeline:** 2-step (Combine CSVs → Extract fragments)

**Quick Start:**
```bash
python3 preprocessing/harvey/step1_convert_raw_csvs.py
python3 preprocessing/harvey/step2_extract_fragments.py
```

**Details:** See [harvey/README.md](harvey/README.md)

---

### 3. Jain (Test Set - Novo Parity)

**Directory:** `preprocessing/jain/`
**Purpose:** Test set for clinical antibodies (Novo Nordisk benchmark)
**Size:** 86 antibodies (59 specific / 27 non-specific)
**Pipeline:** 2-step (Excel → CSV → P5e-S2 preprocessing)

**Quick Start:**
```bash
python3 preprocessing/jain/step1_convert_excel_to_csv.py
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

**Details:** See [jain/README.md](jain/README.md)

---

### 4. Shehata (Test Set - PSR Assay)

**Directory:** `preprocessing/shehata/`
**Purpose:** Test set for paired antibodies (PSR assay)
**Size:** 398 human antibodies
**Pipeline:** 2-step (Excel → CSV → Extract fragments)

**Quick Start:**
```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
python3 preprocessing/shehata/step2_extract_fragments.py
```

**Details:** See [shehata/README.md](shehata/README.md)

---

## Directory Structure

**Pattern:** Each dataset owns its complete preprocessing pipeline

```
preprocessing/
├── README.md              # This file (overview)
├── boughter/              # Training set (3-stage pipeline)
│   ├── README.md
│   ├── stage1_dna_translation.py
│   ├── stage2_stage3_annotation_qc.py
│   └── validate_*.py
├── harvey/                # Test set: nanobodies (2-step pipeline)
│   ├── README.md
│   ├── step1_convert_raw_csvs.py
│   └── step2_extract_fragments.py
├── jain/                  # Test set: clinical Abs (2-step pipeline)
│   ├── README.md
│   ├── step1_convert_excel_to_csv.py
│   └── step2_preprocess_p5e_s2.py
└── shehata/               # Test set: paired Abs (2-step pipeline)
    ├── README.md
    ├── step1_convert_excel_to_csv.py
    └── step2_extract_fragments.py
```

---

## Design Philosophy

### Dataset-Centric Organization

**Principle:** All preprocessing for a dataset lives in ONE directory.

**Benefits:**
1. **Discoverability:** "How do I preprocess Harvey?" → `preprocessing/harvey/`
2. **Maintainability:** Bug in Jain? → All scripts in `preprocessing/jain/`
3. **Consistency:** All datasets follow same pattern
4. **Documentation:** Each dataset has complete pipeline README
5. **Isolation:** Changes to one dataset don't affect others

**Follows industry standards:**
- HuggingFace datasets (each dataset owns preprocessing)
- TensorFlow datasets (dataset-centric structure)
- PyTorch torchvision (one file per dataset)

---

## Common Preprocessing Stages

### Stage 1: Format Conversion
- **Purpose:** Convert raw data (Excel, FASTA, CSV) to standardized CSV
- **Output:** `test_datasets/{dataset}/processed/*.csv`

### Stage 2: Fragment Extraction
- **Purpose:** Annotate with ANARCI, extract CDRs/FWRs
- **Output:** `test_datasets/{dataset}/fragments/*.csv` or `canonical/*.csv`

### Stage 3: Quality Control (Boughter only)
- **Purpose:** Filter sequences, apply Novo Nordisk flagging
- **Output:** Training subset with quality filters

---

## Dependencies

**All preprocessing scripts require:**
- pandas
- numpy
- tqdm

**Fragment extraction requires:**
- riot_na (ANARCI wrapper for antibody annotation)

**Excel conversion requires:**
- openpyxl

**Install all dependencies:**
```bash
uv sync
```

---

## References

- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models
- **ANARCI:** Antibody numbering and receptor classification
- **IMGT:** International ImMunoGeneTics information system

---

**Last Updated:** 2025-11-05
**Status:** ✅ Production Ready
```

---

### Phase 9: Update scripts/conversion/README.md (Redirect)

**Purpose:** Update the conversion README to redirect users to new locations

**File:** `scripts/conversion/README.md`

**Replace entire contents with:**

```markdown
# Data Conversion Scripts

**⚠️  MOVED:** Conversion scripts have been reorganized.

## New Locations (Dataset-Centric)

Conversion scripts now live with their respective datasets in `preprocessing/`:

- **Boughter:** `preprocessing/boughter/stage1_dna_translation.py`
- **Harvey:** `preprocessing/harvey/step1_convert_raw_csvs.py`
- **Jain:** `preprocessing/jain/step1_convert_excel_to_csv.py`
- **Shehata:** `preprocessing/shehata/step1_convert_excel_to_csv.py`

## Rationale

Following industry-standard **dataset-centric organization**:
- All preprocessing for a dataset lives in ONE directory
- Consistent with HuggingFace, TensorFlow datasets, PyTorch patterns
- Easier to discover and maintain

## Legacy Scripts

See `legacy/` for historical incorrect implementations (archived for reference).
```

---

### Phase 10: Clean Up Empty scripts/conversion/

**Option A: Delete scripts/conversion/ entirely**
```bash
# Remove non-legacy files
rm scripts/conversion/README.md

# Remove directory (keep legacy/ in archive or move to docs/)
# OR just leave legacy/ folder with updated README
```

**Option B: Keep scripts/conversion/ with redirect README**
- Already done in Phase 5
- Leave legacy/ folder intact
- README points to new locations

**Recommendation:** Option B (keep legacy/, update README to redirect)

---

### Phase 11: Validation & Testing

**11.1 Check Git Status**
```bash
git status --short
```

**Expected output:**
```
A  preprocessing/harvey/README.md
A  preprocessing/harvey/step1_convert_raw_csvs.py
A  preprocessing/harvey/step2_extract_fragments.py
A  preprocessing/jain/README.md
A  preprocessing/jain/step1_convert_excel_to_csv.py
A  preprocessing/jain/step2_preprocess_p5e_s2.py
A  preprocessing/shehata/README.md
A  preprocessing/shehata/step1_convert_excel_to_csv.py
A  preprocessing/shehata/step2_extract_fragments.py
A  preprocessing/README.md
M  scripts/conversion/README.md
M  scripts/validation/validate_jain_conversion.py
M  scripts/validation/validate_shehata_conversion.py
D  preprocessing/process_harvey.py
D  preprocessing/preprocess_jain_p5e_s2.py
D  preprocessing/process_shehata.py
D  scripts/conversion/convert_harvey_csvs.py
D  scripts/conversion/convert_jain_excel_to_csv.py
D  scripts/conversion/convert_shehata_excel_to_csv.py
```

---

**11.2 Verify Git History Preserved**
```bash
# Check Harvey conversion script history
git log --follow preprocessing/harvey/step1_convert_raw_csvs.py | head -10

# Check Jain preprocessing history
git log --follow preprocessing/jain/step2_preprocess_p5e_s2.py | head -10
```

**Expected:** Full git history from original files preserved

---

**11.3 Test Imports**
```bash
# Check if validation scripts can import
python3 -c "import sys; sys.path.insert(0, 'preprocessing/jain'); from step1_convert_excel_to_csv import load_data; print('✅ Import works')"
```

---

**11.4 Verify No Broken References**
```bash
# Search for old paths in all Python files
grep -r "scripts/conversion/convert_harvey" . --include="*.py" | grep -v ".git" | grep -v "legacy"
grep -r "scripts/conversion/convert_jain" . --include="*.py" | grep -v ".git" | grep -v "legacy"
grep -r "scripts/conversion/convert_shehata" . --include="*.py" | grep -v ".git" | grep -v "legacy"

# Search for old preprocessing paths
grep -r "preprocessing/process_harvey\|preprocessing/preprocess_jain_p5e_s2\|preprocessing/process_shehata" . --include="*.py" | grep -v ".git"
```

**Expected:** Only matches in:
- Updated README files (documentation)
- Legacy folder (archived scripts)
- Git history (preserved for reference)

---

**11.5 Dry-Run Test (Optional)**
```bash
# Test Harvey pipeline (dry-run, don't write files)
python3 preprocessing/harvey/step1_convert_raw_csvs.py --help
python3 preprocessing/harvey/step2_extract_fragments.py --help

# Verify scripts run without errors
```

---

### Phase 12: Commit Changes

**Commit message:**
```bash
git add -A

git commit -m "refactor: Reorganize preprocessing to dataset-centric structure (12 phases)

BREAKING CHANGE: Preprocessing scripts moved to dataset subdirectories

This comprehensive refactor follows industry-standard dataset-centric
organization pattern (HuggingFace, TensorFlow datasets, PyTorch).

## Summary of Changes (~49 files affected)

### Code Moves (6 files - git history preserved)
- scripts/conversion/convert_harvey_csvs.py
  → preprocessing/harvey/step1_convert_raw_csvs.py
- preprocessing/process_harvey.py
  → preprocessing/harvey/step2_extract_fragments.py
- scripts/conversion/convert_jain_excel_to_csv.py
  → preprocessing/jain/step1_convert_excel_to_csv.py
- preprocessing/preprocess_jain_p5e_s2.py
  → preprocessing/jain/step2_preprocess_p5e_s2.py
- scripts/conversion/convert_shehata_excel_to_csv.py
  → preprocessing/shehata/step1_convert_excel_to_csv.py
- preprocessing/process_shehata.py
  → preprocessing/shehata/step2_extract_fragments.py

### New Files (8 files)
Package structure:
- preprocessing/__init__.py (makes preprocessing a proper Python package)
- preprocessing/harvey/__init__.py
- preprocessing/jain/__init__.py
- preprocessing/shehata/__init__.py

Documentation:
- preprocessing/README.md (comprehensive overview of all 4 datasets)
- preprocessing/harvey/README.md (complete Harvey pipeline docs)
- preprocessing/jain/README.md (complete Jain pipeline docs)
- preprocessing/shehata/README.md (complete Shehata pipeline docs)

### Updated Files (35+ files)

Core Scripts (5 files):
- scripts/conversion/README.md (redirect to new locations)
- scripts/validation/validate_jain_conversion.py (clean package import)
- scripts/validation/validate_shehata_conversion.py (clean package import)
- preprocessing/harvey/step1_convert_raw_csvs.py (update output messages)
- preprocessing/jain/step1_convert_excel_to_csv.py (update output messages)

Documentation (30+ files):
- Harvey docs (7 files): docs/harvey/*.md, docs/harvey_data_sources.md
- Jain docs (8 files): docs/jain/*.md, scripts/conversion/legacy/*.md
- Shehata docs (7 files): docs/shehata/*.md
- Test dataset READMEs (12 files): test_datasets/**/*.md
- Cross-dataset docs (5+ files): docs/*.md, README.md
- Test files (1 file): tests/test_harvey_embedding_compatibility.py

## Key Improvements

✅ Consistent structure: All datasets follow Boughter pattern
✅ Discoverability: All preprocessing for dataset X in preprocessing/X/
✅ Clean imports: No sys.path manipulation (proper Python packages)
✅ Complete docs: Each dataset has full pipeline README
✅ Updated references: All 30+ docs point to new locations
✅ Industry standard: Matches HuggingFace, TensorFlow, PyTorch patterns

## Validation Performed

✅ Git history preserved (git log --follow verified)
✅ No broken imports (validation scripts tested)
✅ No old path references in active code (grep verified)
✅ Script output messages updated
✅ All documentation updated

## Safety

✅ Backup on main branch (commit a868338)
✅ No changes to data files or test datasets
✅ No logic changes (only file locations and imports)
✅ Easy rollback: git reset --hard HEAD~1

All core datasets (Boughter, Harvey, Jain, Shehata) remain fully functional.

See PREPROCESSING_DATASET_CENTRIC_REFACTOR_PLAN.md for complete execution details.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Safety & Rollback

### Backup Status

**Current State:**
- ✅ Branch: leroy-jenkins/full-send
- ✅ Backup: main branch (commit a868338)
- ✅ Git history: Preserved with `git mv`
- ✅ No upstream dependencies

**Safety Net:**
```bash
# If anything goes wrong, hard reset to backup
git reset --hard a868338

# Or checkout main branch
git checkout main
```

---

### Rollback Procedure

**If refactor causes issues:**

1. **Immediate rollback:**
   ```bash
   git reset --hard HEAD~1  # Undo last commit
   ```

2. **Restore from backup:**
   ```bash
   git reset --hard a868338  # Reset to pre-refactor state
   ```

3. **Cherry-pick good changes:**
   ```bash
   git checkout main
   git cherry-pick <commit-hash>  # Pick specific changes
   ```

---

## Impact Assessment

### Files Affected

**Moved (6 files):**
- scripts/conversion/convert_harvey_csvs.py → preprocessing/harvey/step1_convert_raw_csvs.py
- scripts/conversion/convert_jain_excel_to_csv.py → preprocessing/jain/step1_convert_excel_to_csv.py
- scripts/conversion/convert_shehata_excel_to_csv.py → preprocessing/shehata/step1_convert_excel_to_csv.py
- preprocessing/process_harvey.py → preprocessing/harvey/step2_extract_fragments.py
- preprocessing/preprocess_jain_p5e_s2.py → preprocessing/jain/step2_preprocess_p5e_s2.py
- preprocessing/process_shehata.py → preprocessing/shehata/step2_extract_fragments.py

**Created (8 files):**
- preprocessing/__init__.py (package structure)
- preprocessing/harvey/__init__.py
- preprocessing/jain/__init__.py
- preprocessing/shehata/__init__.py
- preprocessing/README.md (overview)
- preprocessing/harvey/README.md (pipeline docs)
- preprocessing/jain/README.md (pipeline docs)
- preprocessing/shehata/README.md (pipeline docs)

**Updated - Core Scripts (5 files):**
- scripts/conversion/README.md (redirect to new locations)
- scripts/validation/validate_jain_conversion.py (fix import path)
- scripts/validation/validate_shehata_conversion.py (fix import path)
- preprocessing/harvey/step1_convert_raw_csvs.py (update output messages)
- preprocessing/jain/step1_convert_excel_to_csv.py (update output messages)

**Updated - Documentation (30+ files):**
- Harvey docs (7 files): docs/harvey/*.md, docs/harvey_data_sources.md
- Jain docs (8 files): docs/jain/*.md, scripts/conversion/legacy/*.md
- Shehata docs (7 files): docs/shehata/*.md
- Test dataset READMEs (12 files): test_datasets/{harvey,jain,shehata}/**/*.md
- Cross-dataset docs (5+ files): docs/*.md, README.md
- Test files (1 file): tests/test_harvey_embedding_compatibility.py

**Total:** ~49 files affected (6 moves, 8 new, 35 updates)

---

### Risk Analysis

**Low Risk:**
- ✅ All moves use `git mv` (history preserved)
- ✅ No logic changes, only file locations
- ✅ Validation scripts get import fixes
- ✅ Backup on main branch
- ✅ Easy rollback available

**Medium Risk:**
- ⚠️ Import paths need updating (2 validation scripts)
- ⚠️ Scripts in other repos may reference old paths
- ⚠️ Documentation elsewhere may reference old paths

**Mitigation:**
- Update validation script imports in same commit
- Document old→new path mapping in commit message
- Keep scripts/conversion/README.md as redirect

---

## Success Criteria

**After refactor, verify:**

**Structure:**
- [ ] All 4 datasets have subdirectories in preprocessing/
- [ ] Each dataset has README.md with complete pipeline docs
- [ ] All __init__.py files created (preprocessing/ + 3 dataset subdirs)
- [ ] preprocessing/README.md provides comprehensive overview

**Code:**
- [ ] Git history preserved (`git log --follow` works for all 6 moved files)
- [ ] No broken imports (validation scripts use clean package imports)
- [ ] No sys.path manipulation needed (proper Python packages)
- [ ] Script output messages reference new paths

**Documentation:**
- [ ] scripts/conversion/README.md redirects to new locations
- [ ] All Harvey docs updated (7 files)
- [ ] All Jain docs updated (8+ files)
- [ ] All Shehata docs updated (7 files)
- [ ] All test dataset READMEs updated (12 files)
- [ ] Cross-dataset docs updated (5+ files)
- [ ] Test files updated (1 file)

**Validation:**
- [ ] No references to old paths in active code (grep verification)
- [ ] Validation scripts import successfully
- [ ] All preprocessing scripts can be found via new paths
- [ ] Commit message documents all changes comprehensively

---

## Post-Refactor TODO

**Optional improvements (separate PRs):**

1. **Add validation scripts to dataset directories**
   - Move validate_jain_conversion.py → preprocessing/jain/
   - Move validate_shehata_conversion.py → preprocessing/shehata/
   - Create validate_harvey_conversion.py

2. **Standardize step naming**
   - Consider stage1/stage2 (like Boughter) vs step1/step2
   - Update if consistency preferred

3. **Add per-dataset tests**
   - preprocessing/harvey/test_pipeline.py
   - preprocessing/jain/test_pipeline.py
   - preprocessing/shehata/test_pipeline.py

4. **Update root README.md**
   - Reference new preprocessing/ structure
   - Update quick start guides

---

## Timeline Estimate

**Execution time:** ~60-90 minutes

**Breakdown:**
- Phase 1: Create directories (1 min)
- Phase 2: Move conversion scripts (3 min)
- Phase 3: Move preprocessing scripts (3 min)
- Phase 4: Create __init__.py files (2 min)
- Phase 5: Fix validation script imports (5 min)
- Phase 6: Update script output messages (5 min)
- Phase 7: Update documentation references (15-20 min)
  - 7.1 Harvey docs: 3 min
  - 7.2 Jain docs: 3 min
  - 7.3 Shehata docs: 3 min
  - 7.4 Test dataset READMEs: 3 min
  - 7.5 Cross-dataset docs: 3 min
  - 7.6 Test files: 1 min
- Phase 8: Create dataset READMEs (10 min)
- Phase 9: Update scripts/conversion/README.md (2 min)
- Phase 10: Clean up (1 min)
- Phase 11: Validation & testing (10-15 min)
- Phase 12: Commit (5 min)

**Total:** 1-1.5 hours (conservative estimate with thorough validation)

---

## Questions for Senior Review

1. **Naming convention:** `step1_*.py` vs `stage1_*.py` vs `01_*.py`?
   - ✅ **Decided:** `step1_` for Harvey/Jain/Shehata (consistent with multi-step pattern)
   - ✅ **Keep:** `stage1_` for Boughter (already established, don't change working pattern)

2. **scripts/conversion/ fate:**
   - ✅ **Decided:** Keep with redirect README + legacy/ folder (Option B)
   - Provides clear migration path for users

3. **Validation script location:**
   - ❓ **Open Question:** Should we move validation scripts too?
   - Current: scripts/validation/validate_*_conversion.py
   - Future: preprocessing/{dataset}/validate_conversion.py?
   - **Recommendation:** Keep in scripts/validation/ for now (separate from preprocessing)

4. **Import strategy:**
   - ✅ **Decided:** Use proper Python packages with __init__.py files
   - Clean imports: `from preprocessing.jain.step1_convert_excel_to_csv import ...`
   - No sys.path manipulation needed

5. **Documentation automation:**
   - ❓ **Open Question:** Should we create a script to automate the sed replacements?
   - Or manually execute each Phase 7 substep for safety?
   - **Recommendation:** Manual execution for first run (this refactor), script for future use

---

## Approval Checklist

**Before executing, senior should verify:**

- [ ] Rationale is sound (dataset-centric is industry standard)
- [ ] File moves preserve git history
- [ ] Import fixes are correct
- [ ] README templates are complete and accurate
- [ ] Rollback procedure is clear
- [ ] Risk assessment is accurate
- [ ] Success criteria are measurable
- [ ] No critical files are affected (no model changes, no data changes)

---

## References

**Industry Examples:**
- [HuggingFace datasets structure](https://github.com/huggingface/datasets/tree/main/datasets)
- [TensorFlow datasets](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets)
- [PyTorch torchvision](https://github.com/pytorch/vision/tree/main/torchvision/datasets)

**Internal Docs:**
- SCRIPTS_PREPROCESSING_REFACTOR_PLAN.md (previous refactor)
- PREPROCESSING_STRUCTURE_ANALYSIS.md (options analysis)
- TEST_DATASETS_REORGANIZATION_PLAN.md (4-tier data structure)

---

**Status:** 🟡 AWAITING SENIOR APPROVAL
**Next Step:** Senior review → Approval → Execute → Document

**Plan Author:** Claude Code (at Ray's request)
**Review Date:** [To be filled by reviewer]
**Approved By:** [To be filled by reviewer]
**Execution Date:** [To be filled after approval]
