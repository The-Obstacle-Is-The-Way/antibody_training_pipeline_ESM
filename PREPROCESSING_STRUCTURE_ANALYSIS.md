# Preprocessing Structure Analysis

**Date:** 2025-11-05
**Question:** Should preprocessing be organized by DATASET or by PHASE?

---

## Current State

### Boughter: Dataset-Centric (Subdirectory)
```
preprocessing/boughter/
├── README.md
├── stage1_dna_translation.py      # Phase 1: FASTA → CSV
├── stage2_stage3_annotation_qc.py # Phase 2: Annotation + QC
├── stage4_additional_qc.py        # Phase 3: Additional QC
├── validate_stage1.py
├── validate_stages2_3.py
└── validate_stage4.py
```

**Pros:**
- ✅ All Boughter preprocessing in ONE place
- ✅ Clear dataset ownership
- ✅ Easy to find all scripts for one dataset
- ✅ Self-contained with validation

**Cons:**
- ⚠️ Only needed because Boughter has 3+ stages

---

### Harvey/Jain/Shehata: Phase-Centric (Split)

**Phase 1: Raw → Processed (in scripts/conversion/)**
```
scripts/conversion/
├── convert_harvey_csvs.py         # raw/*.csv → processed/harvey.csv
├── convert_jain_excel_to_csv.py   # raw/*.xlsx → processed/*.csv
└── convert_shehata_excel_to_csv.py # raw/*.xlsx → processed/shehata.csv
```

**Phase 2: Processed → Fragments (in preprocessing/)**
```
preprocessing/
├── process_harvey.py              # processed/harvey.csv → fragments/*.csv
├── preprocess_jain_p5e_s2.py      # processed/*.csv → canonical/*.csv
└── process_shehata.py             # processed/shehata.csv → fragments/*.csv
```

**Pros:**
- ✅ Clear separation of concerns (conversion vs annotation)
- ✅ Simpler for single-stage datasets
- ✅ Conversion scripts are reusable utilities

**Cons:**
- ❌ Inconsistent with Boughter pattern
- ❌ Dataset preprocessing is SPLIT across two directories
- ❌ Harder to find "all preprocessing for dataset X"

---

## Professional Perspective

### Industry Standard: Dataset-Centric Organization

**Common pattern in professional ML repos:**

```
data/
├── dataset_a/
│   ├── download.py
│   ├── preprocess.py
│   ├── validate.py
│   └── README.md
├── dataset_b/
│   ├── download.py
│   ├── preprocess.py
│   ├── validate.py
│   └── README.md
└── shared/
    └── utils.py
```

**Examples:**
- **HuggingFace datasets:** Each dataset in its own directory
- **TensorFlow datasets:** `tensorflow_datasets/image_classification/mnist/`, etc.
- **PyTorch torchvision:** `torchvision/datasets/mnist.py`, `torchvision/datasets/cifar.py`

**Key principle:** **Dataset ownership** - everything for one dataset lives together

---

## Analysis: Should We Reorganize?

### Option 1: Keep Current Structure (Phase-Centric)

**Rationale:**
- Boughter is special (multi-stage, complex)
- Other datasets are simple (2 scripts each)
- Conversion scripts are utility-like

**When this makes sense:**
- Conversion is truly a separate concern
- Scripts are reused across datasets
- Preprocessing is truly single-stage

---

### Option 2: Reorganize to Dataset-Centric (Like Boughter)

**Proposed structure:**
```
preprocessing/
├── boughter/
│   ├── README.md
│   ├── stage1_dna_translation.py
│   ├── stage2_stage3_annotation_qc.py
│   └── validate_*.py
├── harvey/
│   ├── README.md
│   ├── step1_convert_raw_csvs.py      # from scripts/conversion/
│   ├── step2_extract_fragments.py     # from preprocessing/harvey/step2_extract_fragments.py
│   └── validate_*.py (optional)
├── jain/
│   ├── README.md
│   ├── step1_convert_excel_to_csv.py  # from scripts/conversion/
│   ├── step2_preprocess_p5e_s2.py     # from preprocessing/jain/step2_preprocess_p5e_s2.py
│   └── validate_*.py (optional)
└── shehata/
    ├── README.md
    ├── step1_convert_excel_to_csv.py  # from scripts/conversion/
    ├── step2_extract_fragments.py     # from preprocessing/shehata/step2_extract_fragments.py
    └── validate_*.py (optional)
```

**Benefits:**
- ✅ **Consistency:** All datasets follow same pattern
- ✅ **Discoverability:** One place for all dataset preprocessing
- ✅ **Clarity:** Clear pipeline for each dataset
- ✅ **Maintainability:** Easy to understand and modify
- ✅ **Documentation:** Per-dataset README with complete pipeline

**Drawbacks:**
- ⚠️ More directory nesting
- ⚠️ Conversion scripts no longer grouped together
- ⚠️ Need to rename files (step1_, step2_)

---

### Option 3: Hybrid Approach

**Keep conversion utilities separate, organize preprocessing by dataset:**

```
scripts/conversion/
├── README.md ("Utility scripts - see preprocessing/*/README.md for usage")
├── convert_harvey_csvs.py
├── convert_jain_excel_to_csv.py
└── convert_shehata_excel_to_csv.py

preprocessing/
├── boughter/
│   ├── README.md (references conversion if needed)
│   ├── stage1_dna_translation.py
│   └── stage2_stage3_annotation_qc.py
├── harvey/
│   ├── README.md (references preprocessing/harvey/step1_convert_raw_csvs.py)
│   └── process_fragments.py (renamed from process_harvey.py)
├── jain/
│   ├── README.md (references preprocessing/jain/step1_convert_excel_to_csv.py)
│   └── preprocess_p5e_s2.py (renamed from preprocess_jain_p5e_s2.py)
└── shehata/
    ├── README.md (references preprocessing/shehata/step1_convert_excel_to_csv.py)
    └── process_fragments.py (renamed from process_shehata.py)
```

**Benefits:**
- ✅ Consistent subdirectory structure
- ✅ Conversion scripts stay together (easier to maintain utilities)
- ✅ READMEs document the full pipeline (including conversion step)
- ✅ Less refactoring (just move files into subdirs)

**Trade-offs:**
- ⚠️ Still split between two locations
- ✅ But documented clearly in per-dataset READMEs

---

## Recommendation

**Option 3: Hybrid Approach**

**Why:**
1. **Consistency:** Every dataset gets a subdirectory with README
2. **Pragmatic:** Conversion scripts are truly utilities (like scripts/validation/)
3. **Clear documentation:** Each dataset README documents full pipeline
4. **Less disruptive:** Minimal refactoring needed

**Example README pattern:**

```markdown
# Harvey Dataset Preprocessing

## Pipeline

### Step 1: Combine Raw CSVs
```bash
python3 preprocessing/harvey/step1_convert_raw_csvs.py
```

Combines high/low polyreactivity CSVs into single processed CSV.

### Step 2: Extract Fragments
```bash
python3 preprocessing/harvey/process_fragments.py
```

Annotates with ANARCI and extracts VHH fragments.
```

---

## Questions for User

1. **Do you want full dataset-centric (Option 2)?**
   - Move conversion scripts INTO each dataset directory

2. **Do you prefer hybrid (Option 3)?**
   - Keep conversion scripts separate but organize preprocessing by dataset

3. **Keep current structure (Option 1)?**
   - Document why Boughter is different

---

**Next Steps:** Awaiting user decision on structure preference.
