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
