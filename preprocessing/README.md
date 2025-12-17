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
**Size:** 141,474 raw/processed sequences (141,021 ANARCI-validated)
**Pipeline:** 2-step (Combine CSVs → Extract fragments)

**Quick Start:**
```bash
python3 preprocessing/harvey/step1_convert_raw_csvs.py
python3 preprocessing/harvey/step2_extract_fragments.py
```

**Details:** See [harvey/README.md](harvey/README.md)

---

### 3. Jain (Test Set - Clinical Antibodies)

**Directory:** `preprocessing/jain/`
**Purpose:** Test set for clinical antibodies (comparison with Novo Nordisk benchmark)
**Size:** 86 antibodies (57 specific / 29 non-specific; exact Novo parity)
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

**Pattern:** Dataset-centric organization with shared utilities

```
preprocessing/
├── README.md                  # This file (overview)
├── __init__.py                # Package marker
├── fragment_utils.py          # Shared ANARCI annotation & fragment extraction
├── validation_utils.py        # Shared validation logic (schema, gaps, labels)
├── paths.py                   # Centralized path management
├── logging_config.py          # Shared logging configuration
├── boughter/                  # Training set (3-stage pipeline)
│   ├── README.md
│   ├── stage1_dna_translation.py
│   ├── stage2_stage3_annotation_qc.py
│   ├── validate_stages2_3.py  # Uses validation_utils.py
│   └── annotation/
│       └── annotator.py       # Uses fragment_utils.py
├── harvey/                    # Test set: nanobodies (2-step pipeline)
│   ├── README.md
│   ├── step1_convert_raw_csvs.py
│   └── step2_extract_fragments.py  # Uses fragment_utils.py
├── jain/                      # Test set: clinical Abs (2-step pipeline)
│   ├── README.md
│   ├── step1_convert_excel_to_csv.py
│   ├── step2_preprocess_p5e_s2.py
│   ├── step3_extract_fragments.py  # Uses fragment_utils.py
│   └── validate_conversion.py      # Uses validation_utils.py
└── shehata/                   # Test set: paired Abs (2-step pipeline)
    ├── README.md
    ├── step1_convert_excel_to_csv.py
    ├── step2_extract_fragments.py  # Uses fragment_utils.py
    └── validate_conversion.py      # Uses validation_utils.py
```

---

## Shared Utilities (Phase D Refactoring - Nov 2025)

To eliminate code duplication across dataset preprocessing scripts, we extracted common functionality into shared utilities:

### `fragment_utils.py`

**Purpose:** Centralized ANARCI annotation and fragment extraction

**Key Functions:**
- `annotate_sequence(seq_id, sequence, chain)` - Annotate single sequence with ANARCI (IMGT numbering)
- `process_sequences_to_fragments(df, ...)` - Batch process DataFrame of antibodies to extract fragments

**Design Principle:** Always use `annotation.sequence_aa` for full V-domain sequences, NOT fragment reconstruction. This prevents data loss from strict IMGT boundary definitions (see [Sequence Handling & ANARCI Gotchas](../docs/developer-guide/preprocessing-internals.md#sequence-handling--anarci-gotchas)).

**Used by:**
- `boughter/annotation/annotator.py`
- `harvey/step2_extract_fragments.py`
- `jain/step3_extract_fragments.py`
- `shehata/step2_extract_fragments.py`

### `validation_utils.py`

**Purpose:** Shared validation logic for preprocessing pipelines

**Key Functions:**
- `validate_dataframe_with_schema(df, schema, dataset_name)` - Pandera-based DataFrame validation
- `validate_file_exists(path)`, `validate_directory_exists(path)` - File system checks
- `calculate_checksum(file_path)` - SHA256 integrity verification
- `calculate_label_stats(df)`, `log_label_stats(stats, dataset_name)` - Label statistics helpers

**Used by:**
- `boughter/validate_stages2_3.py`
- `jain/validate_conversion.py`
- `shehata/validate_conversion.py`

### `paths.py`

**Purpose:** Centralized path constants for consistent directory structure

**Contains:** Default paths for raw, processed, canonical, and fragment data directories

### `logging_config.py`

**Purpose:** Shared logging configuration across all preprocessing scripts

**Provides:** `setup_logger(name)` function for consistent log formatting

---

## Design Philosophy

### Dataset-Centric Organization with Shared Utilities

**Principle:** Each dataset owns its complete preprocessing pipeline, but shares common utilities.

**Benefits:**
1. **Discoverability:** "How do I preprocess Harvey?" → `preprocessing/harvey/`
2. **Maintainability:** Bug in Jain? → All scripts in `preprocessing/jain/`
3. **Consistency:** All datasets follow same pattern AND share validation/annotation logic
4. **Documentation:** Each dataset has complete pipeline README
5. **No Duplication:** Shared ANARCI/validation logic in one place (~1600 lines deduplicated)
6. **Isolation:** Changes to one dataset don't affect others (dataset-specific scripts)

**Follows industry standards:**
- HuggingFace datasets (dataset-centric structure)
- TensorFlow datasets (dataset-specific preprocessing)
- PyTorch torchvision (one file per dataset)
- **WITH** shared utilities for common operations (like sklearn's `preprocessing` module)

---

## Common Preprocessing Stages

### Stage 1: Format Conversion
- **Purpose:** Convert raw data (Excel, FASTA, CSV) to standardized CSV
- **Output:** `data/test/{dataset}/processed/*.csv`

### Stage 2: Fragment Extraction
- **Purpose:** Annotate with ANARCI, extract CDRs/FWRs
- **Output:** `data/test/{dataset}/fragments/*.csv` or `canonical/*.csv`

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

## Running Preprocessing Scripts

**IMPORTANT:** All preprocessing scripts must be run from the project root directory.

### Why?
Some scripts import from the `preprocessing` package (e.g., validation scripts):
```python
from preprocessing.jain.step1_convert_excel_to_csv import calculate_flags
```

This requires the project root to be in PYTHONPATH.

### How to Run:
```bash
# ✅ RECOMMENDED (run as module):
uv run python -m preprocessing.jain.validate_conversion

# ✅ ALTERNATIVE (from project root):
uv run python preprocessing/jain/validate_conversion.py

# ❌ WRONG (from subdirectory):
cd preprocessing/jain && python validate_conversion.py  # ModuleNotFoundError
```

### Technical Details:
- Running as a module (`-m`) ensures `preprocessing` is treated as a package, allowing absolute imports to work correctly.
- `uv run` automatically adds project root to PYTHONPATH.

### Affected Scripts:
- `preprocessing/jain/validate_conversion.py` (imports from step1)
- Any future scripts that import from preprocessing package

---

## References

- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models
- **ANARCI:** Antibody numbering and receptor classification
- **IMGT:** International ImMunoGeneTics information system

---

**Last Updated:** 2025-11-20 (Phase D refactoring: shared utilities added)
**Status:** ✅ Production Ready
