# Boughter Dataset Preprocessing Pipeline

**Complete 3-stage reproducible pipeline for processing the Boughter 2020 dataset**

---

## Overview

This directory contains the complete preprocessing pipeline for the Boughter et al. 2020 antibody polyreactivity dataset, following the Novo Nordisk methodology (Sakhnini et al. 2025).

**📋 For complete data provenance and source citations, see:**
[`train_datasets/BOUGHTER_DATA_PROVENANCE.md`](../../train_datasets/BOUGHTER_DATA_PROVENANCE.md)

### Pipeline Flow

```
Raw FASTA (1,171 DNA sequences)
    ↓
[Stage 1: DNA Translation & Novo Flagging]
    ↓
train_datasets/boughter/processed/boughter.csv (1,117 protein sequences)
    ↓
[Stages 2+3: ANARCI Annotation + QC Filtering]
    ↓
train_datasets/boughter/*.csv (16 fragments × 1,065 sequences)
    ↓
Training Subset: VH_only_boughter_training.csv (914 sequences)
```

---

## Scripts

### 1. Stage 1: DNA Translation & Novo Flagging

**File:** `stage1_dna_translation.py`

Translates raw DNA FASTA files to protein sequences and applies Novo Nordisk flagging strategy.

```bash
python3 preprocessing/boughter/stage1_dna_translation.py
```

**Inputs:**
- `train_datasets/boughter/raw/*.txt` (1,171 DNA sequences)

**Outputs:**
- `train_datasets/boughter/processed/boughter.csv` (1,117 protein sequences, 95.4% success)
- `train_datasets/boughter/raw/translation_failures.log` (54 failures)

**Flagging Strategy:**
- **0 flags** → Specific (label=0, include in training)
- **1-3 flags** → Mildly polyreactive (exclude from training)
- **4+ flags** → Non-specific (label=1, include in training)

---

### 2. Stages 2+3: ANARCI Annotation + QC Filtering

**File:** `stage2_stage3_annotation_qc.py`

Annotates sequences with ANARCI (IMGT numbering), applies QC filtering, and creates 16 fragment-specific CSV files.

```bash
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py
```

**Inputs:**
- `train_datasets/boughter/processed/boughter.csv` (1,117 sequences from Stage 1)

**Outputs:**
- `train_datasets/boughter/annotated/*_boughter.csv` (16 fragment files, 1,065 rows each)
- `train_datasets/boughter/canonical/VH_only_boughter_training.csv` (914 training sequences)
- `train_datasets/boughter/annotated/annotation_failures.log` (7 ANARCI failures)
- `train_datasets/boughter/annotated/qc_filtered_sequences.txt` (45 QC-filtered sequences)

**16 Fragment Types:**
1. VH_only, VL_only (full variable domains)
2. H-CDR1, H-CDR2, H-CDR3 (heavy chain CDRs)
3. L-CDR1, L-CDR2, L-CDR3 (light chain CDRs)
4. H-CDRs, L-CDRs (concatenated CDRs)
5. H-FWRs, L-FWRs (concatenated frameworks)
6. VH+VL (paired variable domains)
7. All-CDRs, All-FWRs (all concatenated)
8. Full (alias for VH+VL)

**Quality Control:**
- Remove sequences with X (unknown amino acid) in any CDR
- Remove sequences with empty CDRs
- Apply IMGT strict boundaries (CDR-H3: positions 105-117, excludes position 118)

---

## Validation

### Validate Stage 1

**File:** `validate_stage1.py`

Validates DNA translation and Novo flagging.

```bash
python3 preprocessing/boughter/validate_stage1.py
```

**Checks:**
- Sequence count (1,117 expected)
- Valid protein characters
- Correct Novo flagging (0, 4+)
- Required columns present

---

### Validate Stages 2+3

**File:** `validate_stages2_3.py`

Validates ANARCI annotation and fragment extraction.

```bash
python3 preprocessing/boughter/validate_stages2_3.py
```

**Checks:**
- 16 fragment files exist
- Each fragment has 1,065 rows
- `include_in_training` flag present
- Training subset has 914 rows
- No empty sequences
- Label distribution correct

---

## Pipeline Statistics

### Stage 1: DNA Translation
- **Input:** 1,171 raw DNA sequences
- **Output:** 1,117 protein sequences
- **Success Rate:** 95.4%
- **Losses:** 54 translation failures

### Stage 2: ANARCI Annotation
- **Input:** 1,117 protein sequences
- **Output:** 1,110 annotated sequences
- **Success Rate:** 99.4%
- **Losses:** 7 ANARCI annotation failures

### Stage 3: Quality Control
- **Input:** 1,110 annotated sequences
- **Output:** 1,065 QC-passed sequences
- **Retention Rate:** 95.9%
- **Losses:** 45 sequences (X in CDRs or empty CDRs)

### Training Subset
- **Total QC-passed:** 1,065 sequences
- **Training eligible:** 914 sequences (include_in_training=True)
- **Excluded:** 151 sequences (1-3 flags, mildly polyreactive)
- **Breakdown:** 443 specific (0 flags) + 471 non-specific (4+ flags)

---

## Running the Full Pipeline

### Option 1: Manual (step-by-step with validation)

```bash
# Stage 1: DNA Translation
python3 preprocessing/boughter/stage1_dna_translation.py
python3 preprocessing/boughter/validate_stage1.py

# Stages 2+3: ANARCI + QC
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py
python3 preprocessing/boughter/validate_stages2_3.py
```

### Option 2: Automated (future enhancement)

```bash
# Coming soon: automated pipeline orchestration
./preprocessing/boughter/run_full_pipeline.sh
```

---

## Output Files

### Intermediate Files

- `train_datasets/boughter/processed/boughter.csv` (Stage 1 output, 1,117 rows)

### Fragment Files (16 total)

All fragment files contain **1,065 rows** with the following columns:
- `id` - Unique antibody identifier
- `sequence` - Fragment sequence
- `label` - Binary label (0=specific, 1=non-specific)
- `subset` - Source dataset (flu, hiv_nat, etc.)
- `num_flags` - Polyreactivity flag count (0-7)
- `flag_category` - Flag category (specific, mild, non-specific)
- `include_in_training` - Training eligibility flag (True/False)
- `source` - Dataset source (boughter2020)
- `sequence_length` - Length of fragment sequence

### Training File

- `train_datasets/boughter/canonical/VH_only_boughter_training.csv` (914 rows, training subset only)

### Log Files

- `train_datasets/boughter/annotated/annotation_failures.log` - ANARCI annotation failures
- `train_datasets/boughter/annotated/qc_filtered_sequences.txt` - QC-filtered sequence IDs
- `train_datasets/boughter/annotated/validation_report.txt` - Validation summary
- `train_datasets/boughter/raw/translation_failures.log` - DNA translation failures

---

## Configuration

The training pipeline uses:

```yaml
# conf/config.yaml
data:
  train_file: ./train_datasets/boughter/canonical/VH_only_boughter_training.csv
```

**Important:** Training uses the filtered 914-sequence subset, NOT the full 1,065-sequence fragment files.

---

## Documentation

### Data Provenance & Sources

**📋 [`train_datasets/BOUGHTER_DATA_PROVENANCE.md`](../../train_datasets/BOUGHTER_DATA_PROVENANCE.md)**
- Complete data lineage from AIMS_manuscripts repository
- All source citations (Boughter 2020, Guthmiller 2020, etc.)
- Directory structure and file mapping
- Processing pipeline statistics
- Quality control metrics

### Pipeline Documentation

Detailed documentation available in `docs/boughter/`:

- `boughter_data_sources.md` - Complete methodology and requirements
- `BOUGHTER_P0_FIX_REPORT.md` - CDR boundary validation
- `boughter_processing_implementation.md` - Implementation details
- `boughter_processing_status.md` - Processing status and validation

---

## Reproducibility

### Byte-for-Byte Reproducibility

This pipeline is designed for **complete reproducibility** - running the same input through the pipeline will produce **byte-for-byte identical outputs**.

**Design Choice: No Processing Timestamps**

Fragment CSV files do NOT include processing date comments in headers. This ensures:
- ✅ `diff` comparisons show only actual data changes
- ✅ `md5sum` / `sha256sum` validation works cleanly
- ✅ Git diffs highlight real changes, not timestamp noise

**Rationale:**
- Git commit timestamps already track when files were processed
- Processing dates in file headers create false diff signals
- Clean byte-for-byte comparisons enable robust validation

**Verification:**
```bash
# Re-run full pipeline
python3 preprocessing/boughter/stage1_dna_translation.py
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py

# Verify outputs are identical
diff train_datasets/boughter/processed/boughter.csv train_datasets/boughter_BACKUP.csv
# Result: No differences (if input unchanged)
```

**What IS included in headers:**
- Dataset name and fragment type
- CDR extraction method (ANARCI/IMGT)
- CDR boundary definitions
- Sequence count statistics
- Reference documentation

**What is NOT included:**
- Processing dates/timestamps
- System information
- User information

This design enables clean reproducibility validation and version control.

---

## References

- **Boughter et al. (2020)** - "Biochemical patterns of antibody polyreactivity revealed through a bioinformatics-based analysis of CDR loops." *eLife* 9:e61393
- **Sakhnini et al. (2025)** - "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters." (Journal pending)

---

## Troubleshooting

### Stage 1 Issues

**Problem:** Translation failures exceed 5%
- Check FASTA file formatting
- Verify correct genetic code (standard code)
- Check for corrupted DNA sequences

**Problem:** Unexpected sequence counts
- Verify raw FASTA files (should be 1,171 total)
- Check translation_failures.log for patterns

### Stage 2+3 Issues

**Problem:** ANARCI failures exceed 1%
- Check input sequences are valid proteins
- Verify ANARCI/riot_na installation
- Check IMGT database accessibility

**Problem:** QC filtering removes >5% of sequences
- Check for unusual X (unknown amino acid) patterns
- Verify CDR extraction is working correctly
- Review qc_filtered_sequences.txt for patterns

### Validation Issues

**Problem:** Sequence count mismatches
- Re-run pipeline from failing stage
- Check for file corruption
- Verify intermediate file integrity

---

**Last Updated:** 2025-11-04
**Pipeline Version:** 2.0 (reorganized structure)
**Status:** ✅ Production Ready
