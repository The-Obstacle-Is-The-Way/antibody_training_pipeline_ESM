# Tessier 2024 Antibody Polyreactivity Dataset

**Paper:** Feld et al. (2024) "Human antibody polyreactivity is governed primarily by the heavy-chain complementarity-determining regions" *Cell Reports* (Oct 2024)

**Dataset:** 373,457 human antibody sequences with binary polyreactivity labels (CHO cell-based assays)

## Data Not Committed

The processed Tessier dataset files (~500MB total) are not committed to Git to keep the repository lightweight. All data can be regenerated from public sources.

## Regenerating the Dataset

### Prerequisites
1. Download raw data from Zenodo: [https://zenodo.org/records/13791488](https://zenodo.org/records/13791488)
2. Extract to `external_datasets/tessier_2024_polyreactivity/`

### Run Preprocessing Pipeline

```bash
# Stage 1: Extract sequences from Excel (2 min)
uv run python preprocessing/tessier/step1_extract_sequences.py

# Stage 2: ANARCI annotation with IMGT numbering (2-3 hours)
uv run python preprocessing/tessier/step2_annotate_anarci.py

# Stage 3: QC filters and train/val split (1 hour)
uv run python preprocessing/tessier/step3_qc_and_split.py
```

### Output Structure

```
train_datasets/tessier/
├── processed/
│   └── tessier_raw.csv                    # 373k sequences
├── annotated/
│   ├── tessier_annotated.csv              # ~370k sequences (99% annotation success)
│   └── annotation_failures.log
├── canonical/
│   ├── VH_only_tessier_training.csv       # ~296k sequences (80% train)
│   ├── VH_only_tessier_validation.csv     # ~74k sequences (20% val)
│   └── [30 other fragment CSV files]
└── README.md
```

## Dataset Statistics

- **Total sequences:** 373,457 (246k expected, got 52% more from S1+S2!)
- **Polyreactive (label=1):** 208,117 (55.7%)
- **Specific (label=0):** 165,340 (44.3%)
- **VH length:** 122 ± 3 aa (range: 112-131)
- **VL length:** 108 ± 2 aa (range: 103-114)

## Usage for Transfer Learning

This dataset is used for transfer learning to improve polyreactivity prediction on the Ginkgo GDPa1 dataset (197 antibodies).

**Strategy:**
1. Pre-train model on Tessier (373k CHO antibodies)
2. Fine-tune on GDPa1 (197 PR_CHO labels)
3. **Expected improvement:** 0.50664 → 0.70+ Spearman

See `TESSIER_PREPROCESSING_PLAN.md` for full details.
