# Training Datasets

This directory contains all training datasets for the antibody non-specificity classifier.

## Current Datasets

### Boughter 2020 (Primary Training Dataset)

**Source:** Boughter et al. (2020) eLife 9:e61393
**Size:** 914 training sequences (1,117 original → 1,065 QC-passed → 914 training-eligible)
**Labeling:** Novo Nordisk flagging strategy (0 flags = specific, 4+ flags = non-specific)

**Directory:** `boughter/`

**Key Files:**
- `boughter/canonical/VH_only_boughter_training.csv` - PRIMARY TRAINING FILE (914 sequences)
- `boughter/raw/` - Original DNA sequences from AIMS_manuscripts repository
- `boughter/processed/boughter.csv` - Translated protein sequences (Stage 1 output)
- `boughter/annotated/` - ANARCI-annotated fragments (16 types, Stages 2+3)

**Documentation:**
- [BOUGHTER_DATA_PROVENANCE.md](BOUGHTER_DATA_PROVENANCE.md) - Complete data lineage
- [boughter/README.md](boughter/README.md) - Preprocessing pipeline details

---

## Directory Structure

```
train_datasets/
├── README.md (this file)
├── BOUGHTER_DATA_PROVENANCE.md
└── boughter/
    ├── raw/ - Original DNA FASTA files
    ├── processed/ - Stage 1: Translated proteins
    ├── annotated/ - Stages 2+3: ANARCI fragments
    ├── canonical/ - Authoritative training file (914 sequences, production)
    └── (strict QC experiment archived in experiments/strict_qc_2025-11-04/)
```

---

## Adding New Training Datasets

When adding new training datasets (e.g., from other publications):

1. Create a new subdirectory: `train_datasets/<dataset_name>/`
2. Use the same structure: `raw/`, `processed/`, `annotated/`, `canonical/`
3. Create `<dataset_name>/README.md` documenting the preprocessing pipeline
4. Update this README with the new dataset information
5. Follow the same quality control standards as Boughter

---

**Last Updated:** 2025-11-05
