# Shehata Dataset Data Sources

**Date:** 2025-10-31 (Updated: 2025-11-06)
**Status:** ✅ **Complete - Data source verified and pipeline operational**

---

## Raw Data Files

The Shehata dataset preprocessing requires the following Excel files from the original paper's supplementary materials:

### Required Files

| File | Description | Source |
|------|-------------|--------|
| `shehata-mmc2.xlsx` | Supplementary Table S1: Antibody Sequences and Properties | [Shehata et al. 2019 Cell Reports](https://doi.org/10.1016/j.celrep.2019.08.056) |
| `shehata-mmc3.xlsx` | Supplementary Table S2 (not used in current pipeline) | Same source |
| `shehata-mmc4.xlsx` | Supplementary Table S3 (not used in current pipeline) | Same source |
| `shehata-mmc5.xlsx` | Supplementary Table S4 (not used in current pipeline) | Same source |

### Download Instructions

1. Visit the Cell Reports journal page for the paper
2. Navigate to "Supplementary Materials"
3. Download `mmc2.xlsx` through `mmc5.xlsx`
4. Rename them to `shehata-mmc2.xlsx`, `shehata-mmc3.xlsx`, etc.
5. Place them in `test_datasets/shehata/raw/` directory

**Note:** These Excel files are NOT committed to git (see `.gitignore`). They must be downloaded manually from the paper's supplementary materials.

### Paper Reference

**Shehata, L., et al. (2019).** "Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability." *Cell Reports* 28(13):3300-3308.e4.

DOI: https://doi.org/10.1016/j.celrep.2019.08.056

**Note:** This is the original dataset source. Novo Nordisk (Sakhnini et al. 2025) later used this dataset as a test set for their antibody non-specificity prediction models.

### Processed Data

After downloading the raw Excel files, run the preprocessing scripts to generate the cleaned CSV files:

```bash
# Phase 1: Convert Excel to CSV
python3 preprocessing/shehata/step1_convert_excel_to_csv.py

# Phase 2: Extract antibody fragments
python3 preprocessing/shehata/step2_extract_fragments.py
```

This will generate:
- `test_datasets/shehata/processed/shehata.csv` (Phase 1 output)
- `test_datasets/shehata/fragments/*.csv` (16 fragment-specific files)
