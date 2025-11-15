# Shehata Dataset - Raw Source Files

**DO NOT MODIFY THESE FILES - Original sources only**

---

## Files

### Main Data (Shehata et al. 2019 Cell Reports)

**Citation:** Shehata L, Thaventhiran JED, Engelhardt KR, et al. (2019). "Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability." *Cell Reports* 28(13):3300-3308.e4. DOI: 10.1016/j.celrep.2019.08.056

- `shehata-mmc2.xlsx` - **Supplementary Table S1: Antibody Sequences and Properties**
  - 402 rows (398 antibodies used after filtering)
  - Paired VH and VL sequences
  - PSR (Polyspecific Reagent) scores
  - Biophysical properties (Tm, charge, pI, etc.)
  - **Note:** 4 sequences removed during processing (incomplete pairing or missing PSR)

### Unused Files (Archived for Provenance)

- `shehata-mmc3.xlsx` - **Supplementary Table S2** (not used in current pipeline)
- `shehata-mmc4.xlsx` - **Supplementary Table S3** (not used in current pipeline)
- `shehata-mmc5.xlsx` - **Supplementary Table S4** (not used in current pipeline)

**Why kept:** Complete data provenance. These files are part of the original dataset but not required for polyspecificity prediction.

---

## Conversion to CSV

To convert the main Excel file to CSV format:

```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
```

**Input:** `data/test/shehata/raw/shehata-mmc2.xlsx`
**Output:** `data/test/shehata/processed/shehata.csv`

**Processing steps:**
1. Read Excel file (402 rows)
2. Extract VH and VL sequences
3. Extract PSR scores
4. Filter out 4 sequences with missing data
5. Apply PSR threshold (98.24th percentile) for binary labels
6. Save 398 antibodies to CSV

---

## Label Assignment (Sakhnini 2025 Methodology)

**Threshold:** 98.24th percentile of PSR score distribution

**Binary labels:**
- `label=0` (specific): PSR < 98.24th percentile → 391 antibodies
- `label=1` (non-specific): PSR ≥ 98.24th percentile → 7 antibodies

**Methodology source:** Sakhnini A, et al. (2025). "Antibody Non-Specificity Prediction using Protein Language Models and Biophysical Features." *Cell*. DOI: 10.1016/j.cell.2024.12.025

---

## Data Provenance

- **Source:** Cell Reports supplementary materials (Shehata 2019)
- **Downloaded:** Original publication supplementary files
- **Date added:** 2025-01-15
- **Last verified:** 2025-11-05

---

**See:** `../README.md` for complete dataset documentation
