# Shehata Dataset - Raw Source Files

**DO NOT MODIFY THESE FILES - Original sources only**

---

## Files

### Main Data (Shehata et al. 2019 Cell Reports)

**Citation:** Shehata, L. et al. (2019). "Affinity maturation enhances antibody specificity but compromises conformational stability." *Cell Reports* 28(13):3300-3308.e4. DOI: 10.1016/j.celrep.2019.08.056

- `shehata-mmc2.xlsx` - **Supplementary Table S1: Antibody Sequences and Properties**
  - 402 rows total (398 antibodies + 2 legend/metadata rows + 2 antibodies missing PSR)
  - Paired VH and VL sequences
  - PSR (Polyspecific Reagent) scores
  - Biophysical properties (Tm, charge, pI, etc.)
  - **Note:** 4 rows removed during processing (2 non-sequence rows; 2 antibodies missing PSR scores)

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
4. Drop 2 non-sequence legend/metadata rows
5. Drop 2 antibodies without numeric PSR scores
6. Binarize PSR scores into low/high labels (top 7/398 = 98.24th percentile; equivalent here to `psr_score > 0.33`)
7. Save 398 antibodies to CSV

---

## Label Assignment (Shehata 2019; benchmarked in Sakhnini 2025)

**Threshold:** High polyreactivity (label=1) corresponds to `psr_score > 0.33` (Shehata et al. 2019). The conversion script computes a cutoff at the 98.24th percentile to enforce exactly 7/398 non-specific antibodies (Sakhnini et al. 2025); in this dataset that is equivalent to `> 0.33`.

**Binary labels:**
- `label=0` (specific): PSR ≤ 0.33 → 391 antibodies
- `label=1` (non-specific): PSR > 0.33 → 7 antibodies

**Methodology sources:**
- Shehata, L. et al. (2019). *Cell Reports*. Defines high polyreactivity as PSR > 0.33.
- Sakhnini, L.I. et al. (2025). *bioRxiv*. Uses this dataset as a PSR test set and reports 7/398 as non-specific.

---

## Data Provenance

- **Source:** Cell Reports supplementary materials (Shehata 2019)
- **Downloaded:** Original publication supplementary files
- **Date added:** 2025-01-15
- **Last verified:** 2025-11-05

---

**See:** `../README.md` for complete dataset documentation
