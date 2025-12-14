# Harvey Dataset - Raw Source Files

**DO NOT MODIFY THESE FILES - Original sources only**

---

## Files

### Main Data (Harvey et al. 2022)

**high_polyreactivity_high_throughput.csv**
- 71,772 nanobodies + header = 71,773 lines
- Label: 1 (high polyreactivity)
- Source: Harvey et al. 2022 deep sequencing (high PSR pool)

**low_polyreactivity_high_throughput.csv**
- 69,702 nanobodies + header = 69,703 lines
- Label: 0 (low polyreactivity)
- Source: Harvey et al. 2022 deep sequencing (low PSR pool)

**low_throughput_polyspecificity_scores_w_exp.csv** (optional)
- 48 nanobodies + header = 49 lines
- Low-throughput experimental validation data
- Not used in current pipeline

---

## File Format

Each CSV contains IMGT-numbered positions (columns 1-128) and pre-extracted CDR sequences:

**Columns:**
- `1` to `128` - IMGT position columns (amino acids or gaps `-`)
- `CDR1_nogaps` - Pre-extracted H-CDR1 (no gaps)
- `CDR2_nogaps` - Pre-extracted H-CDR2 (no gaps)
- `CDR3_nogaps` - Pre-extracted H-CDR3 (no gaps)
- Additional metadata columns

---

## Conversion to CSV

To convert raw CSVs to combined dataset:

```bash
python3 preprocessing/harvey/step1_convert_raw_csvs.py
```

**Input:**
- `data/test/harvey/raw/high_polyreactivity_high_throughput.csv`
- `data/test/harvey/raw/low_polyreactivity_high_throughput.csv`

**Output:**
- `data/test/harvey/processed/harvey.csv`

**Processing steps:**
1. Read high and low polyreactivity CSVs
2. Extract full sequences from IMGT position columns (remove gaps)
3. Assign binary labels (0=low, 1=high)
4. Combine into single dataset (141,474 nanobodies)
5. Save to processed/harvey.csv

---

## Label Assignment

**Binary labels:**
- `label=0` (low polyreactivity) → 69,702 nanobodies from low_polyreactivity CSV
- `label=1` (high polyreactivity) → 71,772 nanobodies from high_polyreactivity CSV

**Methodology:** FACS sorting with PSR assay + deep sequencing (Harvey et al. 2022)

---

## Data Provenance

- **Source:** `debbiemarkslab/nanobody-polyreactivity` (Harvey et al. 2022)
- **Original location:** `reference_repos/harvey_official_repo/backend/app/experiments/`
- **Copied to:** `data/test/harvey/raw/` (for self-contained dataset structure)
- **Date copied:** 2025-11-05
- **Last verified:** 2025-11-05

**Why copied from reference_repos:**
- Ensures `data/test/` is self-contained
- Consistent with Shehata/Jain 4-tier structure
- No external dependencies for data pipeline

---

## Citations

**Dataset Source:**

Harvey, E.P., et al. (2022). "An in silico method to assess antibody fragment polyreactivity." *Nature Communications* 13, 7554.
DOI: 10.1038/s41467-022-35276-4

---

**See:** `../README.md` for complete dataset documentation
