# Jain Dataset Data Sources

## Raw Data Files

The Jain dataset preprocessing requires the following Excel files from the paper's supplementary materials:

### Required Files

| File | Description | Source |
|------|-------------|--------|
| `pnas.1616408114.sd01.xlsx` | Clinical antibody metadata (137 antibodies) | [Jain et al. 2017 PNAS Supplementary Data 1](https://www.pnas.org/doi/suppl/10.1073/pnas.1616408114/suppl_file/pnas.1616408114.sd01.xlsx) |
| `pnas.1616408114.sd02.xlsx` | Antibody VH/VL sequences (137 antibodies) | [Jain et al. 2017 PNAS Supplementary Data 2](https://www.pnas.org/doi/suppl/10.1073/pnas.1616408114/suppl_file/pnas.1616408114.sd02.xlsx) |
| `pnas.1616408114.sd03.xlsx` | Biophysical measurements including PSR SMP scores (139 entries) | [Jain et al. 2017 PNAS Supplementary Data 3](https://www.pnas.org/doi/suppl/10.1073/pnas.1616408114/suppl_file/pnas.1616408114.sd03.xlsx) |

### File Structure

#### SD01 - Metadata (137 antibodies)
**Columns:**
- `Name`: Antibody name
- `Light chain class`: kappa or lambda
- `Type`: HU (humanized), CH (chimeric), ZU, etc.
- `Original mAb Isotype or Format`: IgG1, IgG2, etc.
- `Clinical Status`: Approved, Phase 1/2/3, Discontinued
- `Phagec`: Yes/No (discovered via phage display)
- `Year Name Proposed`: Year
- `Notes`: Additional information

#### SD02 - Sequences (137 antibodies)
**Columns:**
- `Name`: Antibody name (matches SD01)
- `VH`: Heavy chain variable region sequence
- `VL`: Light chain variable region sequence
- `LC Class`: kappa or lambda
- `Source`: WHO-INN, PDB, etc.
- `Source Detaileda`: Publication list or PDB ID
- `Disclaimers and Known Issues`: Data quality notes
- `Notes`: Additional information

#### SD03 - Biophysical Properties (139 entries)
**Columns:**
- `Name`: Antibody name (137 match SD01/SD02, 2 extra entries are metadata)
- `HEK Titer (mg/L)`: Expression level
- `Fab Tm by DSF (°C)`: Thermal stability
- `SGAC-SINS AS100 ((NH4)2SO4 mM)`: Self-association measurement
- `HIC Retention Time (Min)`: Hydrophobic interaction chromatography
- `SMAC Retention Time (Min)`: Self-interaction chromatography
- `Slope for Accelerated Stability`: Stability metric
- **`Poly-Specificity Reagent (PSR) SMP Score (0-1)`**: **PRIMARY NON-SPECIFICITY METRIC** (0 = specific, 1 = non-specific)
- `Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average`: Self-interaction
- `CIC Retention Time (Min)`: Cross-interaction chromatography
- `CSI-BLI Delta Response (nm)`: Self-interaction measurement
- `ELISA`: ELISA-based polyreactivity score
- `BVP ELISA`: Baculovirus particle ELISA score

### Download Instructions

1. Visit the PNAS paper page: https://www.pnas.org/doi/10.1073/pnas.1616408114
2. Navigate to "Supporting Information"
3. Download all three supplementary data files:
   - `pnas.1616408114.sd01.xlsx`
   - `pnas.1616408114.sd02.xlsx`
   - `pnas.1616408114.sd03.xlsx`
4. Place them in `test_datasets/` directory

**Note:** These Excel files are NOT committed to git (see `.gitignore`). They must be downloaded manually from the paper's supplementary materials.

### Paper Reference

**Jain, T., Sun, T., Durand, S., et al. (2017).** "Biophysical properties of the clinical-stage antibody landscape." *Proceedings of the National Academy of Sciences*, 114(5), 944-949.

DOI: https://doi.org/10.1073/pnas.1616408114

### Usage in Sakhnini et al. 2025

The Sakhnini et al. 2025 paper uses this dataset as a **test set** for evaluating their non-specificity prediction models:

> "Four different datasets were retrieved from public sources... (ii) 137 clinical-stage IgG1-formatted antibodies with their respective non-specificity flag from ELISA with a panel of common antigens"

**Key Methodology Notes from Sakhnini:**
- Dataset is referred to as "Jain dataset" in Sakhnini paper
- Used for external validation of models trained on Boughter dataset
- Parsed into groups: specific (0 flags), mildly non-specific (1-3 flags), non-specific (>3 flags)
- Primary assay: ELISA with panel of 6 ligands (ssDNA, dsDNA, insulin, LPS, cardiolipin, KLH)

### Expected Output

After running the conversion scripts, we expect:
- **137 antibodies** (strict SD01∩SD02∩SD03 match after dropping SD03 metadata rows)
- Columns (ordered):
  - `id`, `heavy_seq`, `light_seq`
  - `flags_total`, `flag_category`, `label`
  - `source`, `smp`, `elisa`, `bvp_elisa`
  - Supporting assay columns (`psr`, `acsins`, `csi_bli`, `cic`, `hic`, `smac`, `sgac_sins`, `as_slope`) for reproducibility
  - Optional QC columns (`heavy_seq_length`, `light_seq_length`)
- Source label: `jain2017`

**Column Derivation (RESOLVED 2025-11-01):**
- `id`: SD0X `Name`
- `heavy_seq` / `light_seq`: SD02 `VH` / `VL` (sanitized for gaps/whitespace)
- `flags_total`: Integer count (0–4) produced by Table 1 thresholds (see below)
- `flag_category`: `specific` (0), `mild` (1–3), `non_specific` (4)
- `label`: Binary view used by Sakhnini
  - 0 → `specific`
  - 1 → `non_specific`
  - `NaN` → `mild` (kept in CSV but excluded during model evaluation)
- `source`: Constant `"jain2017"`
- `smp`: SD03 `Poly-Specificity Reagent (PSR) SMP Score (0-1)`
- `elisa`: SD03 `ELISA` fold-over-background (polyreactivity against 6 antigens: cardiolipin, KLH, LPS, ssDNA, dsDNA, insulin)
- `bvp_elisa`: SD03 `BVP ELISA` (needed for flag audit)
- Remaining assay columns: direct passthrough from SD03 with standardized snake_case names
- QC columns: derived lengths and any anomaly flags emitted by converter (optional)

**Flag Calculation (for `label` column):**

Flags are calculated from 4 assay groups using 90th percentile thresholds of approved antibodies:

| Cluster | Assays | Threshold (approved 90th percentile) | Condition | Flag logic |
|---------|--------|---------------------------------------|-----------|------------|
| Self-interaction / cross-interaction | PSR SMP, AC-SINS Δλ, CSI-BLI, CIC | 0.27, 11.8 nm, 0.01 RU, 10.1 min | `>` for all | Flag = 1 if **any** exceed |
| Chromatography / salt stress | HIC, SMAC, SGAC-SINS | 11.7 min, 12.8 min, 370 mM | `>`, `>`, `<` respectively | Flag = 1 if HIC/SMAC exceed or SGAC-SINS falls below |
| Polyreactivity / plate binding | ELISA, BVP ELISA | 1.9, 4.3 fold-over-background | `>` | Flag = 1 if ELISA or BVP exceeds |
| Accelerated stability | AS SEC slope | 0.08 % loss/day | `>` | Flag = 1 if AS exceeds |

- `flags_total` = sum of cluster flags (range 0–4)
- `flag_category` assignment:
  - 0 → `specific`
  - 1–3 → `mild`
  - 4 → `non_specific`
- `label` = 0 for `specific`, 1 for `non_specific`, `NaN` for `mild`
- Additional metadata (approved-antibody percentile calculations) available in `docs/jain_conversion_verification_report.md` (to be generated)

### Processed Data Workflow

```bash
# Phase 1: Convert PNAS Excel files to CSV
python3 scripts/convert_jain_excel_to_csv.py

# Phase 2: Extract antibody fragments
python3 preprocessing/process_jain.py
```

This will generate:
- `test_datasets/jain.csv` (Phase 1 output - properly converted from PNAS files)
- `test_datasets/jain/*.csv` (16 fragment-specific files from Phase 2)

## Data Provenance Issue (Discovered 2025-11-01)

**Problem:** The existing `test_datasets/jain.csv` in the repository contains only 80 antibodies (not 137), and values don't match the PNAS supplementary files.

**Example discrepancy:**
- `jain.csv` abituzumab `smp`: 0.126
- PNAS SD03 `PSR SMP Score`: 0.167

**Root cause:** Unknown - the existing `jain.csv` origin is unclear. It may have been:
1. Incorrectly converted from PNAS files
2. Provided by Sakhnini et al. with additional filtering
3. From a different source entirely

**Solution:** This documentation and conversion scripts ensure we create the correct dataset from the authoritative PNAS source files.

## ELISA Data Limitation (Identified 2025-11-03)

**Summary:** The publicly available Jain PNAS supplementary files contain only aggregated ELISA values, not per-antigen measurements. This represents a potential deviation from Novo's methodology.

### What the Experimental Protocol Did

From Jain et al. 2017 PNAS Supplementary Information:
> "six different antigens, cardiolipin (50 μg/mL), KLH (5 μg/mL), LPS (10 μg/mL), ssDNA (1 μg/mL), dsDNA (1 μg/mL), and insulin (5 μg/mL), were coated onto ELISA plates individually at 50 μL per well"

**Each antibody was tested against 6 separate antigens in individual wells.**

### What's in Public SD03 File

The public PNAS SD03 supplementary file contains:
- Single `ELISA` column (fold-over-background, aggregated across antigens)
- Single `BVP ELISA` column
- **NO individual per-antigen columns** (cardiolipin, KLH, LPS, ssDNA, dsDNA, insulin)

Confirmed by inspection:
```bash
python3 -c "import pandas as pd; df = pd.read_excel('test_datasets/jain-pnas.1616408114.sd03.xlsx'); print(df.columns.tolist())"
# Output: ['Name', 'HEK Titer (mg/L)', 'Fab Tm by DSF (°C)', ..., 'ELISA', 'BVP ELISA']
# No individual antigen columns present
```

### Methodology Implications

**Our approach (publicly reproducible):**
- Use aggregated `ELISA` column from SD03
- Flag polyreactivity if aggregated value > 1.9 fold-over-background
- Matches what's available in public supplementary files

**Potential Novo approach (per Hybri comment 2025-11-03):**
- May have obtained disaggregated per-antigen values via author communication
- Flag polyreactivity if **any of 6 individual antigens** exceeds threshold
- More sensitive (OR logic across 6 measurements vs. single aggregated value)

**Impact:** Using max-of-6-values (Novo) vs aggregated-value (us) could produce different polyreactivity flags, affecting the flag_polyreactivity cluster and overall label assignment.

### Verification Status

- ✅ **Confirmed:** Public SD03 contains only aggregated ELISA column
- ⚠️ **Unconfirmed:** Whether Novo used disaggregated per-antigen data
- 📝 **Action:** Document as known limitation; we use publicly available data

**Reference:** Discord discussion with Hybri, 2025-11-03, where Novo methodology was clarified as using "6 individual ELISA raw values" obtained via email with paper authors.
