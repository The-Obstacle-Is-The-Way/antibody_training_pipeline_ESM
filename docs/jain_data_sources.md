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
- **137 antibodies** (from SD01/SD02/SD03 intersection)
- Columns: `id`, `heavy_seq`, `light_seq`, `label`, `source`, `smp`, `ova`
- Source label: `jain2017`

**Column Derivation (RESOLVED 2025-11-01):**
- `id`: From SD01/SD02/SD03 'Name' column
- `heavy_seq`: From SD02 'VH' column
- `light_seq`: From SD02 'VL' column
- `label`: Binary classification derived from flag count:
  - 0 = Specific (0 flags)
  - 1 = Non-specific (≥4 flags)
  - **Exclude:** 1-3 flags (mildly non-specific) during training
- `source`: "jain2017"
- `smp`: From SD03 'Poly-Specificity Reagent (PSR) SMP Score (0-1)' column
- `ova`: From SD03 'ELISA' column (NOT 'BVP ELISA' - see analysis below)

**Flag Calculation (for `label` column):**

Flags are calculated from 4 assay groups using 90th percentile thresholds of approved antibodies:

1. **Polyreactivity (ELISA, BVP):** Flag if ELISA > 1.9 OR BVP > 4.3
2. **Self-Interaction (PSR, CSI, AC-SINS, CIC):** Flag if PSR > 0.26 OR others exceed thresholds
3. **Chromatography (SGAC100, SMAC, HIC):** Flag if any exceeds threshold
4. **Stability (AS):** Flag if AS exceeds threshold

Total flags: 0-4, where ≥4 = non-specific (label=1)

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
