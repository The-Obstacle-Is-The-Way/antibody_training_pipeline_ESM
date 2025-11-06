# Jain Dataset - Processed Files

All files here are derived from `raw/` using scripts in `scripts/conversion/` and `preprocessing/`.

---

## Conversion (Excel → CSV)

**Script:** `preprocessing/jain/step1_convert_excel_to_csv.py`

### Base Files (137 antibodies)

- `jain_sd01.csv` ← `raw/jain-pnas.1616408114.sd01.xlsx`
  - Sequences + metadata

- `jain_sd02.csv` ← `raw/jain-pnas.1616408114.sd02.xlsx`
  - Biophysical properties (PSR, AC-SINS, HIC, etc.)

- `jain_sd03.csv` ← `raw/jain-pnas.1616408114.sd03.xlsx`
  - Thermal stability (Fab Tm, scFv Tm)

### Merged + ELISA Integration

- `jain.csv` **(137 antibodies)** - Base merged dataset
  - Combines sd01, sd02, sd03
  - No ELISA data yet

- `jain_with_private_elisa_FULL.csv` **(137 antibodies)** - With ELISA flags
  - Adds private ELISA data from `raw/Private_Jain2017_ELISA_indiv.xlsx`
  - Calculates `elisa_flags` (0-6 range)
  - Calculates `total_flags` (0-10 range, for reference)
  - Categorizes: specific (ELISA=0), mild (ELISA 1-3), non-specific (ELISA ≥4)

- `jain_ELISA_ONLY_116.csv` **(116 antibodies)** - ELISA QC filtered
  - Removes ELISA 1-3 ("mild" aggregators)
  - Keeps ELISA 0 (specific) and ELISA ≥4 (non-specific)
  - **This is the SSOT for preprocessing pipelines**

---

## Deprecated Files

- `jain_86_elisa_1.3.csv` - OLD intermediate file from incorrect methodology
  - Used wrong ELISA threshold
  - Kept for historical reference only
  - **DO NOT USE**

---

## Data Flow

```
raw/*.xlsx
  ↓ [convert_jain_excel_to_csv.py]
jain_sd01/02/03.csv → jain.csv
  ↓ [merge Private_ELISA_indiv.xlsx]
jain_with_private_elisa_FULL.csv (137 antibodies)
  ↓ [remove ELISA 1-3]
jain_ELISA_ONLY_116.csv (116 antibodies) ← SSOT
  ↓ [preprocess_jain_p5e_s2.py]
../canonical/jain_86_novo_parity.csv (86 antibodies)
```

---

## Regenerating Files

To regenerate all processed files from raw:

```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/jain/step1_convert_excel_to_csv.py

# Step 2: Create 86-antibody benchmarks
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

---

**See:** `JAIN_COMPLETE_GUIDE.md` (repo root) for complete documentation
