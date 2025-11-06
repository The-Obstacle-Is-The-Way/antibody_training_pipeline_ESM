# Data Conversion Scripts

**⚠️  MOVED:** Conversion scripts have been reorganized.

## New Locations (Dataset-Centric)

Conversion scripts now live with their respective datasets in `preprocessing/`:

- **Boughter:** `preprocessing/boughter/stage1_dna_translation.py`
- **Harvey:** `preprocessing/harvey/step1_convert_raw_csvs.py`
- **Jain:** `preprocessing/jain/step1_convert_excel_to_csv.py`
- **Shehata:** `preprocessing/shehata/step1_convert_excel_to_csv.py`

## Rationale

Following industry-standard **dataset-centric organization**:
- All preprocessing for a dataset lives in ONE directory
- Consistent with HuggingFace, TensorFlow datasets, PyTorch patterns
- Easier to discover and maintain

## Legacy Scripts

See `legacy/` for historical incorrect implementations (archived for reference).
