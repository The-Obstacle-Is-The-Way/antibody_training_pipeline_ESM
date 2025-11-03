# Data Conversion Scripts

These scripts convert raw data files from various sources into standardized CSV format.

**Purpose:** Essential for reproducibility - these transform Excel/FASTA files into the CSV format used by the training pipeline.

## Scripts

- `convert_boughter_to_csv.py` - Converts Boughter FASTA (DNA) → CSV with protein sequences
- `convert_harvey_csvs.py` - Combines Harvey high/low polyreactivity CSVs
- `convert_jain_excel_to_csv.py` - Converts Jain Excel → CSV  
- `convert_shehata_excel_to_csv.py` - Converts Shehata Excel → CSV

## Usage

Run these BEFORE training if you need to regenerate the CSV files from source data.

**Note:** The main preprocessing pipeline uses `preprocessing/process_*.py` scripts, which may call these conversion scripts internally.
