"""
Shared default dataset paths.

Centralizes the relative locations of dataset assets so loaders do not hard-code
string literals scattered across modules.
"""

from pathlib import Path

BOUGHTER_ANNOTATED_DIR = Path("train_datasets/boughter/annotated")
BOUGHTER_PROCESSED_CSV = Path("train_datasets/boughter/boughter_translated.csv")

HARVEY_OUTPUT_DIR = Path("data/test/harvey/fragments")
HARVEY_HIGH_POLY_CSV = Path(
    "data/test/harvey/raw/high_polyreactivity_high_throughput.csv"
)
HARVEY_LOW_POLY_CSV = Path(
    "data/test/harvey/raw/low_polyreactivity_high_throughput.csv"
)

JAIN_OUTPUT_DIR = Path("data/test/jain/fragments")
JAIN_FULL_CSV = Path("data/test/jain/processed/jain_with_private_elisa_FULL.csv")
JAIN_SD03_CSV = Path("data/test/jain/processed/jain_sd03.csv")

SHEHATA_OUTPUT_DIR = Path("data/test/shehata/fragments")
SHEHATA_EXCEL_PATH = Path("data/test/shehata/raw/shehata-mmc2.xlsx")
