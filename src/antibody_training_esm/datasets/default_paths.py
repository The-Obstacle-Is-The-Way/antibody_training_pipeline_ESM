"""
Shared default dataset paths.

Centralizes the relative locations of dataset assets so loaders do not hard-code
string literals scattered across modules.

DEPRECATED: Use src/antibody_training_esm/settings.py instead.
This module now delegates to the central settings for backward compatibility.
"""

from antibody_training_esm.settings import settings

BOUGHTER_ANNOTATED_DIR = settings.BOUGHTER_ANNOTATED_DIR
BOUGHTER_PROCESSED_CSV = settings.BOUGHTER_PROCESSED_CSV

HARVEY_OUTPUT_DIR = settings.HARVEY_OUTPUT_DIR
HARVEY_HIGH_POLY_CSV = settings.HARVEY_HIGH_POLY_CSV
HARVEY_LOW_POLY_CSV = settings.HARVEY_LOW_POLY_CSV

JAIN_OUTPUT_DIR = settings.JAIN_OUTPUT_DIR
JAIN_FULL_CSV = settings.JAIN_FULL_CSV
JAIN_SD03_CSV = settings.JAIN_SD03_CSV

SHEHATA_OUTPUT_DIR = settings.SHEHATA_OUTPUT_DIR
SHEHATA_EXCEL_PATH = settings.SHEHATA_EXCEL_PATH
