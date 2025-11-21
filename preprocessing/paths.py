"""
Centralized path configuration for preprocessing scripts.

All data paths for preprocessing pipelines defined here for easy modification.
Delegates to src/antibody_training_esm/settings.py for a Single Source of Truth.
"""

from pathlib import Path

from antibody_training_esm.settings import settings

# Project root
PROJECT_ROOT = settings.PROJECT_ROOT

# Base data directories
DATA_DIR = settings.DATA_DIR
DATA_TRAIN_DIR = settings.DATA_TRAIN_DIR
DATA_TEST_DIR = settings.DATA_TEST_DIR

# ============================================================================
# Boughter (training set)
# ============================================================================
BOUGHTER_DIR = settings.BOUGHTER_DIR
BOUGHTER_RAW_DIR = settings.BOUGHTER_RAW_DIR
BOUGHTER_PROCESSED_DIR = settings.BOUGHTER_PROCESSED_DIR
BOUGHTER_ANNOTATED_DIR = settings.BOUGHTER_ANNOTATED_DIR
BOUGHTER_CANONICAL_DIR = settings.BOUGHTER_CANONICAL_DIR

# Specific files
BOUGHTER_STAGE1_OUTPUT = settings.BOUGHTER_PROCESSED_CSV  # Note: Same path
BOUGHTER_TRAINING_SUBSET = BOUGHTER_CANONICAL_DIR / "VH_only_boughter_training.csv"
BOUGHTER_CANONICAL_CSV = settings.BOUGHTER_CANONICAL_CSV

# ============================================================================
# Jain (test set - Novo parity benchmark)
# ============================================================================
JAIN_DIR = settings.JAIN_DIR
JAIN_RAW_DIR = settings.JAIN_RAW_DIR
JAIN_PROCESSED_DIR = settings.JAIN_PROCESSED_DIR
JAIN_FRAGMENTS_DIR = settings.JAIN_FRAGMENTS_DIR
JAIN_CANONICAL_DIR = settings.JAIN_CANONICAL_DIR

# Specific files
JAIN_RAW_EXCEL = JAIN_RAW_DIR / "jain_clinical_antibodies_with_private_elisa.xlsx"
JAIN_PRIVATE_ELISA_EXCEL = JAIN_RAW_DIR / "Private_Jain2017_ELISA_indiv.xlsx"
JAIN_SD01_EXCEL = JAIN_RAW_DIR / "jain-pnas.1616408114.sd01.xlsx"
JAIN_SD02_EXCEL = JAIN_RAW_DIR / "jain-pnas.1616408114.sd02.xlsx"
JAIN_SD03_EXCEL = JAIN_RAW_DIR / "jain-pnas.1616408114.sd03.xlsx"

JAIN_FULL_CSV = settings.JAIN_FULL_CSV
JAIN_SD03_CSV = settings.JAIN_SD03_CSV
JAIN_ELISA_116_CSV = JAIN_PROCESSED_DIR / "jain_ELISA_ONLY_116.csv"
JAIN_P5E_S2 = JAIN_PROCESSED_DIR / "jain_p5e_s2_preprocessed.csv"
JAIN_86_PARITY_CSV = JAIN_CANONICAL_DIR / "jain_86_novo_parity.csv"
JAIN_VH_ONLY_86_CSV = JAIN_CANONICAL_DIR / "VH_only_jain_86_p5e_s2.csv"

# ============================================================================
# Harvey (test set - nanobodies)
# ============================================================================
HARVEY_DIR = settings.HARVEY_DIR
HARVEY_RAW_DIR = settings.HARVEY_RAW_DIR
HARVEY_PROCESSED_DIR = settings.HARVEY_PROCESSED_DIR
HARVEY_FRAGMENTS_DIR = settings.HARVEY_FRAGMENTS_DIR

# Specific files
HARVEY_HIGH_CSV = settings.HARVEY_HIGH_POLY_CSV
HARVEY_LOW_CSV = settings.HARVEY_LOW_POLY_CSV
HARVEY_FULL_CSV = HARVEY_PROCESSED_DIR / "harvey.csv"
HARVEY_VHH_ONLY = HARVEY_FRAGMENTS_DIR / "VHH_only_harvey.csv"

HARVEY_RAW_NS = HARVEY_RAW_DIR / "nanobody_nonspecific.csv"  # Legacy?
HARVEY_RAW_S = HARVEY_RAW_DIR / "nanobody_specific.csv"  # Legacy?
HARVEY_COMBINED = HARVEY_PROCESSED_DIR / "harvey_combined.csv"  # Legacy?

# ============================================================================
# Shehata (test set - PSR assay)
# ============================================================================
SHEHATA_DIR = settings.SHEHATA_DIR
SHEHATA_RAW_DIR = settings.SHEHATA_RAW_DIR
SHEHATA_PROCESSED_DIR = settings.SHEHATA_PROCESSED_DIR
SHEHATA_FRAGMENTS_DIR = settings.SHEHATA_FRAGMENTS_DIR
SHEHATA_CANONICAL_DIR = settings.SHEHATA_CANONICAL_DIR

# Specific files
SHEHATA_RAW_EXCEL = settings.SHEHATA_EXCEL_PATH
SHEHATA_PROCESSED_CSV = SHEHATA_PROCESSED_DIR / "shehata.csv"
SHEHATA_CANONICAL_CSV = SHEHATA_CANONICAL_DIR / "shehata_398.csv"

# ============================================================================
# Experiments
# ============================================================================
EXPERIMENTS_DIR = settings.EXPERIMENTS_DIR
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
CACHE_DIR = EXPERIMENTS_DIR / "cache"
BENCHMARKS_DIR = EXPERIMENTS_DIR / "benchmarks"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
LOGS_DIR = RUNS_DIR / "logs"


# ============================================================================
# Helper function for dynamic path construction
# ============================================================================
def get_dataset_path(dataset: str, stage: str) -> Path:
    """
    Get standardized dataset path.

    Args:
        dataset: Dataset name (boughter, jain, harvey, shehata)
        stage: Processing stage (raw, processed, fragments, canonical)

    Returns:
        Path object

    Example:
        >>> get_dataset_path("jain", "raw")
        PosixPath('.../data/test/jain/raw')
    """
    dataset_map = {
        "boughter": {
            "raw": BOUGHTER_RAW_DIR,
            "processed": BOUGHTER_PROCESSED_DIR,
            "canonical": BOUGHTER_CANONICAL_DIR,
        },
        "jain": {
            "raw": JAIN_RAW_DIR,
            "processed": JAIN_PROCESSED_DIR,
            "fragments": JAIN_FRAGMENTS_DIR,
            "canonical": JAIN_CANONICAL_DIR,
        },
        "harvey": {
            "raw": HARVEY_RAW_DIR,
            "processed": HARVEY_PROCESSED_DIR,
            "fragments": HARVEY_FRAGMENTS_DIR,
        },
        "shehata": {
            "raw": SHEHATA_RAW_DIR,
            "processed": SHEHATA_PROCESSED_DIR,
            "fragments": SHEHATA_FRAGMENTS_DIR,
            "canonical": SHEHATA_CANONICAL_DIR,
        },
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset}")
    if stage not in dataset_map[dataset]:
        raise ValueError(f"Unknown stage '{stage}' for dataset '{dataset}'")

    return dataset_map[dataset][stage]
