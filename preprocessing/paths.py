"""
Centralized path configuration for preprocessing scripts.

All data paths for preprocessing pipelines defined here for easy modification.
Follows same pattern as src/antibody_training_esm/datasets/default_paths.py.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Base data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_TRAIN_DIR = DATA_DIR / "train"
DATA_TEST_DIR = DATA_DIR / "test"

# ============================================================================
# Boughter (training set)
# ============================================================================
BOUGHTER_DIR = DATA_TRAIN_DIR / "boughter"
BOUGHTER_RAW_DIR = BOUGHTER_DIR / "raw"
BOUGHTER_PROCESSED_DIR = BOUGHTER_DIR / "processed"
BOUGHTER_ANNOTATED_DIR = BOUGHTER_DIR / "annotated"
BOUGHTER_CANONICAL_DIR = BOUGHTER_DIR / "canonical"

# Specific files
BOUGHTER_STAGE1_OUTPUT = BOUGHTER_PROCESSED_DIR / "boughter.csv"
BOUGHTER_TRAINING_SUBSET = BOUGHTER_CANONICAL_DIR / "VH_only_boughter_training.csv"
BOUGHTER_CANONICAL_CSV = (
    BOUGHTER_CANONICAL_DIR / "boughter_vh_914.csv"
)  # Alias or future name

# ============================================================================
# Jain (test set - Novo parity benchmark)
# ============================================================================
JAIN_DIR = DATA_TEST_DIR / "jain"
JAIN_RAW_DIR = JAIN_DIR / "raw"
JAIN_PROCESSED_DIR = JAIN_DIR / "processed"
JAIN_FRAGMENTS_DIR = JAIN_DIR / "fragments"
JAIN_CANONICAL_DIR = JAIN_DIR / "canonical"

# Specific files
JAIN_RAW_EXCEL = JAIN_RAW_DIR / "jain_clinical_antibodies_with_private_elisa.xlsx"
JAIN_PRIVATE_ELISA_EXCEL = JAIN_RAW_DIR / "Private_Jain2017_ELISA_indiv.xlsx"
JAIN_SD01_EXCEL = JAIN_RAW_DIR / "jain-pnas.1616408114.sd01.xlsx"
JAIN_SD02_EXCEL = JAIN_RAW_DIR / "jain-pnas.1616408114.sd02.xlsx"
JAIN_SD03_EXCEL = JAIN_RAW_DIR / "jain-pnas.1616408114.sd03.xlsx"

JAIN_FULL_CSV = JAIN_PROCESSED_DIR / "jain_with_private_elisa_FULL.csv"
JAIN_SD03_CSV = JAIN_PROCESSED_DIR / "jain_sd03.csv"
JAIN_ELISA_116_CSV = JAIN_PROCESSED_DIR / "jain_ELISA_ONLY_116.csv"
JAIN_P5E_S2 = JAIN_PROCESSED_DIR / "jain_p5e_s2_preprocessed.csv"
JAIN_86_PARITY_CSV = JAIN_CANONICAL_DIR / "jain_86_novo_parity.csv"
JAIN_VH_ONLY_86_CSV = JAIN_CANONICAL_DIR / "VH_only_jain_86_p5e_s2.csv"

# ============================================================================
# Harvey (test set - nanobodies)
# ============================================================================
HARVEY_DIR = DATA_TEST_DIR / "harvey"
HARVEY_RAW_DIR = HARVEY_DIR / "raw"
HARVEY_PROCESSED_DIR = HARVEY_DIR / "processed"
HARVEY_FRAGMENTS_DIR = HARVEY_DIR / "fragments"

# Specific files
HARVEY_HIGH_CSV = HARVEY_RAW_DIR / "high_polyreactivity_high_throughput.csv"
HARVEY_LOW_CSV = HARVEY_RAW_DIR / "low_polyreactivity_high_throughput.csv"
HARVEY_FULL_CSV = HARVEY_PROCESSED_DIR / "harvey.csv"
HARVEY_VHH_ONLY = HARVEY_FRAGMENTS_DIR / "VHH_only_harvey.csv"

HARVEY_RAW_NS = HARVEY_RAW_DIR / "nanobody_nonspecific.csv"  # Legacy?
HARVEY_RAW_S = HARVEY_RAW_DIR / "nanobody_specific.csv"  # Legacy?
HARVEY_COMBINED = HARVEY_PROCESSED_DIR / "harvey_combined.csv"  # Legacy?

# ============================================================================
# Shehata (test set - PSR assay)
# ============================================================================
SHEHATA_DIR = DATA_TEST_DIR / "shehata"
SHEHATA_RAW_DIR = SHEHATA_DIR / "raw"
SHEHATA_PROCESSED_DIR = SHEHATA_DIR / "processed"
SHEHATA_FRAGMENTS_DIR = SHEHATA_DIR / "fragments"
SHEHATA_CANONICAL_DIR = SHEHATA_DIR / "canonical"

# Specific files
SHEHATA_RAW_EXCEL = SHEHATA_RAW_DIR / "shehata-mmc2.xlsx"
SHEHATA_PROCESSED_CSV = SHEHATA_PROCESSED_DIR / "shehata.csv"
SHEHATA_CANONICAL_CSV = SHEHATA_CANONICAL_DIR / "shehata_398.csv"

# ============================================================================
# Experiments
# ============================================================================
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
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
