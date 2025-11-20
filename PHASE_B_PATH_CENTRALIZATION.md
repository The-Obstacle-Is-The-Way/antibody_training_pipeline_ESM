# Phase B: Path Centralization

**Effort:** 2-3 hours
**Risk:** MEDIUM
**Dependencies:** Phase A complete
**Branch:** `claude/refactor-phase-b`

---

## Overview

Eliminate 50+ hardcoded paths scattered across 17 preprocessing scripts by creating a single source of truth.

**Goal:** Create `preprocessing/paths.py` and migrate all scripts to use centralized path constants.

**Why this matters:**
- Changing directory structure currently breaks 17 scripts
- Hardcoded paths violate DRY principle
- Centralized paths make testing/deployment easier

---

## Fixes Included

| Fix # | Description | Effort | Risk |
|-------|-------------|--------|------|
| #7 | Centralize hardcoded paths | 2-3 hours | MEDIUM |

---

## Task B1: Create preprocessing/paths.py (30 min)

### Deliverable
New file with ~180 lines of path constants for all datasets and experiments.

### Implementation

Create `preprocessing/paths.py`:

```python
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
BOUGHTER_CANONICAL_DIR = BOUGHTER_DIR / "canonical"

# Specific files
BOUGHTER_STAGE1_DNA = BOUGHTER_RAW_DIR / "Boughter_VH_DNA.csv"
BOUGHTER_STAGE2_ANNOTATED = BOUGHTER_PROCESSED_DIR / "stage2_annotated.csv"
BOUGHTER_STAGE3_QC = BOUGHTER_PROCESSED_DIR / "stage3_qc_passed.csv"
BOUGHTER_CANONICAL_CSV = BOUGHTER_CANONICAL_DIR / "boughter_vh_914.csv"

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
JAIN_ELISA_116 = JAIN_PROCESSED_DIR / "jain_ELISA_ONLY_116.csv"
JAIN_P5E_S2 = JAIN_PROCESSED_DIR / "jain_p5e_s2_preprocessed.csv"
JAIN_CANONICAL_CSV = JAIN_CANONICAL_DIR / "jain_86_novo_parity.csv"

# ============================================================================
# Harvey (test set - nanobodies)
# ============================================================================
HARVEY_DIR = DATA_TEST_DIR / "harvey"
HARVEY_RAW_DIR = HARVEY_DIR / "raw"
HARVEY_PROCESSED_DIR = HARVEY_DIR / "processed"
HARVEY_FRAGMENTS_DIR = HARVEY_DIR / "fragments"

# Specific files
HARVEY_RAW_NS = HARVEY_RAW_DIR / "nanobody_nonspecific.csv"
HARVEY_RAW_S = HARVEY_RAW_DIR / "nanobody_specific.csv"
HARVEY_COMBINED = HARVEY_PROCESSED_DIR / "harvey_combined.csv"
HARVEY_VHH_ONLY = HARVEY_FRAGMENTS_DIR / "VHH_only_harvey.csv"

# ============================================================================
# Shehata (test set - PSR assay)
# ============================================================================
SHEHATA_DIR = DATA_TEST_DIR / "shehata"
SHEHATA_RAW_DIR = SHEHATA_DIR / "raw"
SHEHATA_PROCESSED_DIR = SHEHATA_DIR / "processed"
SHEHATA_FRAGMENTS_DIR = SHEHATA_DIR / "fragments"
SHEHATA_CANONICAL_DIR = SHEHATA_DIR / "canonical"

# Specific files
SHEHATA_RAW_EXCEL = SHEHATA_RAW_DIR / "shehata_antibody_data.xlsx"
SHEHATA_PROCESSED_CSV = SHEHATA_PROCESSED_DIR / "shehata_processed.csv"
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
```

### Verification
```bash
# File exists
ls preprocessing/paths.py

# Can be imported
python -c "from preprocessing.paths import JAIN_RAW_DIR; print(JAIN_RAW_DIR)"
```

---

## Task B2: Migrate Boughter Scripts (30 min)

### Files to Update (4 files)
1. `preprocessing/boughter/stage1_dna_translation.py`
2. `preprocessing/boughter/stage2_stage3_annotation_qc.py`
3. `preprocessing/boughter/validate_stage1.py`
4. `preprocessing/boughter/validate_stages2_3.py`

### Pattern

**BEFORE:**
```python
from pathlib import Path

# Hardcoded paths
RAW_DIR = Path("data/train/boughter/raw")
PROCESSED_DIR = Path("data/train/boughter/processed")
CANONICAL_DIR = Path("data/train/boughter/canonical")
```

**AFTER:**
```python
from preprocessing.paths import (
    BOUGHTER_RAW_DIR,
    BOUGHTER_PROCESSED_DIR,
    BOUGHTER_CANONICAL_DIR,
)

# Use imported constants
RAW_DIR = BOUGHTER_RAW_DIR
PROCESSED_DIR = BOUGHTER_PROCESSED_DIR
CANONICAL_DIR = BOUGHTER_CANONICAL_DIR
```

### Verification
```bash
# Scripts still run
uv run python preprocessing/boughter/validate_stage1.py
uv run python preprocessing/boughter/validate_stages2_3.py
```

---

## Task B3: Migrate Jain Scripts (30 min)

### Files to Update (5 files)
1. `preprocessing/jain/step1_convert_excel_to_csv.py`
2. `preprocessing/jain/step2_preprocess_p5e_s2.py`
3. `preprocessing/jain/step3_extract_fragments.py`
4. `preprocessing/jain/validate_conversion.py`
5. `preprocessing/jain/test_novo_parity.py`

### Pattern

**BEFORE:**
```python
from pathlib import Path

RAW_DIR = Path("data/test/jain/raw")
PROCESSED_DIR = Path("data/test/jain/processed")
FRAGMENTS_DIR = Path("data/test/jain/fragments")
```

**AFTER:**
```python
from preprocessing.paths import (
    JAIN_RAW_DIR,
    JAIN_PROCESSED_DIR,
    JAIN_FRAGMENTS_DIR,
    JAIN_CANONICAL_DIR,
)

RAW_DIR = JAIN_RAW_DIR
PROCESSED_DIR = JAIN_PROCESSED_DIR
FRAGMENTS_DIR = JAIN_FRAGMENTS_DIR
```

### Verification
```bash
# Scripts still run
uv run python preprocessing/jain/validate_conversion.py
uv run python preprocessing/jain/test_novo_parity.py --help
```

---

## Task B4: Migrate Harvey Scripts (20 min)

### Files to Update (3 files)
1. `preprocessing/harvey/step1_convert_raw_csvs.py`
2. `preprocessing/harvey/step2_extract_fragments.py`
3. `tests/integration/preprocessing/test_harvey_psr_threshold.py`

### Pattern

**BEFORE:**
```python
RAW_DIR = Path("data/test/harvey/raw")
FRAGMENTS_DIR = Path("data/test/harvey/fragments")
CSV_PATH = "data/test/harvey/fragments/VHH_only_harvey.csv"
```

**AFTER:**
```python
from preprocessing.paths import (
    HARVEY_RAW_DIR,
    HARVEY_FRAGMENTS_DIR,
    HARVEY_VHH_ONLY,
)

RAW_DIR = HARVEY_RAW_DIR
FRAGMENTS_DIR = HARVEY_FRAGMENTS_DIR
CSV_PATH = HARVEY_VHH_ONLY
```

### Verification
```bash
# Scripts still run
uv run python preprocessing/harvey/step1_convert_raw_csvs.py --help
```

---

## Task B5: Migrate Shehata Scripts (20 min)

### Files to Update (3 files)
1. `preprocessing/shehata/step1_convert_excel_to_csv.py`
2. `preprocessing/shehata/step2_extract_fragments.py`
3. `preprocessing/shehata/validate_conversion.py`

### Pattern

**BEFORE:**
```python
RAW_DIR = Path("data/test/shehata/raw")
PROCESSED_DIR = Path("data/test/shehata/processed")
```

**AFTER:**
```python
from preprocessing.paths import (
    SHEHATA_RAW_DIR,
    SHEHATA_PROCESSED_DIR,
    SHEHATA_CANONICAL_DIR,
)

RAW_DIR = SHEHATA_RAW_DIR
PROCESSED_DIR = SHEHATA_PROCESSED_DIR
```

### Verification
```bash
# Scripts still run
uv run python preprocessing/shehata/validate_conversion.py
```

---

## Task B6: Final Verification (30 min)

### Comprehensive Testing

```bash
# 1. Check no hardcoded paths remain (except in paths.py)
grep -rn "data/train\|data/test" preprocessing/ --include="*.py" | grep -v paths.py
# Should return NOTHING

# 2. Spot check: Run one script from each dataset
echo "=== Boughter ==="
uv run python preprocessing/boughter/validate_stage1.py

echo "=== Jain ==="
uv run python preprocessing/jain/validate_conversion.py

echo "=== Harvey ==="
uv run python preprocessing/harvey/step1_convert_raw_csvs.py --help

echo "=== Shehata ==="
uv run python preprocessing/shehata/validate_conversion.py

# 3. Run all tests
uv run pytest

# 4. Quality gates
make all
```

### Success Criteria
- [ ] `preprocessing/paths.py` exists
- [ ] Zero hardcoded "data/" paths in preprocessing scripts (except paths.py)
- [ ] All 17 scripts updated
- [ ] All validation scripts pass
- [ ] All 468 tests pass
- [ ] `make all` passes

---

## Phase Completion Checklist

### All Tasks Complete
- [ ] Task B1: Created preprocessing/paths.py
- [ ] Task B2: Migrated 4 Boughter scripts
- [ ] Task B3: Migrated 5 Jain scripts
- [ ] Task B4: Migrated 3 Harvey scripts
- [ ] Task B5: Migrated 3 Shehata scripts
- [ ] Task B6: Final verification passed

### Quality Gates
- [ ] Run `make all` (format → lint → typecheck → test)
- [ ] All preprocessing scripts run successfully
- [ ] No hardcoded paths found (except paths.py)
- [ ] Security scan: `uv run bandit -r src/ preprocessing/`

### Git Workflow
```bash
# Create branch
git checkout dev
git pull origin dev
git checkout -b claude/refactor-phase-b

# Make changes (complete all 6 tasks above)

# Commit
git add -A
git commit -m "$(cat <<'EOF'
refactor: Phase B - Centralize hardcoded paths

Created preprocessing/paths.py as single source of truth for all data paths.
Eliminated 50+ hardcoded path strings across 17 preprocessing scripts.

**Task B1: Create preprocessing/paths.py**
- Centralized path constants for all 4 datasets
- Included experiment paths (checkpoints, cache, benchmarks)
- Added helper function for dynamic path construction
- ~180 lines of well-documented path definitions

**Tasks B2-B5: Migrate Scripts**
- Boughter (4 scripts): stage1, stage2_3, validate_stage1, validate_stages2_3
- Jain (5 scripts): step1, step2, step3, validate, test_novo_parity
- Harvey (3 scripts): step1, step2, test_psr_threshold
- Shehata (3 scripts): step1, step2, validate

**Pattern Applied:**
BEFORE: Path("data/test/jain/raw")
AFTER: from preprocessing.paths import JAIN_RAW_DIR

**Task B6: Verification**
- All 17 scripts run successfully
- Zero hardcoded paths remain (verified via grep)
- All validation scripts pass

**Quality Gates: ✅ ALL PASSED**
- make all: PASSED
- pytest (468 tests): PASSED
- bandit security scan: PASSED
- All preprocessing scripts: PASSED

**Impact:**
- Changing directory structure now requires editing ONE file
- DRY principle restored (no duplicate path strings)
- Easier testing (can override paths for test fixtures)
- Foundation laid for Phase C (file splitting)

**Files Changed:**
- NEW: preprocessing/paths.py
- MODIFIED: 17 preprocessing scripts

**Next:** Phase C - File Splitting
EOF
)"

# Push and create PR
git push -u origin claude/refactor-phase-b
gh pr create --title "Phase B: Path Centralization - Eliminate 50+ Hardcoded Paths" \
  --body "Completes Phase B of technical debt cleanup. See commit message for details." \
  --base dev
```

---

## Success Metrics

**Before Phase B:**
- Hardcoded paths: 50+ instances across 17 files
- Path sources: Scattered everywhere
- Changing structure: Requires editing 17 files

**After Phase B:**
- Hardcoded paths: 0 (all in paths.py) ✅
- Path sources: Single file (preprocessing/paths.py) ✅
- Changing structure: Edit 1 file ✅

---

**Phase B Complete! Ready for Phase C (File Splitting)**
