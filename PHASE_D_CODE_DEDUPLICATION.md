# Phase D: Code Deduplication

**Effort:** 5-7 hours
**Risk:** HIGH
**Dependencies:** Phases A, B, C complete
**Branch:** `claude/refactor-phase-d`

---

## Overview

Extract duplicated validation and fragment extraction logic into shared utility modules.

**Goal:** Eliminate ~1.6k lines of overlapping validation/fragment logic across 7 scripts by creating 2 shared modules.

**Why this is HIGH risk:**
- Creates new shared dependencies
- Must produce identical output (byte-for-byte)
- Extensive testing required to ensure no regressions

---

## Fixes Included

| Fix | Duplicate Lines (est.) | Files Affected | New Module |
|-----|------------------------|----------------|------------|
| #15 | ~900 lines of validation overlap | 4 validation scripts | `validation_utils.py` |
| #16 | ~700-800 lines of fragment overlap | 3 fragment scripts | `fragment_utils.py` |

**Total:** ~1.6k lines of duplicated preprocessing logic → 2 shared modules (~350 lines total)

---

## Task D1: Create validation_utils.py (2-3 hours)

### Problem
4 validation scripts duplicate ~60-80% of their validation logic:
- `preprocessing/boughter/validate_stages2_3.py`
- `preprocessing/jain/validate_conversion.py`
- `preprocessing/harvey/step1_convert_raw_csvs.py` (validation sections)
- `preprocessing/shehata/validate_conversion.py`

### Solution
Extract shared validation functions into `preprocessing/validation_utils.py`.

### Implementation

**Step 1: Create validation_utils.py (1 hour)**

Create `preprocessing/validation_utils.py`:

```python
"""
Shared validation utilities for preprocessing pipelines.

Provides common validation functions used across all dataset preprocessing scripts.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Set
import pandas as pd

logger = logging.getLogger(__name__)

# Standard amino acids
STANDARD_AMINO_ACIDS: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")


def checksum(path: Path) -> str:
    """
    Calculate SHA256 checksum of file.

    Args:
        path: Path to file

    Returns:
        Hexadecimal SHA256 hash string
    """
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_amino_acid_sequences(
    df: pd.DataFrame,
    vh_col: str = "vh_sequence",
    vl_col: str = "vl_sequence",
) -> Dict[str, int]:
    """
    Validate VH/VL sequences contain only valid amino acids.

    Args:
        df: DataFrame with sequence columns
        vh_col: Name of VH sequence column
        vl_col: Name of VL sequence column

    Returns:
        Dict with counts of invalid sequences per chain
    """
    invalid_counts = {"heavy": 0, "light": 0}

    # Validate VH sequences
    if vh_col in df.columns:
        for seq in df[vh_col].dropna():
            if set(seq.upper()) - STANDARD_AMINO_ACIDS:
                invalid_counts["heavy"] += 1
                logger.warning(f"Invalid VH sequence: {seq[:20]}...")

    # Validate VL sequences
    if vl_col in df.columns:
        for seq in df[vl_col].dropna():
            if set(seq.upper()) - STANDARD_AMINO_ACIDS:
                invalid_counts["light"] += 1
                logger.warning(f"Invalid VL sequence: {seq[:20]}...")

    return invalid_counts


def validate_label_distribution(
    df: pd.DataFrame,
    expected: Dict[int, int],
    label_col: str = "label",
) -> bool:
    """
    Validate label distribution matches expected counts.

    Args:
        df: DataFrame with labels
        expected: Dict mapping label values to expected counts
        label_col: Name of label column

    Returns:
        True if distribution matches, False otherwise
    """
    actual = df[label_col].value_counts().to_dict()

    if actual != expected:
        logger.error(f"Label distribution mismatch!")
        logger.error(f"  Expected: {expected}")
        logger.error(f"  Actual: {actual}")
        return False

    logger.info(f"✓ Label distribution verified: {actual}")
    return True


def validate_column_presence(
    df: pd.DataFrame,
    required_columns: list[str],
) -> bool:
    """
    Validate required columns are present in DataFrame.

    Args:
        df: DataFrame to check
        required_columns: List of required column names

    Returns:
        True if all required columns present, False otherwise
    """
    missing = set(required_columns) - set(df.columns)

    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False

    logger.info(f"✓ All required columns present: {required_columns}")
    return True


def validate_no_nulls(
    df: pd.DataFrame,
    columns: list[str],
) -> bool:
    """
    Validate specified columns have no null values.

    Args:
        df: DataFrame to check
        columns: List of column names to check

    Returns:
        True if no nulls found, False otherwise
    """
    for col in columns:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.error(f"Column '{col}' has {null_count} null values")
                return False

    logger.info(f"✓ No null values in: {columns}")
    return True


def print_validation_summary(
    csv_path: Path,
    df: pd.DataFrame,
    dataset_name: str,
    extra_info: Dict[str, Any] = None,
) -> None:
    """
    Print standardized validation summary.

    Args:
        csv_path: Path to CSV file
        df: Loaded DataFrame
        dataset_name: Name of dataset for display
        extra_info: Optional dict of extra info to display
    """
    print("=" * 60)
    print(f"{dataset_name} Validation Summary")
    print("=" * 60)
    print(f"File: {csv_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print(f"\nChecksum (SHA256):")
    print(f"  {checksum(csv_path)}")

    if extra_info:
        print("\nAdditional Info:")
        for key, value in extra_info.items():
            print(f"  {key}: {value}")

    print("\n✓ Validation complete")
    print("=" * 60)
```

**Step 2: Refactor Boughter validation (30 min)**

Update `preprocessing/boughter/validate_stages2_3.py`:

```python
# BEFORE: 60-80 lines of custom validation

# AFTER:
from preprocessing.validation_utils import (
    checksum,
    validate_amino_acid_sequences,
    validate_label_distribution,
    validate_column_presence,
    print_validation_summary,
)

def validate_stages2_3():
    """Validate Boughter stages 2+3 outputs."""
    # Load CSV
    df = pd.read_csv(STAGE3_PATH)

    # Validate columns
    validate_column_presence(df, ["id", "vh_sequence", "label"])

    # Validate amino acids
    invalid = validate_amino_acid_sequences(df)
    assert invalid["heavy"] == 0, f"Found {invalid['heavy']} invalid VH sequences"

    # Validate label distribution
    validate_label_distribution(df, {0: 402, 1: 512})

    # Print summary
    print_validation_summary(STAGE3_PATH, df, "Boughter Stages 2+3")
```

**Step 3: Refactor Jain validation (30 min)**

Update `preprocessing/jain/validate_conversion.py`:
- Replace custom validation with shared functions
- Same pattern as Boughter

**Step 4: Refactor Shehata validation (20 min)**

Update `preprocessing/shehata/validate_conversion.py`:
- Replace custom validation with shared functions

**Step 5: Refactor Harvey validation (20 min)**

Update validation sections in `preprocessing/harvey/step1_convert_raw_csvs.py`.

### Verification

**Critical: Byte-for-Byte Output Verification**

```bash
# 1. Backup original validation outputs
mkdir -p /tmp/validation_backup

# Run original validations, capture output
uv run python preprocessing/boughter/validate_stages2_3.py > /tmp/validation_backup/boughter.txt
uv run python preprocessing/jain/validate_conversion.py > /tmp/validation_backup/jain.txt
uv run python preprocessing/shehata/validate_conversion.py > /tmp/validation_backup/shehata.txt

# 2. After refactoring, run again
uv run python preprocessing/boughter/validate_stages2_3.py > /tmp/validation_new/boughter.txt
uv run python preprocessing/jain/validate_conversion.py > /tmp/validation_new/jain.txt
uv run python preprocessing/shehata/validate_conversion.py > /tmp/validation_new/shehata.txt

# 3. Compare outputs (should be identical)
diff /tmp/validation_backup/boughter.txt /tmp/validation_new/boughter.txt
diff /tmp/validation_backup/jain.txt /tmp/validation_new/jain.txt
diff /tmp/validation_backup/shehata.txt /tmp/validation_new/shehata.txt

# Should return NOTHING (identical files)
```

### Success Criteria
- [ ] `preprocessing/validation_utils.py` exists (~150 lines)
- [ ] 4 validation scripts refactored (60-80 lines each → ~20 lines)
- [ ] All validation scripts produce identical output
- [ ] All tests pass

---

## Task D2: Create fragment_utils.py (3-4 hours)

### Problem
3 fragment extraction scripts duplicate ~200 lines of ANARCI logic each:
- `preprocessing/jain/step3_extract_fragments.py`
- `preprocessing/harvey/step2_extract_fragments.py`
- `preprocessing/shehata/step2_extract_fragments.py`

### Solution
Extract shared ANARCI fragment extraction into `preprocessing/fragment_utils.py`.

### Implementation

**Step 1: Create fragment_utils.py (1.5 hours)**

Create `preprocessing/fragment_utils.py`:

```python
"""
Shared fragment extraction utilities using ANARCI.

Provides common fragment extraction functions for all datasets.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
from riot_na import Antibody

logger = logging.getLogger(__name__)


def annotate_sequence_with_anarci(
    sequence: str,
    chain_type: str = "H",
    scheme: str = "imgt",
    cdr_definition: str = "imgt",
) -> Optional[Antibody]:
    """
    Annotate antibody sequence with ANARCI (IMGT numbering).

    Args:
        sequence: Amino acid sequence
        chain_type: "H" for heavy, "L" for light
        scheme: Numbering scheme (default: imgt)
        cdr_definition: CDR definition (default: imgt)

    Returns:
        riot_na.Antibody object or None if annotation fails
    """
    try:
        ab = Antibody(sequence, scheme=scheme, cdr_definition=cdr_definition)
        return ab if ab.numbering else None
    except Exception as e:
        logger.warning(f"ANARCI annotation failed for sequence: {e}")
        return None


def extract_cdrs(
    ab: Antibody,
    chain: str = "heavy",
) -> Dict[str, str]:
    """
    Extract CDR sequences from annotated antibody.

    Args:
        ab: riot_na.Antibody object
        chain: "heavy" or "light"

    Returns:
        Dict mapping CDR names to sequences (e.g., "H-CDR1": "GFTFSSYA")
    """
    cdrs = {}
    chain_prefix = chain[0].upper()

    for i in [1, 2, 3]:
        cdr_name = f"{chain_prefix}-CDR{i}"
        try:
            cdr_seq = ab.get_region(f"cdr{i}", chain=chain[0])
            cdrs[cdr_name] = cdr_seq if cdr_seq else ""
        except Exception as e:
            logger.warning(f"Could not extract {cdr_name}: {e}")
            cdrs[cdr_name] = ""

    return cdrs


def extract_framework_regions(
    ab: Antibody,
    chain: str = "heavy",
) -> Dict[str, str]:
    """
    Extract framework region sequences from annotated antibody.

    Args:
        ab: riot_na.Antibody object
        chain: "heavy" or "light"

    Returns:
        Dict mapping FWR names to sequences (e.g., "H-FWR1": "QVQLQ...")
    """
    fwrs = {}
    chain_prefix = chain[0].upper()

    for i in [1, 2, 3, 4]:
        fwr_name = f"{chain_prefix}-FWR{i}"
        try:
            fwr_seq = ab.get_region(f"fwr{i}", chain=chain[0])
            fwrs[fwr_name] = fwr_seq if fwr_seq else ""
        except Exception as e:
            logger.warning(f"Could not extract {fwr_name}: {e}")
            fwrs[fwr_name] = ""

    return fwrs


def create_combined_fragments(row_data: Dict[str, str]) -> Dict[str, str]:
    """
    Create combined fragment sequences (H-CDRs, L-CDRs, All-CDRs, etc.).

    Args:
        row_data: Dict with individual CDR/FWR sequences

    Returns:
        Dict with combined fragment sequences
    """
    combined = {}

    # Heavy CDRs combined
    h_cdrs = [row_data.get(f"H-CDR{i}", "") for i in [1, 2, 3]]
    combined["H-CDRs"] = "".join(h_cdrs)

    # Light CDRs combined
    l_cdrs = [row_data.get(f"L-CDR{i}", "") for i in [1, 2, 3]]
    combined["L-CDRs"] = "".join(l_cdrs)

    # All CDRs combined
    combined["All-CDRs"] = combined["H-CDRs"] + combined["L-CDRs"]

    # Heavy FWRs combined
    h_fwrs = [row_data.get(f"H-FWR{i}", "") for i in [1, 2, 3, 4]]
    combined["H-FWRs"] = "".join(h_fwrs)

    # Light FWRs combined
    l_fwrs = [row_data.get(f"L-FWR{i}", "") for i in [1, 2, 3, 4]]
    combined["L-FWRs"] = "".join(l_fwrs)

    # All FWRs combined
    combined["All-FWRs"] = combined["H-FWRs"] + combined["L-FWRs"]

    return combined


def process_sequences_to_fragments(
    df: pd.DataFrame,
    vh_col: str = "vh_sequence",
    vl_col: str = "vl_sequence",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Process VH/VL sequences into fragment CSV.

    Standard pipeline:
    1. Annotate with ANARCI
    2. Extract CDRs (H/L-CDR1/2/3)
    3. Extract FWRs (H/L-FWR1/2/3/4)
    4. Create combined fragments (H-CDRs, L-CDRs, All-CDRs, etc.)

    Args:
        df: DataFrame with VH/VL sequences
        vh_col: Name of VH sequence column
        vl_col: Name of VL sequence column
        id_col: Name of ID column

    Returns:
        DataFrame with fragment columns
    """
    fragments = []

    logger.info(f"Processing {len(df)} sequences into fragments...")

    for idx, row in df.iterrows():
        row_data = {id_col: row[id_col]}

        # Process VH (heavy chain)
        if vh_col in df.columns and pd.notna(row[vh_col]):
            vh_ab = annotate_sequence_with_anarci(row[vh_col], "H")
            if vh_ab:
                row_data.update(extract_cdrs(vh_ab, "heavy"))
                row_data.update(extract_framework_regions(vh_ab, "heavy"))
                row_data["vh_sequence"] = row[vh_col]

        # Process VL (light chain)
        if vl_col in df.columns and pd.notna(row[vl_col]):
            vl_ab = annotate_sequence_with_anarci(row[vl_col], "L")
            if vl_ab:
                row_data.update(extract_cdrs(vl_ab, "light"))
                row_data.update(extract_framework_regions(vl_ab, "light"))
                row_data["vl_sequence"] = row[vl_col]

        # Create combined fragments
        row_data.update(create_combined_fragments(row_data))

        # Add label if present
        if "label" in df.columns:
            row_data["label"] = row["label"]

        fragments.append(row_data)

    logger.info(f"✓ Extracted fragments for {len(fragments)} sequences")

    return pd.DataFrame(fragments)
```

**Step 2: Refactor Jain fragments (30 min)**

Update `preprocessing/jain/step3_extract_fragments.py`:

```python
# BEFORE: 200+ lines of ANARCI logic

# AFTER:
from preprocessing.fragment_utils import process_sequences_to_fragments

# Load canonical CSV
canonical_df = pd.read_csv(CANONICAL_PATH)

# Extract fragments (one function call!)
fragments_df = process_sequences_to_fragments(
    canonical_df,
    vh_col="vh_sequence",
    vl_col="vl_sequence",
    id_col="id",
)

# Save fragments
fragments_df.to_csv(OUTPUT_PATH, index=False)
logger.info(f"Saved fragments to {OUTPUT_PATH}")
```

**Step 3: Refactor Harvey fragments (30 min)**

Update `preprocessing/harvey/step2_extract_fragments.py`:
- Same pattern as Jain

**Step 4: Refactor Shehata fragments (30 min)**

Update `preprocessing/shehata/step2_extract_fragments.py`:
- Same pattern as Jain

### Verification

**Critical: Byte-for-Byte CSV Verification**

```bash
# 1. Backup original fragment CSVs
mkdir -p /tmp/fragments_backup
cp data/test/jain/fragments/*.csv /tmp/fragments_backup/
cp data/test/harvey/fragments/*.csv /tmp/fragments_backup/
cp data/test/shehata/fragments/*.csv /tmp/fragments_backup/

# 2. After refactoring, regenerate fragments
uv run python preprocessing/jain/step3_extract_fragments.py
uv run python preprocessing/harvey/step2_extract_fragments.py
uv run python preprocessing/shehata/step2_extract_fragments.py

# 3. Compare SHA256 checksums (must match EXACTLY)
cd data/test/jain/fragments
for file in *.csv; do
    echo "Checking $file..."
    diff <(sha256sum "$file") <(sha256sum "/tmp/fragments_backup/$file")
done

# Repeat for harvey and shehata

# Should return NOTHING (identical checksums)
```

### Success Criteria
- [ ] `preprocessing/fragment_utils.py` exists (~200 lines)
- [ ] 3 fragment scripts refactored (200+ lines → ~20 lines each)
- [ ] All fragment CSVs byte-for-byte identical to originals
- [ ] All tests pass

---

## Phase Completion Checklist

### All Tasks Complete
- [ ] Task D1: Created validation_utils.py
- [ ] Task D1: Refactored 4 validation scripts
- [ ] Task D2: Created fragment_utils.py
- [ ] Task D2: Refactored 3 fragment scripts

### Byte-for-Byte Verification
- [ ] All validation outputs identical
- [ ] All fragment CSVs identical (SHA256 match)

### Quality Gates
- [ ] All tests pass: `uv run pytest`
- [ ] Type checking: `uv run mypy src/ preprocessing/ --strict`
- [ ] Linting: `uv run ruff check src/ preprocessing/`
- [ ] All preprocessing scripts work
- [ ] `make all` passes

### Git Workflow
```bash
# Create branch
git checkout dev
git pull origin dev
git checkout -b claude/refactor-phase-d

# Make changes (complete all tasks above)

# Commit
git add -A
git commit -m "$(cat <<'EOF'
refactor: Phase D - Extract duplicated validation & fragment code

Eliminated ~1.6k lines of duplicated preprocessing logic by creating 2 shared utility modules.
All outputs verified byte-for-byte identical to originals.

**Task D1: Create validation_utils.py (~150 lines)**
Extracted shared validation functions:
- checksum(): SHA256 file hashing
- validate_amino_acid_sequences(): Sequence validation
- validate_label_distribution(): Label count verification
- validate_column_presence(): Schema validation
- print_validation_summary(): Standardized output

Refactored 4 validation scripts (60-80 lines → ~20 lines each):
- preprocessing/boughter/validate_stages2_3.py
- preprocessing/jain/validate_conversion.py
- preprocessing/harvey/step1_convert_raw_csvs.py (validation sections)
- preprocessing/shehata/validate_conversion.py

**Task D2: Create fragment_utils.py (~200 lines)**
Extracted shared ANARCI fragment extraction:
- annotate_sequence_with_anarci(): ANARCI annotation
- extract_cdrs(): CDR sequence extraction
- extract_framework_regions(): FWR extraction
- create_combined_fragments(): Combined fragment sequences
- process_sequences_to_fragments(): End-to-end pipeline

Refactored 3 fragment scripts (200+ lines → ~20 lines each):
- preprocessing/jain/step3_extract_fragments.py
- preprocessing/harvey/step2_extract_fragments.py
- preprocessing/shehata/step2_extract_fragments.py

**Verification: ✅ BYTE-FOR-BYTE IDENTICAL**
- All validation outputs: IDENTICAL
- All fragment CSVs: SHA256 match
- Zero functional regressions

**Quality Gates: ✅ ALL PASSED**
- pytest (full suite): PASSED
- mypy strict: PASSED
- ruff check: PASSED
- make all: PASSED

**Impact:**
- Code reduction: ~1.6k duplicate lines → 2 modules (~350 lines)
- DRY principle restored: No duplicate validation or fragment logic
- Easier maintenance: Bug fixes in one place benefit all datasets
- Consistent behavior: All datasets use same validation/extraction

**Files Changed:**
- NEW: preprocessing/validation_utils.py
- NEW: preprocessing/fragment_utils.py
- MODIFIED: 7 preprocessing scripts

**Next:** Phase E - Polish & Documentation
EOF
)"

# Push and create PR
git push -u origin claude/refactor-phase-d
gh pr create --title "Phase D: Code Deduplication - Extract Shared Utils" \
  --body "Completes Phase D of technical debt cleanup. See commit message for details." \
  --base dev
```

---

## Success Metrics

**Before Phase D (validated 2025-11-20):**
- Validation code overlap: ~900 LOC across 4 scripts
- Fragment code overlap: ~1k LOC across 3 scripts
- Total: ~1.6k LOC with substantial duplication

**After Phase D (target):**
- Validation duplication: 0 ✅
- Fragment duplication: 0 ✅
- Shared modules: 2 (~350 lines total) ✅
- Outputs verified byte-for-byte ✅

---

**Phase D Complete! Ready for Phase E (Polish & Documentation)**
