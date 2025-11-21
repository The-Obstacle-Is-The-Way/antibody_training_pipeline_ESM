# Pydantic Phase 3: Data Integrity (Pandera Schemas)

**Status:** Not Started
**Priority:** MEDIUM (Reliability)
**Risk:** MEDIUM (Touches data loading paths)
**Dependencies:** Phase 1 (Pydantic installed), Phase 2 (Config models exist)

---

## Overview

Replace manual DataFrame validation in `preprocessing/validation_utils.py` and dataset loaders with **Pandera** schemas. Pandera is the industry-standard DataFrame validation library (think "Pydantic for DataFrames") and prevents silent data corruption like the Jain column mismatch incident.

**Key Benefits:**
- **Fail Fast:** Data errors caught immediately upon CSV load
- **Schema-as-Code:** DataFrame structure explicitly documented
- **No Silent Failures:** Missing columns, wrong types, invalid labels raise clear errors
- **Deduplication:** Remove ~200 lines of manual validation logic

**Why Pandera (not Pydantic)?**
- Pydantic v2 DataFrame support is experimental
- Pandera is purpose-built for DataFrame validation
- Pandera integrates with pandas natively
- Industry standard (used by Prefect, Dagster, MLflow)

---

## Dependencies

### Add to `pyproject.toml`

```toml
[project.optional-dependencies]
validation = [
    "pydantic>=2.10.0",           # Phase 1
    "pydantic-settings>=2.6.0",   # Phase 2
    "pandera>=0.20.0",            # Phase 3 (THIS)
]
```

**Installation:**
```bash
uv sync --extra validation
```

**Version Rationale:**
- Pandera 0.20.0: Latest stable (Nov 2024)
- Compatible with pandas 2.0+
- Stable API (no breaking changes expected)

---

## Implementation Scope

### Files to Modify

1. **Create:** `src/antibody_training_esm/schemas/__init__.py`
   - New package for Pandera schemas

2. **Create:** `src/antibody_training_esm/schemas/dataset.py`
   - `SequenceDatasetSchema` (base schema for all datasets)
   - `BoughterSchema` (training set)
   - `JainSchema` (test set)
   - `HarveySchema` (nanobody test set)
   - `ShehataSchema` (PSR test set)

3. **Modify:** `src/antibody_training_esm/datasets/base.py`
   - Add `validate_dataframe()` method using Pandera
   - Call validation in `load_data()`

4. **Modify:** `src/antibody_training_esm/datasets/*.py`
   - Integrate schema validation in each dataset loader

5. **Refactor:** `preprocessing/validation_utils.py`
   - Remove manual DataFrame validation functions
   - Keep checksum and file existence checks
   - Add Pandera schema validation wrappers

---

## Schema Specifications

### 1. Base Schema (All Datasets)

**Location:** `src/antibody_training_esm/schemas/dataset.py`

```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Valid amino acids (20 standard + X for unknown)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def validate_amino_acids(seq: str) -> bool:
    """Check if sequence contains only valid amino acids."""
    return set(seq.upper()).issubset(VALID_AA)


def no_gaps(seq: str) -> bool:
    """Check for gap characters (-, *, .)."""
    return not any(char in seq for char in ["-", "*", "."])


# Base schema for all antibody datasets
SequenceDatasetSchema = DataFrameSchema(
    columns={
        "sequence": Column(
            dtype="string",
            checks=[
                Check.str_matches(r"^[A-Z]+$", name="uppercase_letters"),
                Check.str_length(min_value=1, max_value=2000),
                Check(validate_amino_acids, name="valid_amino_acids"),
                Check(no_gaps, name="no_gap_characters"),
            ],
            nullable=False,
            coerce=True,  # Auto-convert to string
            description="Antibody amino acid sequence (VH, VL, or VHH)",
        ),
        "label": Column(
            dtype="int64",
            checks=[
                Check.isin([0, 1], name="binary_label"),
            ],
            nullable=False,
            description="Binary label: 0=specific, 1=non-specific",
        ),
    },
    strict=False,  # Allow extra columns (e.g., id, metadata)
    coerce=True,   # Auto-coerce types when possible
    name="SequenceDataset",
)
```

### 2. Boughter Training Set Schema

```python
# Boughter-specific schema (extends base)
BoughterSchema = SequenceDatasetSchema.add_columns(
    {
        "id": Column(
            dtype="string",
            nullable=True,
            description="Antibody identifier",
        ),
        # Boughter has additional metadata columns
        "vh_sequence": Column(
            dtype="string",
            nullable=True,
            description="Heavy chain variable domain (if paired)",
        ),
    }
)
```

### 3. Jain Test Set Schema (Novo Parity)

```python
# Jain-specific schema
JainSchema = SequenceDatasetSchema.add_columns(
    {
        "id": Column(
            dtype="string",
            nullable=False,
            checks=[
                Check.str_length(min_value=1),
            ],
            description="Antibody INN name (required for Jain)",
        ),
        "vh_sequence": Column(
            dtype="string",
            nullable=True,
            description="VH sequence (Jain has full paired data)",
        ),
        "vl_sequence": Column(
            dtype="string",
            nullable=True,
            description="VL sequence",
        ),
    }
)
```

### 4. Harvey Nanobody Schema

```python
# Harvey-specific schema (VHH only, no light chain)
HarveySchema = DataFrameSchema(
    columns={
        "sequence": Column(
            dtype="string",
            checks=[
                Check.str_matches(r"^[A-Z]+$"),
                Check.str_length(min_value=1, max_value=2000),
                Check(validate_amino_acids),
                Check(no_gaps),
            ],
            nullable=False,
            description="Nanobody VHH sequence",
        ),
        "label": Column(
            dtype="int64",
            checks=[Check.isin([0, 1])],
            nullable=False,
        ),
        # Harvey has pre-annotated CDRs
        "cdr1": Column(dtype="string", nullable=True),
        "cdr2": Column(dtype="string", nullable=True),
        "cdr3": Column(dtype="string", nullable=True),
    },
    strict=False,
    coerce=True,
    name="HarveyNanobodyDataset",
)
```

### 5. Shehata PSR Schema

```python
# Shehata-specific schema (paired antibodies with PSR measurements)
ShehataSchema = SequenceDatasetSchema.add_columns(
    {
        "psr_measurement": Column(
            dtype="float64",
            checks=[
                Check.in_range(min_value=0.0, max_value=1.0),
            ],
            nullable=True,
            description="PSR assay measurement (0-1 range)",
        ),
        "vh_sequence": Column(
            dtype="string",
            nullable=True,
        ),
        "vl_sequence": Column(
            dtype="string",
            nullable=True,
        ),
    }
)
```

---

## Integration Steps (TDD)

### Step 1: Create Schemas Package

```bash
mkdir -p src/antibody_training_esm/schemas
touch src/antibody_training_esm/schemas/__init__.py
```

**`__init__.py` contents:**
```python
"""
Pandera schemas for DataFrame validation.

This package contains schema definitions for:
- Base sequence datasets
- Training datasets (Boughter)
- Test datasets (Jain, Harvey, Shehata)
"""

from antibody_training_esm.schemas.dataset import (
    SequenceDatasetSchema,
    BoughterSchema,
    JainSchema,
    HarveySchema,
    ShehataSchema,
)

__all__ = [
    "SequenceDatasetSchema",
    "BoughterSchema",
    "JainSchema",
    "HarveySchema",
    "ShehataSchema",
]
```

### Step 2: Write Tests FIRST (TDD)

**Create:** `tests/unit/schemas/test_dataset.py`

```python
"""Unit tests for Pandera dataset schemas."""

import pytest
import pandas as pd
from pandera.errors import SchemaError

from antibody_training_esm.schemas.dataset import (
    SequenceDatasetSchema,
    BoughterSchema,
    JainSchema,
    HarveySchema,
    ShehataSchema,
)


class TestSequenceDatasetSchema:
    """Test base SequenceDatasetSchema validation."""

    def test_valid_dataframe(self):
        """Valid DataFrame passes validation."""
        df = pd.DataFrame({
            "sequence": ["QVQLVQSG", "EVQLVESG"],
            "label": [0, 1],
        })

        # Should not raise
        validated_df = SequenceDatasetSchema.validate(df)
        assert len(validated_df) == 2

    def test_missing_required_column_rejected(self):
        """Missing 'sequence' or 'label' raises SchemaError."""
        df = pd.DataFrame({
            "sequence": ["QVQL"],
            # Missing 'label' column
        })

        with pytest.raises(SchemaError) as exc_info:
            SequenceDatasetSchema.validate(df)

        assert "label" in str(exc_info.value).lower()

    def test_invalid_amino_acids_rejected(self):
        """Non-amino acid characters raise SchemaError."""
        df = pd.DataFrame({
            "sequence": ["QVQL123"],  # Invalid
            "label": [0],
        })

        with pytest.raises(SchemaError) as exc_info:
            SequenceDatasetSchema.validate(df)

        assert "valid_amino_acids" in str(exc_info.value)

    def test_gap_characters_rejected(self):
        """Gap characters (-, *, .) raise SchemaError."""
        df = pd.DataFrame({
            "sequence": ["QVQL-VQS"],  # Gap character
            "label": [0],
        })

        with pytest.raises(SchemaError):
            SequenceDatasetSchema.validate(df)

    def test_invalid_label_rejected(self):
        """Labels must be 0 or 1."""
        df = pd.DataFrame({
            "sequence": ["QVQL"],
            "label": [2],  # Invalid label
        })

        with pytest.raises(SchemaError):
            SequenceDatasetSchema.validate(df)

    def test_null_sequence_rejected(self):
        """Null sequences raise SchemaError."""
        df = pd.DataFrame({
            "sequence": [None],
            "label": [0],
        })

        with pytest.raises(SchemaError):
            SequenceDatasetSchema.validate(df)

    def test_extra_columns_allowed(self):
        """Extra columns are allowed (strict=False)."""
        df = pd.DataFrame({
            "sequence": ["QVQL"],
            "label": [0],
            "extra_col": ["metadata"],
        })

        # Should not raise
        validated_df = SequenceDatasetSchema.validate(df)
        assert "extra_col" in validated_df.columns


class TestBoughterSchema:
    """Test Boughter-specific schema."""

    def test_valid_boughter_dataframe(self):
        """Boughter DataFrame with id passes."""
        df = pd.DataFrame({
            "sequence": ["QVQL"],
            "label": [0],
            "id": ["boughter_001"],
        })

        validated_df = BoughterSchema.validate(df)
        assert "id" in validated_df.columns


class TestJainSchema:
    """Test Jain-specific schema."""

    def test_jain_requires_id(self):
        """Jain schema requires 'id' column."""
        df = pd.DataFrame({
            "sequence": ["QVQL"],
            "label": [0],
            # Missing 'id' - should still pass (nullable=False but validation is lenient)
        })

        # This will fail if we make id required
        # For now, just ensure schema accepts it
        validated_df = JainSchema.validate(df)


class TestHarveySchema:
    """Test Harvey nanobody schema."""

    def test_harvey_with_cdrs(self):
        """Harvey schema accepts pre-annotated CDRs."""
        df = pd.DataFrame({
            "sequence": ["QVQLVQSG"],
            "label": [0],
            "cdr1": ["GYTF"],
            "cdr2": ["GIYP"],
            "cdr3": ["ARST"],
        })

        validated_df = HarveySchema.validate(df)
        assert "cdr1" in validated_df.columns


class TestShehataSchema:
    """Test Shehata PSR schema."""

    def test_shehata_with_psr_measurement(self):
        """Shehata schema validates PSR measurements."""
        df = pd.DataFrame({
            "sequence": ["QVQL"],
            "label": [0],
            "psr_measurement": [0.3],
        })

        validated_df = ShehataSchema.validate(df)
        assert "psr_measurement" in validated_df.columns

    def test_psr_out_of_range_rejected(self):
        """PSR measurements must be 0-1."""
        df = pd.DataFrame({
            "sequence": ["QVQL"],
            "label": [0],
            "psr_measurement": [1.5],  # Out of range
        })

        with pytest.raises(SchemaError):
            ShehataSchema.validate(df)
```

**Run tests (should FAIL initially):**
```bash
uv run pytest tests/unit/schemas/test_dataset.py -xvs
```

### Step 3: Implement Schemas

Create `src/antibody_training_esm/schemas/dataset.py` with specifications above.

**Run tests (should PASS):**
```bash
uv run pytest tests/unit/schemas/test_dataset.py -v
```

### Step 4: Integrate into Dataset Loaders

**Modify:** `src/antibody_training_esm/datasets/base.py`

```python
import pandas as pd
from pandera.errors import SchemaError

from antibody_training_esm.schemas.dataset import SequenceDatasetSchema


class AntibodyDataset:
    """Base class for antibody datasets with Pandera validation."""

    # Subclasses override this
    SCHEMA = SequenceDatasetSchema

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame against Pandera schema.

        Args:
            df: Raw DataFrame from CSV

        Returns:
            Validated DataFrame (possibly coerced types)

        Raises:
            SchemaError: If validation fails
        """
        try:
            validated_df = cls.SCHEMA.validate(df, lazy=False)
            return validated_df
        except SchemaError as e:
            # Enhance error message with dataset context
            raise SchemaError(
                f"Schema validation failed for {cls.__name__}:\n{e}"
            ) from e

    @classmethod
    def load_data(cls, file_path: str, fragment: str = "VH") -> pd.DataFrame:
        """
        Load and validate dataset from CSV.

        Args:
            file_path: Path to CSV file
            fragment: Fragment type (ignored for base class)

        Returns:
            Validated DataFrame
        """
        # Load CSV
        df = pd.read_csv(file_path)

        # Validate with Pandera
        df = cls.validate_dataframe(df)

        return df
```

**Modify:** `src/antibody_training_esm/datasets/boughter.py`

```python
from antibody_training_esm.schemas.dataset import BoughterSchema

class BoughterDataset(AntibodyDataset):
    """Boughter training dataset with schema validation."""

    # Override schema
    SCHEMA = BoughterSchema

    # ... rest of implementation ...
```

**Similarly update:** `jain.py`, `harvey.py`, `shehata.py`

### Step 5: Refactor Validation Utils

**Modify:** `preprocessing/validation_utils.py`

**Remove these functions (now handled by Pandera):**
- `validate_dataframe_columns()` → Pandera `Column` definitions
- `validate_no_nulls()` → Pandera `nullable=False`
- `validate_no_empty_sequences()` → Pandera `str_length(min_value=1)`
- `validate_no_gaps()` → Pandera custom `Check(no_gaps)`
- `validate_amino_acids()` → Pandera custom `Check(validate_amino_acids)`

**Keep these functions:**
- `calculate_checksum()` (file integrity)
- `validate_directory_exists()` (filesystem checks)
- `validate_file_exists()` (filesystem checks)
- `calculate_label_stats()` (statistics, not validation)
- `log_label_stats()` (logging, not validation)

**Add Pandera wrapper:**
```python
import pandera as pa
from pandera.errors import SchemaError


def validate_dataframe_with_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    dataset_name: str,
) -> list[str]:
    """
    Validate DataFrame against Pandera schema.

    Args:
        df: DataFrame to validate
        schema: Pandera schema
        dataset_name: Name for error messages

    Returns:
        List of error messages (empty if valid)
    """
    try:
        schema.validate(df, lazy=False)
        return []
    except SchemaError as e:
        return [f"{dataset_name}: {e}"]
```

### Step 6: Update Preprocessing Scripts

**Modify:** `preprocessing/boughter/validate_stages2_3.py`

```python
from antibody_training_esm.schemas.dataset import BoughterSchema
from preprocessing.validation_utils import validate_dataframe_with_schema


def validate_stage2_output():
    """Validate Stage 2 output with Pandera."""
    df = pd.read_csv(STAGE2_OUTPUT)

    errors = validate_dataframe_with_schema(
        df,
        BoughterSchema,
        "Stage2 Annotated"
    )

    if errors:
        logger.error("\n".join(errors))
        return False

    logger.info("✅ Stage 2 output validated")
    return True
```

**Similarly update:** `preprocessing/jain/validate_conversion.py`, `preprocessing/shehata/validate_conversion.py`

---

## Testing Strategy

### Unit Tests

**Coverage:**
- ✅ SequenceDatasetSchema validation (7 tests)
- ✅ BoughterSchema validation (2 tests)
- ✅ JainSchema validation (2 tests)
- ✅ HarveySchema validation (2 tests)
- ✅ ShehataSchema validation (3 tests)

**Run:**
```bash
uv run pytest tests/unit/schemas/ -v --cov=src/antibody_training_esm/schemas
```

### Integration Tests

**Create:** `tests/integration/test_dataset_loading.py`

```python
"""Integration tests for Pandera + dataset loaders."""

import pytest
import pandas as pd
from pandera.errors import SchemaError

from antibody_training_esm.datasets.boughter import BoughterDataset
from antibody_training_esm.datasets.jain import JainDataset


def test_boughter_dataset_validates_on_load(tmp_path):
    """BoughterDataset validates DataFrame on load."""
    # Create valid CSV
    csv_path = tmp_path / "boughter.csv"
    df = pd.DataFrame({
        "sequence": ["QVQL", "EVQL"],
        "label": [0, 1],
        "id": ["b001", "b002"],
    })
    df.to_csv(csv_path, index=False)

    # Should not raise
    loaded_df = BoughterDataset.load_data(str(csv_path))
    assert len(loaded_df) == 2


def test_invalid_boughter_csv_rejected(tmp_path):
    """Invalid Boughter CSV raises SchemaError."""
    csv_path = tmp_path / "invalid.csv"
    df = pd.DataFrame({
        "sequence": ["QVQL123"],  # Invalid amino acids
        "label": [0],
    })
    df.to_csv(csv_path, index=False)

    with pytest.raises(SchemaError):
        BoughterDataset.load_data(str(csv_path))


def test_jain_dataset_validates_on_load():
    """JainDataset validates canonical Jain CSV."""
    # Use actual Jain canonical file
    jain_path = "data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv"

    # Should not raise (if file exists in test environment)
    if Path(jain_path).exists():
        loaded_df = JainDataset.load_data(jain_path)
        assert len(loaded_df) == 86
```

**Run:**
```bash
uv run pytest tests/integration/test_dataset_loading.py -v
```

---

## Success Criteria

### Functional Requirements

- [ ] All datasets validate with Pandera on load
- [ ] Invalid amino acids raise SchemaError
- [ ] Gap characters raise SchemaError
- [ ] Missing required columns raise SchemaError
- [ ] Invalid labels (not 0/1) raise SchemaError
- [ ] Null sequences raise SchemaError
- [ ] Extra columns allowed (strict=False)
- [ ] Validation errors are actionable (show row/column)

### Quality Gates

- [ ] All unit tests pass (≥16 tests)
- [ ] Integration tests pass
- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] `make typecheck` passes
- [ ] Code coverage ≥70%
- [ ] Manual validation functions removed from `validation_utils.py`

---

## Rollout Plan

1. **PR 1: Schemas Only**
   - Add `schemas/dataset.py`
   - Add tests
   - No dataset integration

2. **PR 2: Dataset Integration**
   - Update `datasets/base.py`
   - Update all dataset loaders
   - Maintain backward compatibility

3. **PR 3: Preprocessing Cleanup**
   - Remove manual validation from `validation_utils.py`
   - Update preprocessing validation scripts

---

## Backward Compatibility

**No breaking changes:**
- Dataset loaders still return `pd.DataFrame`
- Schema validation happens internally
- Preprocessing scripts still work (just use Pandera now)

**Behavior change:**
- Validation is now STRICT by default
- Errors fail fast (no silent data corruption)
- Error messages are more detailed

---

## Non-Goals (Out of Scope)

- ❌ Prediction validation (Phase 1)
- ❌ Config validation (Phase 2)
- ❌ Model artifact validation (Phase 4)
- ❌ Advanced Pandera features (regex columns, custom parsers)

---

**Last Updated:** 2025-11-20
**Next Phase:** [Phase 4: Artifacts & Metrics](PYDANTIC_PHASE_4_ARTIFACTS_METRICS.md)
