"""Unit tests for Pandera dataset schemas."""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from antibody_training_esm.schemas.dataset import (
    get_boughter_schema,
    get_harvey_schema,
    get_jain_schema,
    get_sequence_dataset_schema,
    get_shehata_schema,
)


class TestSequenceDatasetSchema:
    """Test base SequenceDatasetSchema validation."""

    def test_local_schema(self) -> None:
        """Debug test: Local schema definition."""
        import pandera.pandas as pa

        schema = pa.DataFrameSchema(
            {"sequence": pa.Column(str, checks=pa.Check.str_matches("^[A-Z]+$"))}
        )
        df = pd.DataFrame({"sequence": ["ABC"]})
        schema.validate(df)

    def test_valid_dataframe(self) -> None:
        """Valid DataFrame passes validation."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQLVQSG", "EVQLVESG"],
                "label": [0, 1],
            }
        )
        # Should not raise
        validated_df = get_sequence_dataset_schema().validate(df)
        assert len(validated_df) == 2

    def test_missing_required_column_rejected(self) -> None:
        """Missing 'sequence' or 'label' raises SchemaError."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQL"],
                # Missing 'label' column
            }
        )
        with pytest.raises(SchemaError) as exc_info:
            get_sequence_dataset_schema().validate(df)
        assert "label" in str(exc_info.value).lower()

    def test_invalid_amino_acids_rejected(self) -> None:
        """Non-amino acid characters raise SchemaError."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQLZ"],  # Invalid amino acid Z
                "label": [0],
            }
        )
        with pytest.raises(SchemaError) as exc_info:
            get_sequence_dataset_schema().validate(df)

        message = str(exc_info.value)
        assert "valid_amino_acids" in message

    def test_gap_characters_rejected(self) -> None:
        """Gap characters (-, *, .) raise SchemaError."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQL-VQS"],  # Gap character
                "label": [0],
            }
        )
        with pytest.raises(SchemaError):
            get_sequence_dataset_schema().validate(df)

    def test_invalid_label_rejected(self) -> None:
        """Labels must be 0 or 1."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQL"],
                "label": [2],  # Invalid label
            }
        )
        with pytest.raises(SchemaError):
            get_sequence_dataset_schema().validate(df)

    def test_null_sequence_rejected(self) -> None:
        """Null sequences raise SchemaError."""
        df = pd.DataFrame(
            {
                "sequence": [None],
                "label": [0],
            }
        )
        with pytest.raises(SchemaError):
            get_sequence_dataset_schema().validate(df)

    def test_extra_columns_allowed(self) -> None:
        """Extra columns are allowed (strict=False)."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQL"],
                "label": [0],
                "extra_col": ["metadata"],
            }
        )
        # Should not raise
        validated_df = get_sequence_dataset_schema().validate(df)
        assert "extra_col" in validated_df.columns


class TestBoughterSchema:
    """Test Boughter-specific schema."""

    def test_valid_boughter_dataframe(self) -> None:
        """Boughter DataFrame with id passes."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQL"],
                "label": [0],
                "id": ["boughter_001"],
            }
        )
        validated_df = get_boughter_schema().validate(df)
        assert "id" in validated_df.columns


class TestJainSchema:
    """Test Jain-specific schema."""

    def test_jain_requires_id(self) -> None:
        """Jain schema requires 'id' column."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQL"],
                "label": [0],
                # Missing 'id'
            }
        )
        with pytest.raises(SchemaError) as exc_info:
            get_jain_schema().validate(df)
        assert "id" in str(exc_info.value)


class TestHarveySchema:
    """Test Harvey nanobody schema."""

    def test_harvey_with_cdrs(self) -> None:
        """Harvey schema accepts pre-annotated CDRs."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQLVQSG"],
                "label": [0],
                "cdr1": ["GYTF"],
                "cdr2": ["GIYP"],
                "cdr3": ["ARST"],
            }
        )
        validated_df = get_harvey_schema().validate(df)
        assert "cdr1" in validated_df.columns


class TestShehataSchema:
    """Test Shehata PSR schema."""

    def test_shehata_with_psr_measurement(self) -> None:
        """Shehata schema validates PSR measurements."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQL"],
                "label": [0],
                "psr_measurement": [0.3],
            }
        )
        validated_df = get_shehata_schema().validate(df)
        assert "psr_measurement" in validated_df.columns

    def test_psr_out_of_range_rejected(self) -> None:
        """PSR measurements must be 0-1."""
        df = pd.DataFrame(
            {
                "sequence": ["QVQL"],
                "label": [0],
                "psr_measurement": [1.5],  # Out of range
            }
        )
        with pytest.raises(SchemaError):
            get_shehata_schema().validate(df)
