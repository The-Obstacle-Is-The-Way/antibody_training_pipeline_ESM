from __future__ import annotations

import re

import pandas as pd
import pandera.backends.pandas  # noqa: F401  # registers pandas backend
import pandera.pandas as pa

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")
_UPPERCASE_PATTERN = re.compile(r"^[A-Z]+$")
_NO_GAP_PATTERN = re.compile(r"^[^*.-]+$")


def _regex_check(pattern: re.Pattern[str], name: str) -> pa.Check:
    return pa.Check(
        lambda series: bool(series.str.match(pattern).fillna(False).all()),
        name=name,
    )


def _length_check(min_value: int, max_value: int, name: str) -> pa.Check:
    return pa.Check(
        lambda series: bool(series.str.len().between(min_value, max_value).all()),
        name=name,
    )


def _amino_acid_check(series: pd.Series) -> bool:
    return bool(series.dropna().map(lambda seq: set(str(seq)).issubset(VALID_AA)).all())


def _no_gap_check(series: pd.Series) -> bool:
    return bool(series.str.match(_NO_GAP_PATTERN).fillna(False).all())


# Base schema for all antibody datasets (production: strict, no NaN labels)
def get_sequence_dataset_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "sequence": pa.Column(
                dtype="string",
                checks=[
                    _regex_check(_UPPERCASE_PATTERN, name="uppercase_letters"),
                    _length_check(1, 2000, name="length_1_2000"),
                    pa.Check(_amino_acid_check, name="valid_amino_acids"),
                    pa.Check(_no_gap_check, name="no_gap_characters"),
                ],
                nullable=False,
                coerce=True,  # Auto-convert to string
                description="Antibody amino acid sequence (VH, VL, or VHH)",
            ),
            "label": pa.Column(
                dtype="int64",
                checks=[
                    pa.Check(
                        lambda series: series.isin([0, 1]).all(),
                        name="binary_label",
                    ),
                ],
                nullable=False,
                description="Binary label: 0=specific, 1=non-specific",
            ),
        },
        strict=False,  # Allow extra columns (e.g., id, metadata)
        coerce=True,  # Auto-coerce types when possible
        name="SequenceDataset",
    )


# Preprocessing schema (allows nullable labels for held-out/intermediate data)
def get_preprocessing_schema() -> pa.DataFrameSchema:
    """
    Schema for preprocessing intermediate files (e.g., Boughter annotated/).

    Allows nullable labels for sequences held out due to quality flags.
    For production training/testing, use get_sequence_dataset_schema() instead.
    """
    return pa.DataFrameSchema(
        columns={
            "sequence": pa.Column(
                dtype="string",
                checks=[
                    _regex_check(_UPPERCASE_PATTERN, name="uppercase_letters"),
                    _length_check(1, 2000, name="length_1_2000"),
                    pa.Check(_amino_acid_check, name="valid_amino_acids"),
                    pa.Check(_no_gap_check, name="no_gap_characters"),
                ],
                nullable=False,
                coerce=True,
                description="Antibody amino acid sequence (VH, VL, or VHH)",
            ),
            "label": pa.Column(
                dtype="float64",  # float64 to handle NaN
                checks=[
                    # Only check non-null values are 0 or 1
                    pa.Check(
                        lambda series: series.dropna().isin([0, 1, 0.0, 1.0]).all(),
                        name="binary_label_when_present",
                    ),
                ],
                nullable=True,  # Allow NaN for held-out sequences
                coerce=True,
                description="Binary label: 0=specific, 1=non-specific (nullable for held-out)",
            ),
        },
        strict=False,
        coerce=True,
        name="PreprocessingDataset",
    )


# Boughter-specific schema (extends base)
def get_boughter_schema() -> pa.DataFrameSchema:
    return get_sequence_dataset_schema().add_columns(
        {
            "id": pa.Column(
                dtype="string",
                nullable=True,
                required=False,
                description="Antibody identifier",
            ),
            # Boughter has additional metadata columns
            "vh_sequence": pa.Column(
                dtype="string",
                nullable=True,
                required=False,
                description="Heavy chain variable domain (if paired)",
            ),
        }
    )


# Jain-specific schema
def get_jain_schema() -> pa.DataFrameSchema:
    return get_sequence_dataset_schema().add_columns(
        {
            "id": pa.Column(
                dtype="string",
                nullable=False,
                checks=[
                    pa.Check.str_length(min_value=1),
                ],
                description="Antibody INN name (required for Jain)",
            ),
            "vh_sequence": pa.Column(
                dtype="string",
                nullable=True,
                required=False,
                description="VH sequence (Jain has full paired data)",
            ),
            "vl_sequence": pa.Column(
                dtype="string",
                nullable=True,
                required=False,
                description="VL sequence",
            ),
        }
    )


# Jain preprocessing schema (allows nullable labels for full stage)
def get_jain_preprocessing_schema() -> pa.DataFrameSchema:
    return get_preprocessing_schema().add_columns(
        {
            "id": pa.Column(
                dtype="string",
                nullable=False,
                checks=[
                    pa.Check.str_length(min_value=1),
                ],
                description="Antibody INN name (required for Jain)",
            ),
            "vh_sequence": pa.Column(
                dtype="string",
                nullable=True,
                required=False,
                description="VH sequence (Jain has full paired data)",
            ),
            "vl_sequence": pa.Column(
                dtype="string",
                nullable=True,
                required=False,
                description="VL sequence",
            ),
        }
    )


# Harvey-specific schema (VHH only, no light chain)
def get_harvey_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "sequence": pa.Column(
                dtype="string",
                checks=[
                    _regex_check(_UPPERCASE_PATTERN, name="uppercase_letters"),
                    _length_check(1, 2000, name="length_1_2000"),
                    pa.Check(_amino_acid_check, name="valid_amino_acids"),
                    pa.Check(_no_gap_check, name="no_gap_characters"),
                ],
                nullable=False,
                description="Nanobody VHH sequence",
            ),
            "label": pa.Column(
                dtype="int64",
                checks=[pa.Check.isin([0, 1])],
                nullable=False,
            ),
            # Harvey has pre-annotated CDRs
            "cdr1": pa.Column(dtype="string", nullable=True, required=False),
            "cdr2": pa.Column(dtype="string", nullable=True, required=False),
            "cdr3": pa.Column(dtype="string", nullable=True, required=False),
        },
        strict=False,
        coerce=True,
        name="HarveyNanobodyDataset",
    )


# Shehata-specific schema (paired antibodies with PSR measurements)
def get_shehata_schema() -> pa.DataFrameSchema:
    return get_sequence_dataset_schema().add_columns(
        {
            "psr_measurement": pa.Column(
                dtype="float64",
                checks=[
                    pa.Check.in_range(min_value=0.0, max_value=1.0),
                ],
                nullable=True,
                required=False,
                description="PSR assay measurement (0-1 range)",
            ),
            "vh_sequence": pa.Column(
                dtype="string",
                nullable=True,
                required=False,
            ),
            "vl_sequence": pa.Column(
                dtype="string",
                nullable=True,
                required=False,
            ),
        }
    )
