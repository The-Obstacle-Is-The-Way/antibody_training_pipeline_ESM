import pandera.pandas as pa


# Base schema for all antibody datasets
def get_sequence_dataset_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "sequence": pa.Column(
                dtype="string",
                checks=[
                    pa.Check.str_matches(r"^[A-Z]+$", name="uppercase_letters"),
                    pa.Check.str_length(min_value=1, max_value=2000),
                    # Valid amino acids (20 standard + X for unknown)
                    pa.Check.str_matches(
                        r"^[ACDEFGHIKLMNPQRSTVWYX]+$", name="valid_amino_acids"
                    ),
                ],
                nullable=False,
                coerce=True,  # Auto-convert to string
                description="Antibody amino acid sequence (VH, VL, or VHH)",
            ),
            "label": pa.Column(
                dtype="int64",
                checks=[
                    pa.Check.isin([0, 1], name="binary_label"),
                ],
                nullable=False,
                description="Binary label: 0=specific, 1=non-specific",
            ),
        },
        strict=False,  # Allow extra columns (e.g., id, metadata)
        coerce=True,  # Auto-coerce types when possible
        name="SequenceDataset",
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


# Harvey-specific schema (VHH only, no light chain)
def get_harvey_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "sequence": pa.Column(
                dtype="string",
                checks=[
                    pa.Check.str_matches(r"^[A-Z]+$"),
                    pa.Check.str_length(min_value=1, max_value=2000),
                    pa.Check.str_matches(
                        r"^[ACDEFGHIKLMNPQRSTVWYX]+$", name="valid_amino_acids"
                    ),
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
