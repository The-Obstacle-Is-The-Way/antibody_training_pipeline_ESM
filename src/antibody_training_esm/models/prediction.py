from typing import Literal

from pydantic import BaseModel, Field, field_validator

AssayType = Literal["ELISA", "PSR"]


class PredictionRequest(BaseModel):
    """
    Single sequence prediction request.

    Validates amino acid sequence and optional parameters.
    """

    sequence: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Antibody amino acid sequence (VH or VL)",
        examples=["QVQLVQSGAEVKKPGASVKVSCKASGYTFT..."],
    )

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold (0-1)",
    )

    assay_type: AssayType | None = Field(
        default=None,
        description="Assay type for calibrated thresholds",
    )

    @field_validator("sequence")
    @classmethod
    def validate_amino_acids(cls, v: str) -> str:
        """Validate sequence contains only valid amino acids."""
        # Clean whitespace
        cleaned = v.strip().upper()

        if not cleaned:
            raise ValueError("Sequence cannot be empty after cleaning")

        # Standard 20 amino acids + X (unknown)
        valid_chars = set("ACDEFGHIKLMNPQRSTVWYX")
        invalid_chars = set(cleaned) - valid_chars

        if invalid_chars:
            raise ValueError(
                f"Invalid characters found: {', '.join(sorted(invalid_chars))}. "
                f"Only standard amino acids (ACDEFGHIKLMNPQRSTVWY) and X are allowed."
            )

        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMHWVRQAPGQGLEWMG",
                    "threshold": 0.5,
                    "assay_type": "ELISA",
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request for multiple sequences.

    Supports both inline lists and file uploads (future).
    """

    sequences: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,  # Batch size limit
        description="List of antibody sequences",
    )

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    assay_type: AssayType | None = None

    @field_validator("sequences")
    @classmethod
    def validate_all_sequences(cls, v: list[str]) -> list[str]:
        """Validate each sequence in batch."""
        cleaned = []
        errors = []

        for i, seq in enumerate(v):
            try:
                # Reuse PredictionRequest validator
                request = PredictionRequest(sequence=seq)
                cleaned.append(request.sequence)
            except ValueError as e:
                errors.append(f"Sequence {i + 1}: {e}")

        if errors:
            raise ValueError("Batch validation failed:\n" + "\n".join(errors))

        return cleaned


class PredictionResult(BaseModel):
    """
    Prediction result for a single sequence.

    Standardizes output format across CLI, Gradio, and future APIs.
    """

    sequence: str = Field(..., description="Input sequence (cleaned)")

    prediction: Literal["specific", "non-specific"] = Field(
        ...,
        description="Classification result",
    )

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of non-specificity (class 1)",
    )

    threshold: float = Field(
        ...,
        description="Threshold used for classification",
    )

    assay_type: AssayType | None = Field(
        default=None,
        description="Assay type if calibrated threshold was used",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH...",
                    "prediction": "specific",
                    "probability": 0.23,
                    "threshold": 0.5,
                    "assay_type": "ELISA",
                }
            ]
        }
    }
