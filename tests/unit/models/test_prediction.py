"""Unit tests for prediction models."""

import pytest
from pydantic import ValidationError

from antibody_training_esm.models.prediction import (
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResult,
)


class TestPredictionRequest:
    """Test suite for PredictionRequest validation."""

    def test_valid_sequence(self) -> None:
        """Valid sequence passes validation."""
        request = PredictionRequest(sequence="QVQLVQSGAEVKKPGASVKVSCKASGYTFT")
        assert request.sequence == "QVQLVQSGAEVKKPGASVKVSCKASGYTFT"
        assert request.threshold == 0.5  # default
        assert request.assay_type is None  # default

    def test_sequence_cleaned(self) -> None:
        """Whitespace is stripped and uppercased."""
        request = PredictionRequest(sequence="  qvql  ")
        assert request.sequence == "QVQL"

    def test_empty_sequence_rejected(self) -> None:
        """Empty sequence raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(sequence="")

        errors = exc_info.value.errors()
        # Pydantic v2 min_length error message
        assert any("at least 1 character" in str(e).lower() for e in errors)

        # Test whitespace-only sequence (passes min_length, fails custom validator)
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(sequence="   ")

        errors = exc_info.value.errors()
        assert any("empty" in str(e).lower() for e in errors)

    def test_invalid_amino_acids_rejected(self) -> None:
        """Non-amino acid characters raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(sequence="QVQL123")

        errors = exc_info.value.errors()
        assert any("invalid characters" in str(e).lower() for e in errors)

    def test_gap_characters_rejected(self) -> None:
        """Gap characters (-, *, .) are rejected."""
        for gap_char in ["-", "*", "."]:
            with pytest.raises(ValidationError):
                PredictionRequest(sequence=f"QVQL{gap_char}VQS")

    def test_threshold_out_of_range_rejected(self) -> None:
        """Threshold must be 0-1."""
        with pytest.raises(ValidationError):
            PredictionRequest(sequence="QVQL", threshold=1.5)

        with pytest.raises(ValidationError):
            PredictionRequest(sequence="QVQL", threshold=-0.1)

    def test_invalid_assay_type_rejected(self) -> None:
        """Only ELISA and PSR are valid."""
        with pytest.raises(ValidationError):
            PredictionRequest(
                sequence="QVQL",
                assay_type="INVALID",  # type: ignore
            )

    def test_sequence_length_limits(self) -> None:
        """Sequences longer than 2000 are rejected."""
        long_seq = "A" * 2001
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(sequence=long_seq)

        errors = exc_info.value.errors()
        assert any("max_length" in str(e).lower() for e in errors)


class TestBatchPredictionRequest:
    """Test suite for BatchPredictionRequest validation."""

    def test_valid_batch(self) -> None:
        """Valid batch passes validation."""
        request = BatchPredictionRequest(sequences=["QVQL", "EVQL", "DIQM"])
        assert len(request.sequences) == 3

    def test_empty_batch_rejected(self) -> None:
        """Empty batch list is rejected."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(sequences=[])

    def test_batch_size_limit(self) -> None:
        """Batches >1000 are rejected."""
        large_batch = ["QVQL"] * 1001
        with pytest.raises(ValidationError):
            BatchPredictionRequest(sequences=large_batch)

    def test_invalid_sequence_in_batch_rejected(self) -> None:
        """One invalid sequence fails entire batch."""
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(sequences=["QVQL", "INVALID123", "EVQL"])

        error_msg = str(exc_info.value)
        assert "Sequence 2" in error_msg  # 1-indexed


class TestPredictionResult:
    """Test suite for PredictionResult output."""

    def test_valid_result(self) -> None:
        """Valid result constructs correctly."""
        result = PredictionResult(
            sequence="QVQL",
            prediction="specific",
            probability=0.23,
            threshold=0.5,
        )
        assert result.prediction == "specific"
        assert result.probability == 0.23

    def test_invalid_prediction_rejected(self) -> None:
        """Only 'specific' or 'non-specific' allowed."""
        with pytest.raises(ValidationError):
            PredictionResult(
                sequence="QVQL",
                prediction="unknown",  # type: ignore
                probability=0.5,
                threshold=0.5,
            )

    def test_probability_out_of_range_rejected(self) -> None:
        """Probability must be 0-1."""
        with pytest.raises(ValidationError):
            PredictionResult(
                sequence="QVQL",
                prediction="specific",
                probability=1.5,
                threshold=0.5,
            )
