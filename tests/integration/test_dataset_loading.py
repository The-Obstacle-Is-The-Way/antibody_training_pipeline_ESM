"""Integration tests for Pandera + dataset loaders."""

from pathlib import Path

import pandas as pd
import pandera.backends.pandas  # noqa: F401
import pytest

from antibody_training_esm.datasets.boughter import BoughterDataset
from antibody_training_esm.datasets.jain import JainDataset


def test_boughter_dataset_validates_on_load(tmp_path: Path) -> None:
    """BoughterDataset validates DataFrame on load."""

    # Create valid CSV with VH_sequence

    csv_path = tmp_path / "boughter.csv"

    df = pd.DataFrame(
        {
            "VH_sequence": ["QVQL", "EVQL"],
            "VL_sequence": ["QIVL", "DIVL"],
            "label": [0, 1],
            "id": ["b001", "b002"],
            "subset": ["flu", "flu"],
            "num_flags": [0, 4],
        }
    )

    df.to_csv(csv_path, index=False)

    # Should not raise

    dataset = BoughterDataset()

    loaded_df = dataset.load_data(str(csv_path))

    assert len(loaded_df) == 2

    assert "sequence" in loaded_df.columns  # Created from VH_sequence


def test_invalid_boughter_csv_rejected(tmp_path: Path) -> None:
    """Invalid Boughter CSV raises SchemaError."""

    csv_path = tmp_path / "invalid.csv"

    df = pd.DataFrame(
        {
            "VH_sequence": ["QVQL123"],  # Invalid amino acids
            "label": [0],
            "id": ["b001"],
            "subset": ["flu"],
            "num_flags": [0],
        }
    )

    df.to_csv(csv_path, index=False)

    dataset = BoughterDataset()

    with pytest.raises(ValueError, match="Schema validation failed"):
        dataset.load_data(str(csv_path))


def test_jain_dataset_validates_on_load() -> None:
    """JainDataset validates canonical Jain CSV."""

    # Use actual Jain canonical file

    # Path relative to project root

    jain_path = Path("data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv")

    if jain_path.exists():
        dataset = JainDataset()

        # Use 'parity' stage to trigger filtering logic, but need sd03 for that?

        # load_data default stage is 'full' which expects jain_with_private_elisa_FULL.csv

        # If we provide full_csv_path as the canonical file, it should work if it has required columns.

        # The canonical file has 86 rows.

        # It has "id", "VH_sequence", "label".

        # JainDataset.load_data expects "elisa_flags" if stage='ssot' or 'parity'.

        # So we should use stage='full' (no filtering) or ensure columns exist.

        # Let's check what columns are in canonical file.

        # We can just try loading it as 'full' dataset.

        try:
            loaded_df = dataset.load_data(full_csv_path=str(jain_path), stage="full")

            assert len(loaded_df) > 0

        except ValueError as e:
            pytest.fail(f"Canonical Jain dataset failed validation: {e}")

    else:
        pytest.skip("Canonical Jain dataset not found")
