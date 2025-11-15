"""
Test Dataset-Specific Thresholds for ELISA vs PSR Assays

This demonstrates how to use the classifier's assay_type parameter
to get calibrated predictions for different assay types.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from antibody_training_esm.core.classifier import BinaryClassifier


def test_with_assay_type(
    model: BinaryClassifier,
    test_file: str | Path,
    dataset_name: str,
    assay_type: str,
    target_cm: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.int_], float]:
    """Test model with assay-specific threshold"""
    print(f"\n{'=' * 60}")
    print(f"Testing {dataset_name} with assay_type='{assay_type}'")
    print(f"{'=' * 60}")

    # Load data
    df = pd.read_csv(test_file, comment="#")

    # Determine which sequence column to use (dataset-dependent)
    if "vh_sequence" in df.columns:
        sequences = df["vh_sequence"].tolist()
    elif "sequence" in df.columns:
        sequences = df["sequence"].tolist()
    else:
        raise KeyError(
            f"Expected 'vh_sequence' or 'sequence' column in {test_file}, "
            f"found columns: {list(df.columns)}"
        )
    y_true = df["label"].to_numpy(dtype=int)

    print(f"  Dataset size: {len(sequences)} sequences")
    print(f"  Class distribution: {pd.Series(y_true).value_counts().to_dict()}")

    # Extract embeddings
    print("  Extracting ESM-1v embeddings...")
    X_embeddings = model.embedding_extractor.extract_batch_embeddings(sequences)

    # Run inference with assay-specific threshold
    print(f"  Running inference with assay_type='{assay_type}'...")
    y_pred = model.predict(X_embeddings, assay_type=assay_type)

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print("\n  Results:")
    print(f"    Confusion matrix: {cm.tolist()}")
    print(f"    Accuracy: {acc * 100:.1f}%")

    print("\n  Novo benchmark:")
    print(f"    Confusion matrix: {target_cm}")

    # Compare
    diff = abs(cm - target_cm).sum()
    print(f"\n  Difference from Novo: {diff} (sum of absolute differences)")
    if diff < 5:
        print("  ✓ Close match to Novo!")
    elif diff < 10:
        print("  ~ Reasonable match to Novo")
    else:
        print("  ✗ Significant difference from Novo")

    return cm, float(acc)


def main() -> int:
    print("=" * 60)
    print("TESTING ASSAY-SPECIFIC THRESHOLDS")
    print("=" * 60)

    # Load trained model
    model_path = "models/boughter_vh_esm1v_logreg.pkl"
    print(f"\nLoading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("  Model loaded successfully!")

    novo_jain = np.array([[40, 19], [10, 17]])  # 66.28% accuracy (Novo benchmark)
    # Note: Shehata is an external test set with expected performance, not a Novo benchmark
    novo_shehata = np.array([[229, 162], [2, 5]])  # ~58.8% accuracy (expected)

    # Test Jain with ELISA threshold (default 0.5)
    print("\n" + "=" * 60)
    print("TEST 1: Jain Dataset (ELISA assay)")
    print("=" * 60)
    jain_cm, jain_acc = test_with_assay_type(
        model,
        "data/test/jain/canonical/jain_86_novo_parity.csv",
        "Jain",
        "ELISA",
        novo_jain,
    )

    # Test Shehata with PSR threshold (0.549)
    print("\n" + "=" * 60)
    print("TEST 2: Shehata Dataset (PSR assay)")
    print("=" * 60)
    shehata_cm, shehata_acc = test_with_assay_type(
        model,
        "data/test/shehata/fragments/VH_only_shehata.csv",
        "Shehata",
        "PSR",
        novo_shehata,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nUsing assay-specific thresholds:")
    print(f"  Jain (ELISA):   {jain_acc * 100:.1f}% accuracy")
    print(f"  Shehata (PSR):  {shehata_acc * 100:.1f}% accuracy")

    print("\nBenchmarks:")
    print("  Jain (ELISA):   66.28% accuracy (Novo benchmark)")
    print("  Shehata (PSR):  ~58.8% accuracy (expected with PSR threshold)")

    print("\n" + "=" * 60)
    print("HOW TO USE IN YOUR CODE:")
    print("=" * 60)
    print(
        """
# For ELISA-based datasets (Jain, Boughter):
predictions = model.predict(X_embeddings, assay_type='ELISA')

# For PSR-based datasets (Shehata, Harvey):
predictions = model.predict(X_embeddings, assay_type='PSR')

# For custom threshold:
predictions = model.predict(X_embeddings, threshold=0.6)

# Default behavior (threshold=0.5):
predictions = model.predict(X_embeddings)
"""
    )

    print("=" * 60)
    print("Testing completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
