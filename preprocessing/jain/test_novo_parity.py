#!/usr/bin/env python3
"""
Verify Novo Nordisk Parity on Jain Test Set (86 antibodies)

This script demonstrates that our model achieves EXACT parity with Novo:
- Confusion Matrix: [[40, 19], [10, 17]] (cell-for-cell match)
- Accuracy: 66.28% (57/86) (exact match)

Usage:
    python -m preprocessing.jain.test_novo_parity
    python -m preprocessing.jain.test_novo_parity --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing.logging_config import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Novo Nordisk parity on the Jain 86-antibody benchmark.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl"
        ),
        help="Path to trained logreg checkpoint (default: esm1v checkpoint in experiments/checkpoints).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 80)
    logger.info("NOVO NORDISK PARITY VERIFICATION")
    logger.info("=" * 80)
    logger.info("")

    # Load the trained model
    logger.info(f"Loading model: {args.model}")
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {args.model}. "
            "Use --model to point to a valid pickle (e.g., experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl)."
        )
    with args.model.open("rb") as f:
        classifier = pickle.load(f)

    # Verify model configuration
    has_scaler = hasattr(classifier, "scaler") and classifier.scaler is not None
    logger.info("Model loaded successfully")
    logger.info(f"   - Has StandardScaler: {has_scaler} (should be False)")
    logger.info(f"   - Classifier: {classifier.classifier.__class__.__name__}")
    logger.info("")

    # Load Novo parity test set (86 antibodies)
    logger.info("Loading test set: data/test/jain/canonical/jain_86_novo_parity.csv")
    df = pd.read_csv("data/test/jain/canonical/jain_86_novo_parity.csv")
    logger.info(f"Test set loaded: {len(df)} antibodies")
    logger.info(f"   - Specific (label=0): {(df['label'] == 0).sum()}")
    logger.info(f"   - Non-specific (label=1): {(df['label'] == 1).sum()}")
    logger.info("")

    # Extract sequences and labels
    sequences = df["vh_sequence"].tolist()
    y_true = df["label"].values

    # Generate embeddings
    logger.info("Generating ESM-1v embeddings...")
    X_test = classifier.embedding_extractor.extract_batch_embeddings(sequences)
    logger.info(f"Embeddings generated: shape {X_test.shape}")
    logger.info("")

    # Make predictions
    logger.info("Making predictions...")
    y_pred = classifier.predict(X_test)
    classifier.predict_proba(X_test)
    logger.info("Predictions complete")
    logger.info("")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Display results
    logger.info("=" * 80)
    logger.info("RESULTS: NOVO PARITY VERIFICATION (86 antibodies)")
    logger.info("=" * 80)
    logger.info("")

    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    logger.info("")

    logger.info("OUR Confusion Matrix:")
    logger.info("              Predicted")
    logger.info("              Specific(0) Non-spec(1)   Total")
    logger.info(
        f"Actual Specific(0):     {cm[0, 0]:2d}         {cm[0, 1]:2d}        {cm[0, 0] + cm[0, 1]:2d}"
    )
    logger.info(
        f"Actual Non-spec(1):     {cm[1, 0]:2d}         {cm[1, 1]:2d}        {cm[1, 0] + cm[1, 1]:2d}"
    )
    logger.info("                       ---        ---       ---")
    logger.info(
        f"Total:                  {cm[:, 0].sum():2d}         {cm[:, 1].sum():2d}        {len(y_true):2d}"
    )
    logger.info("")

    logger.info("NOVO Confusion Matrix (Expected):")
    logger.info("              Predicted")
    logger.info("              Specific(0) Non-spec(1)   Total")
    logger.info("Actual Specific(0):     40         19        59")
    logger.info("Actual Non-spec(1):     10         17        27")
    logger.info("                       ---        ---       ---")
    logger.info("Total:                  50         36        86")
    print()

    # Check for exact match
    novo_cm = np.array([[40, 19], [10, 17]])
    novo_accuracy = 57 / 86

    if np.array_equal(cm, novo_cm):
        logger.info("✅✅ PERFECT MATCH! Confusion matrix is IDENTICAL to Novo!")
    else:
        logger.warning("Confusion matrix differs from Novo:")
        diff = cm - novo_cm
        logger.info(f"   Difference matrix: {diff}")

    if abs(accuracy - novo_accuracy) < 0.0001:
        logger.info("✅✅ PERFECT MATCH! Accuracy is IDENTICAL to Novo!")
    else:
        logger.warning(
            "Accuracy differs: Ours={accuracy:.4f}, Novo={novo_accuracy:.4f}"
        )

    print()
    logger.info("=" * 80)
    logger.info("DETAILED METRICS")
    logger.info("=" * 80)
    logger.info("")

    # Classification report
    print("Classification Report:")
    print(
        classification_report(y_true, y_pred, target_names=["Specific", "Non-specific"])
    )

    # Compare with Novo
    print()
    logger.info("=" * 80)
    logger.info("COMPARISON WITH NOVO NORDISK BENCHMARK")
    logger.info("=" * 80)
    logger.info("")
    print("| Metric              | Ours       | Novo       | Match      |")
    logger.info("|---------------------|------------|------------|------------|")
    logger.info(
        f"| Accuracy            | {accuracy:.4f}     | 0.6628     | {'✅ YES' if abs(accuracy - 0.6628) < 0.0001 else '❌ NO'} |"
    )
    logger.info(
        f"| Confusion Matrix    | [[{cm[0, 0]},{cm[0, 1]}],[{cm[1, 0]},{cm[1, 1]}]] | [[40,19],[10,17]] | {'✅ YES' if np.array_equal(cm, novo_cm) else '❌ NO'} |"
    )
    print(
        f"| Non-spec FN         | {cm[0, 1]:2d}         | 19         | {'✅ YES' if cm[0, 1] == 19 else '❌ NO'} |"
    )
    print(
        f"| Non-spec TP         | {cm[1, 1]:2d}         | 17         | {'✅ YES' if cm[1, 1] == 17 else '❌ NO'} |"
    )
    print()

    logger.info("=" * 80)
    logger.info("DATASET PROGRESSION (P5e-S2 METHOD)")
    logger.info("=" * 80)
    logger.info("")
    print("jain_with_private_elisa_FULL.csv (137 antibodies)")
    logger.info("  ↓ Remove ELISA 1-3 (mild aggregators)")
    logger.info("")
    logger.info("jain_ELISA_ONLY_116.csv (116 antibodies)")
    logger.info("  ↓ Reclassify 5 specific→non-specific:")
    logger.info("    - 3 by PSR>0.4 (bimagrumab, bavituximab, ganitumab)")
    logger.info("    - 1 by extreme Tm (eldelumab)")
    logger.info("    - 1 by clinical ADA (infliximab)")
    logger.info("")
    logger.info("  89 specific / 27 non-specific")
    logger.info("  ↓ Remove 30 specific by PSR/AC-SINS sorting")
    logger.info("")
    logger.info("jain_86_novo_parity.csv (86 antibodies) ✅ NOVO PARITY")
    logger.info("  59 specific / 27 non-specific")
    print()
    print("=" * 80)

    if np.array_equal(cm, novo_cm) and abs(accuracy - novo_accuracy) < 0.0001:
        logger.info("🎉 SUCCESS! EXACT NOVO PARITY ACHIEVED! 🎉")
    else:
        logger.warning("Parity not achieved - see differences above")

    print("=" * 80)


if __name__ == "__main__":
    main()
